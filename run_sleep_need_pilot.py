"""CLI runner for EXP-SLEEP-NEED-001 with strict schema validation and two-stage execution."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pqt_qcat.memory.exp_sleep_need_001 import (
    validate_config,
    generate_task_panel,
    evaluate_joint_positive_control,
    run_stage_experiment,
    generate_russian_report,
)


def compute_manifest_sha256(directory: Path) -> dict[str, str]:
    """Compute SHA-256 for all files in directory, omitting manifest itself."""
    manifest = {}
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f == "manifest_sha256.json":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, directory).replace("\\", "/")
            h = hashlib.sha256()
            with open(full_path, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            manifest[rel_path] = h.hexdigest()
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Run EXP-SLEEP-NEED-001 two-stage experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to protocol configuration JSON file.")
    parser.add_argument("--stage", choices=["all", "wake_only", "sleep_only"], default="all", help="Execution stage.")
    parser.add_argument("--force-sleep", action="store_true", help="Force sleep stage execution even if wake criteria are TASK_NOT_INFORMATIVE.")
    parser.add_argument("--output", type=Path, default=None, help="Custom output directory.")
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 1. Validate configuration schema and parameters
    validate_config(config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SLEEP_NEED_RUN_{timestamp}"
    run_dir = args.output or (root_dir / "results" / "exploratory" / "SLEEP_NEED_001" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = run_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    # Save exact config
    with open(protocol_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Git metadata
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(root_dir)).stdout.strip()
    except Exception:
        git_sha = "UNKNOWN"
    with open(protocol_dir / "source_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{git_sha}\n")

    try:
        git_diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True, cwd=str(root_dir)).stdout
    except Exception:
        git_diff = ""
    with open(protocol_dir / "git_diff.patch", "w", encoding="utf-8") as f:
        f.write(git_diff)

    # Save environment dependencies
    import numpy as np
    import scipy
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    deps = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "git_commit": git_sha,
        "pip_freeze": pip_freeze.strip().split("\n"),
    }
    with open(protocol_dir / "dependencies.json", "w", encoding="utf-8") as f:
        json.dump(deps, f, indent=2)

    # Generate and publish task panel
    panel_maps = generate_task_panel(
        generator_seed=config["panel"]["generator_seed"],
        total_maps=config["panel"]["total_distinct_mappings"],
    )
    with open(run_dir / "panel_mappings.json", "w", encoding="utf-8") as f:
        json.dump(panel_maps, f, indent=2)

    cal_seeds = config["calibration_seeds"]
    cal_maps = panel_maps[:config["panel"]["calibration_first_n"]]

    print("=================================================================")
    print(f"EXP-SLEEP-NEED-001 RUNNER | Run ID: {run_id}")
    print(f"Seeds: {cal_seeds} | Panel mappings: {len(cal_maps)} (0..{len(cal_maps)-1})")
    print("=================================================================")

    # Stage 1: Pre-Sleep Joint Positive Control
    print("\n--- PHASE 1: JOINT POSITIVE CONTROL ---")
    joint_control_res = evaluate_joint_positive_control(
        seeds=cal_seeds,
        panel_maps=cal_maps,
        epochs=config["wake"].get("joint_positive_control_epochs", 25),
        eta_K=config["wake"].get("eta", 0.8),
        delta_core_max=config["wake"].get("coordinate_clip", 0.5),
        ticks=config["wake"].get("ticks_per_query", 8),
    )
    with open(run_dir / "joint_positive_control.json", "w", encoding="utf-8") as f:
        json.dump(joint_control_res, f, indent=2)
    print(f"Joint Positive Control Accuracy: {joint_control_res['mean_joint_accuracy']:.4f} (95% CI: {joint_control_res['ci_95']}) | Passed (>=0.75): {joint_control_res['passed']}")

    # Stage 1B: Wake-Only Calibration
    print("\n--- PHASE 2: WAKE CALIBRATION (24 histories) ---")
    cal_dir = run_dir / "calibration"
    
    # Decide whether to execute sleep
    run_sleep_now = (args.stage == "all" and args.force_sleep) or (args.stage == "sleep_only")
    
    wake_cal_contrasts = run_stage_experiment(
        stage_name="calibration",
        seeds=cal_seeds,
        panel_maps=cal_maps,
        output_stage_dir=cal_dir,
        config=config,
        execute_sleep=run_sleep_now,
    )
    generate_russian_report("calibration", wake_cal_contrasts, cal_dir, n_branches=len(cal_seeds) * len(cal_maps) * 2 * (4 if run_sleep_now else 1))

    # Evaluate Preconditions for Sleep
    i_contrast = wake_cal_contrasts["interference_contrast_pre_tail"]
    passed_interference = i_contrast["passed_threshold"]
    passed_joint = joint_control_res["passed"]

    can_proceed_to_sleep = passed_interference and passed_joint
    
    print("\n=================================================================")
    print("GATE EVALUATION FOR SLEEP STAGE:")
    print(f"  1. Joint Positive Control (>=0.75): {passed_joint} ({joint_control_res['mean_joint_accuracy']:.4f})")
    print(f"  2. Pre-tail Interference Contrast (>=0.10): {passed_interference} ({i_contrast['mean']:+.4f}, 95% CI: {i_contrast['ci_95']})")
    print(f"  Overall Sleep Gate Passed: {can_proceed_to_sleep}")
    print("=================================================================")

    if not can_proceed_to_sleep and not args.force_sleep:
        print("\n[GATE BLOCKED]: Preconditions for sleep stage not met.")
        print("Per Protocol Section 10: Pilot completes with TASK_NOT_INFORMATIVE. Sleep stage skipped to save compute.")
        final_status = "TASK_NOT_INFORMATIVE"
    elif args.stage == "wake_only":
        print("\n[WAKE-ONLY]: User specified --stage wake_only. Halting before sleep stage.")
        final_status = "WAKE_CALIBRATION_COMPLETE"
    else:
        if not run_sleep_now:
            print("\n--- PHASE 3: SLEEP CONSOLIDATION STAGE (192 sleep branches) ---")
            sleep_contrasts = run_stage_experiment(
                stage_name="calibration",
                seeds=cal_seeds,
                panel_maps=cal_maps,
                output_stage_dir=cal_dir,
                config=config,
                execute_sleep=True,
            )
            generate_russian_report("calibration", sleep_contrasts, cal_dir, n_branches=len(cal_seeds) * len(cal_maps) * 2 * 4)
            final_status = sleep_contrasts.get("D_primary_contrast", {}).get("verdict", "SLEEP_COMPLETE")
        else:
            final_status = wake_cal_contrasts.get("D_primary_contrast", {}).get("verdict", "SLEEP_COMPLETE")

    status_summary = {
        "run_id": run_id,
        "date": datetime.datetime.now().isoformat(),
        "git_commit": git_sha,
        "joint_positive_control": joint_control_res,
        "interference_contrast_pre_tail": i_contrast,
        "sleep_gate_passed": can_proceed_to_sleep,
        "forced_sleep": args.force_sleep,
        "final_status": final_status,
    }
    with open(run_dir / "status_summary.json", "w", encoding="utf-8") as f:
        json.dump(status_summary, f, indent=2)

    # Compute full manifest
    manifest = compute_manifest_sha256(run_dir)
    with open(protocol_dir / "manifest_sha256.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nCompleted run {run_id}. Manifest has {len(manifest)} entries.")


if __name__ == "__main__":
    main()
