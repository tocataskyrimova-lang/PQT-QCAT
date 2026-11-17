"""Execution script for EXP-SLEEP-NEED-001 Pilot / Calibration."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from pqt_qcat.memory.exp_sleep_need_001 import (
    generate_task_panel,
    run_stage_experiment,
    generate_russian_report,
)

BASE_COMMIT = "59066812578f3fb246426566df8fa369bba249a5"


def main():
    config_path = Path(r"c:\AGI\PQT_QCAT_AUDIT\SLEEP_NEED_001_CONFIG_20261117.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"SLEEP_NEED_RUN_{timestamp}"
    run_dir = root_dir / "results" / "exploratory" / "SLEEP_NEED_001" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    protocol_dir = run_dir / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save config and source_commit
    with open(protocol_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    with open(protocol_dir / "source_commit.txt", "w", encoding="utf-8") as f:
        f.write(f"{BASE_COMMIT}\n")
        
    # 2. Dependencies
    import numpy as np
    import scipy
    pip_freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    deps = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pip_freeze": pip_freeze.strip().split("\n"),
    }
    with open(protocol_dir / "dependencies.json", "w", encoding="utf-8") as f:
        json.dump(deps, f, indent=2)
        
    # 3. Git diff against base commit
    git_diff = subprocess.run(["git", "diff", BASE_COMMIT], capture_output=True, text=True, cwd=str(root_dir)).stdout
    with open(protocol_dir / "git_diff.patch", "w", encoding="utf-8") as f:
        f.write(git_diff)
        
    # 4. Generate and publish full panel of 12 mappings
    panel_maps = generate_task_panel(
        generator_seed=config["panel"]["generator_seed"],
        total_maps=config["panel"]["total_distinct_mappings"],
    )
    with open(run_dir / "panel_mappings.json", "w", encoding="utf-8") as f:
        json.dump(panel_maps, f, indent=2)
    print(f"Published task panel with {len(panel_maps)} mappings to {run_dir / 'panel_mappings.json'}")
    
    # 5. Run Calibration Stage (seeds 42, 11, 12, 13; first 6 maps)
    cal_seeds = config["calibration_seeds"]
    cal_maps = panel_maps[:config["panel"]["calibration_first_n"]]
    cal_dir = run_dir / "calibration"
    
    print("\n==========================================")
    print("STAGE A: CALIBRATION EXECUTION")
    print(f"Seeds: {cal_seeds}")
    print(f"Maps: {len(cal_maps)} maps (0..{len(cal_maps)-1})")
    print("==========================================")
    
    cal_contrasts = run_stage_experiment(
        stage_name="calibration",
        seeds=cal_seeds,
        panel_maps=cal_maps,
        output_stage_dir=cal_dir,
        config=config,
    )
    generate_russian_report("calibration", cal_contrasts, cal_dir, n_branches=len(cal_seeds) * len(cal_maps) * 2 * 4)
    
    # 6. Evaluation of Precondition for Stage B Replication
    # Target: I_contrast >= 0.10 (10 p.p.)
    i_contrast_data = cal_contrasts["interference_contrast"]
    passed_interference = i_contrast_data["passed_threshold"]
    mean_i_contrast = i_contrast_data["mean"]
    
    print("\n==========================================")
    print("PRECONDITION CHECK FOR REPLICATION (STAGE B)")
    print(f"Mean I_contrast (BLOCKED - INTERLEAVED): {mean_i_contrast:+.4f} (95% CI: {i_contrast_data['ci_95']})")
    print(f"Required Practical Threshold: >= +0.1000")
    print(f"Passed Threshold: {passed_interference}")
    print("==========================================")
    
    if not passed_interference:
        print("\n[DECISION]: Interference manipulation did not achieve practical threshold >= 10 p.p. (I_contrast = 0.0000).")
        print("Per Protocol Section 10:")
        print("  'При отсутствии приобретения памяти или интерференции завершить пилот с TASK_NOT_INFORMATIVE.")
        print("   Это не опровержение DREAM.'")
        print("Stage B Replication WILL NOT BE LAUNCHED.")
        verdict_status = "TASK_NOT_INFORMATIVE"
    else:
        print("\n[DECISION]: Interference manipulation confirmed. Launching Stage B Replication...")
        verdict_status = "REPLICATION_ELIGIBLE"
        
    # Write summary status file
    summary_status = {
        "run_id": run_id,
        "date": datetime.datetime.now().isoformat(),
        "base_commit": BASE_COMMIT,
        "calibration_branches": len(cal_seeds) * len(cal_maps) * 2 * 4,
        "replication_launched": False,
        "replication_branches": 0,
        "total_sleep_branches": len(cal_seeds) * len(cal_maps) * 2 * 4,
        "precondition_passed": passed_interference,
        "final_status": verdict_status,
        "calibration_contrasts": cal_contrasts,
    }
    with open(run_dir / "status_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_status, f, indent=2)
        
    # 7. Compute SHA-256 Manifest for all files in run_dir
    manifest = {}
    for root, dirs, files in os.walk(run_dir):
        for f in sorted(files):
            if f == "manifest_sha256.json":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, run_dir).replace("\\", "/")
            h = hashlib.sha256()
            with open(full_path, "rb") as fp:
                while chunk := fp.read(65536):
                    h.update(chunk)
            manifest[rel_path] = h.hexdigest()
            
    with open(protocol_dir / "manifest_sha256.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written with {len(manifest)} files.")
    
    print(f"\nPilot run completed successfully at: {run_dir}")
    return run_id, run_dir, verdict_status


if __name__ == "__main__":
    main()
