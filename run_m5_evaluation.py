"""Milestone M5 Evaluation Runner (LIFE-002 Independent Evaluator).

Usage:
  python run_m5_evaluation.py --calibration-only [--episodes N]
  python run_m5_evaluation.py --seed 42 [--episodes N]
  python run_m5_evaluation.py --generate-manifest
  python run_m5_evaluation.py --verify-manifest
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from pqt_qcat.evaluator.independent_evaluator import (
    IndependentEvaluator,
    EvaluationConfig,
    SEALED_SEEDS,
    CALIBRATION_SEEDS,
)
from pqt_qcat.evaluator.manifest import generate_sha256_manifest, verify_sha256_manifest


def main():
    parser = argparse.ArgumentParser(description="M5 Independent Evaluation Protocol Runner")
    parser.add_argument("--seed", type=int, default=42, help="Seed to evaluate")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of test episodes per scenario")
    parser.add_argument("--calibration-only", action="store_true", help="Run across calibration seeds [42, 11, 12, 13]")
    parser.add_argument("--generate-manifest", action="store_true", help="Generate SHA-256 cryptographic manifest")
    parser.add_argument("--verify-manifest", action="store_true", help="Verify SHA-256 cryptographic manifest")
    parser.add_argument("--unseal-final-seeds", action="store_true", help="Requires explicit ILORI authorization")

    args = parser.parse_args()

    if args.generate_manifest:
        print("Generating SHA-256 manifest...")
        res = generate_sha256_manifest(root_dir)
        print(f"Manifest generated successfully: {res['total_files_hashed']} files hashed.")
        print(f"JSON: {res['manifest_path_json']}")
        print(f"TXT:  {res['manifest_path_txt']}")
        return

    if args.verify_manifest:
        print("Verifying SHA-256 manifest...")
        success, count, mismatches = verify_sha256_manifest(root_dir=root_dir)
        if success:
            print(f"SUCCESS: All {count} files verified matching recorded SHA-256!")
        else:
            print(f"FAILURE: {len(mismatches)} mismatches detected:")
            for m in mismatches:
                print(f"  {m}")
            sys.exit(1)
        return

    # Check unseal gate
    if not args.unseal_final_seeds:
        if args.seed in SEALED_SEEDS:
            print(f"ERROR: Seed {args.seed} is strictly sealed under pre-registration protocol!")
            print("To run evaluation, use calibration seeds: 42, 11, 12, 13.")
            sys.exit(1)

    eval_cfg = EvaluationConfig(
        n_episodes=args.episodes,
        raw_logs_dir=str(root_dir / "results" / "raw_logs"),
        unseal_authorized=args.unseal_final_seeds,
        save_raw_logs=True,
    )
    evaluator = IndependentEvaluator(config=eval_cfg)

    target_seeds = CALIBRATION_SEEDS if args.calibration_only else [args.seed]

    print("=" * 80)
    print("RUNNING MILESTONE M5 INDEPENDENT EVALUATION PROTOCOL (H0–H6)")
    print(f"Target Seeds: {target_seeds}")
    print(f"Episodes per Scenario: {args.episodes}")
    print(f"Raw Logs Directory: {eval_cfg.raw_logs_dir}")
    print("=" * 80)

    all_results = {}
    t_start = time.perf_counter()

    for s in target_seeds:
        print(f"\n---> Evaluating Seed {s} ({args.episodes} episodes/scenario)...")
        res_s = evaluator.run_full_suite(seed=s, n_episodes=args.episodes)
        all_results[str(s)] = res_s
        print(f"Completed Seed {s} in {res_s['elapsed_seconds']:.2f}s")
        for h_name, h_res in res_s["evaluations"].items():
            print(f"  [{h_name}] Status: {h_res['status']}")

    total_time = time.perf_counter() - t_start
    print(f"\nAll evaluations finished in {total_time:.2f} seconds.")

    out_file = root_dir / "results" / "m5_calibration_evaluation.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved evaluation report to {out_file}")


if __name__ == "__main__":
    main()
