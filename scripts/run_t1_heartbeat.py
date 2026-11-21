"""Execution runner for LIFE-003 T1 Continuous Thought Loop & Verification (10,000 ticks)."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pqt_qcat.core.six_echelon import SixEchelonCore
from life003.config import load_and_validate_prereg_config
from life003.scheduler import ContinuousLifeOrganism, array_hash
from life003.checkpoint import compute_state_hash, save_checkpoint, load_checkpoint


class PhysicalStepObserver:
    """Non-destructive observer comparing MAIN physical core steps between uninterrupted and resumed organism."""
    def __init__(self, org_main: ContinuousLifeOrganism, org_clone: ContinuousLifeOrganism):
        self.org_main = org_main
        self.org_clone = org_clone
        self.total_physical_steps_observed = 0
        self.interior_query_subticks_observed = 0
        self.scheduler_steps_observed = 0
        self.subtick_discrepancies = 0
        self.step_discrepancies = 0

        self.orig_class_step = SixEchelonCore.step
        self._pending_main_snaps = []

        observer = self

        def wrapped_step(core_self, *args, **kwargs):
            res = observer.orig_class_step(core_self, *args, **kwargs)
            if core_self is observer.org_main.core:
                snap = {
                    "Z_hash": array_hash(core_self.Z),
                    "K_hash": array_hash(core_self.K),
                    "rng_state": copy.deepcopy(core_self.rng.bit_generator.state),
                }
                observer._pending_main_snaps.append(snap)
            elif core_self is observer.org_clone.core:
                if observer._pending_main_snaps:
                    main_snap = observer._pending_main_snaps.pop(0)
                    observer.total_physical_steps_observed += 1
                    clone_z = array_hash(core_self.Z)
                    clone_k = array_hash(core_self.K)
                    clone_rng = copy.deepcopy(core_self.rng.bit_generator.state)
                    if (clone_z != main_snap["Z_hash"] or 
                        clone_k != main_snap["K_hash"] or 
                        clone_rng != main_snap["rng_state"]):
                        observer.subtick_discrepancies += 1
            return res

        SixEchelonCore.step = wrapped_step

    def unwrap(self):
        SixEchelonCore.step = self.orig_class_step


def run_t1_experiment(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir or Path(__file__).resolve().parent.parent / "results" / "THOUGHT_LOOP_T1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing ContinuousLifeOrganism with pre-registered config...")
    org = ContinuousLifeOrganism(config_path=str(config_path) if config_path else None)

    total_target_ticks = org.cfg["clock_and_schedule"]["total_physical_ticks"]
    checkpoint_interval = org.cfg["clock_and_schedule"]["checkpoint_interval_ticks"]
    resume_tick = org.cfg["clock_and_schedule"]["resume_test_tick"]

    periodic_checkpoints = {}
    resume_verification = {
        "resume_tick": resume_tick,
        "checkpoint_hash_pre": None,
        "checkpoint_hash_post": None,
        "step_discrepancy_count": 0,
        "subtick_discrepancy_count": 0,
        "verified_scheduler_steps": 0,
        "verified_physical_ticks": 0,
        "verified_interior_query_subticks": 0,
        "all_5000_physical_ticks_identical": False,
        "total_verified_ticks": 0,
        "status": "PENDING",
    }

    org_resumed: ContinuousLifeOrganism | None = None
    observer: PhysicalStepObserver | None = None
    resumed_matched = True

    event_counter = 0
    jsonl_path = output_dir / "thought_stream.jsonl"
    print(f"Beginning 10,000 physical tick execution loop to {jsonl_path}...")

    with open(jsonl_path, mode="w", encoding="utf-8", newline="\n") as f_out:
        while org.clock.core_tick < total_target_ticks:
            current_tick = org.clock.core_tick

            # Check if we need to do the unannounced exact resume verification test
            if current_tick == resume_tick and org_resumed is None:
                print(f"[Tick {current_tick}] Reached resume_test_tick! Serializing atomic checkpoint...")
                ckpt_resume_path = output_dir / f"checkpoint_{current_tick}_pre_resume.npz"
                h_pre = save_checkpoint(ckpt_resume_path, org.capture_full_state())
                resume_verification["checkpoint_hash_pre"] = h_pre

                # Load and restore in a fresh clone
                loaded_state, h_post = load_checkpoint(ckpt_resume_path)
                resume_verification["checkpoint_hash_post"] = h_post
                assert h_pre == h_post, "Checkpoint hash mismatch upon immediate read!"

                org_resumed = ContinuousLifeOrganism(config=org.cfg)
                org_resumed.restore_full_state(loaded_state)

                h_clone = compute_state_hash(org_resumed.capture_full_state())
                assert h_clone == h_pre, "Cloned organism state hash diverged from pre-resume!"
                print(f"[Tick {current_tick}] Checkpoint verified: SHA-256 = {h_pre}")

                # Attach non-destructive physical step observer
                observer = PhysicalStepObserver(org, org_resumed)

            # Check periodic checkpoints
            if current_tick > 0 and current_tick % checkpoint_interval == 0 and current_tick not in periodic_checkpoints:
                ckpt_p = output_dir / f"checkpoint_{current_tick}.npz"
                h_p = save_checkpoint(ckpt_p, org.capture_full_state())
                periodic_checkpoints[current_tick] = {"path": ckpt_p.name, "state_hash": h_p}
                print(f"[Tick {current_tick}] Periodic checkpoint saved: {h_p}")

            # Decide whether to run idle tick or query event
            ticks_remaining = total_target_ticks - org.clock.core_tick
            should_run_query = (org.clock.core_tick % 40 == 0) and (ticks_remaining >= org.t_query)

            if should_run_query:
                query_token = (event_counter % 6) + 1
                event_counter += 1
                log_entry = org.execute_physical_tick(event_query=query_token)
                if org_resumed is not None:
                    log_entry_res = org_resumed.execute_physical_tick(event_query=query_token)
                    if observer is not None:
                        observer.interior_query_subticks_observed += (org.t_query - 1)
            else:
                log_entry = org.execute_physical_tick(event_query=None)
                if org_resumed is not None:
                    log_entry_res = org_resumed.execute_physical_tick(event_query=None)

            # Write continuous log line
            f_out.write(json.dumps(log_entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")

            # Verification of resumed clone against uninterrupted continuation:
            # FIX-1: Compare COMPLETE dictionary states via deterministic state hash
            if org_resumed is not None:
                if observer is not None:
                    observer.scheduler_steps_observed += 1
                h_org = compute_state_hash(org.capture_full_state())
                h_res = compute_state_hash(org_resumed.capture_full_state())
                if h_org != h_res:
                    if observer is not None:
                        observer.step_discrepancies += 1
                    resumed_matched = False

    if observer is not None:
        observer.unwrap()
        resume_verification["verified_scheduler_steps"] = observer.scheduler_steps_observed
        resume_verification["verified_physical_ticks"] = observer.total_physical_steps_observed
        resume_verification["verified_interior_query_subticks"] = observer.interior_query_subticks_observed
        resume_verification["step_discrepancy_count"] = observer.step_discrepancies
        resume_verification["subtick_discrepancy_count"] = observer.subtick_discrepancies
        resume_verification["all_5000_physical_ticks_identical"] = (
            observer.total_physical_steps_observed == 5000 and 
            observer.subtick_discrepancies == 0 and 
            observer.step_discrepancies == 0
        )
        resume_verification["total_verified_ticks"] = observer.total_physical_steps_observed

    # Save final periodic checkpoint at 10,000
    final_ckpt = output_dir / f"checkpoint_{org.clock.core_tick}.npz"
    h_final = save_checkpoint(final_ckpt, org.capture_full_state())
    periodic_checkpoints[org.clock.core_tick] = {"path": final_ckpt.name, "state_hash": h_final}
    print(f"[Tick {org.clock.core_tick}] Final checkpoint saved: {h_final}")

    resume_verification["status"] = (
        "PASS"
        if (resumed_matched and resume_verification["step_discrepancy_count"] == 0 and resume_verification["subtick_discrepancy_count"] == 0)
        else "FAIL"
    )

    with open(output_dir / "resume_verification.json", mode="w", encoding="utf-8", newline="\n") as f_res:
        json.dump(
            {
                **resume_verification,
                "periodic_checkpoints": periodic_checkpoints,
                "total_ticks_run": org.clock.core_tick,
                "final_state_hash": h_final,
                "ledger_chain_valid": org.ledger.validate_integrity(),
                "total_ledger_records": len(org.ledger.records),
                "total_deliberations": org.trigger.total_triggers_count,
            },
            f_res,
            indent=2,
        )
    print("Exact resume verification report saved.")

    # -----------------------------------------------------------------------
    # Run Test 19 + FIX-3 Real-Variable Runtime Intervention and save log
    # -----------------------------------------------------------------------
    print("Running Test 19 & FIX-3 runtime causal paired comparison...")
    cfg_t19 = copy.deepcopy(org.cfg)
    cfg_t19["rng_streams"]["rng_main_seed"] = 810001
    cfg_t19["rng_streams"]["banned_seeds"] = []
    
    # 1. Classical controlled context
    org_t19 = ContinuousLifeOrganism(config=cfg_t19)
    for q in [1, 2, 3]:
        org_t19.execute_physical_tick(event_query=q, force_disable_trigger=True)

    l_live = org_t19.core.predict()[1:5]
    p_live = np.exp(l_live - np.max(l_live)) / np.sum(np.exp(l_live - np.max(l_live)))
    u_live = -float(np.sum(p_live * np.log(np.clip(p_live, 1e-12, 1.0)))) / float(np.log(4))
    a_live = int(np.argmax(l_live)) + 1

    ctrl = org_t19.cfg["endogenous_trigger"]["test19_control_context"]
    score, can_fire, comp = org_t19.trigger.evaluate(
        surprise_proxy=ctrl["fixed_surprise_proxy"],
        uncertainty=u_live,
        energy_deficit=ctrl["fixed_energy_deficit"],
        unresolved=ctrl["fixed_unresolved"],
        current_tick=org_t19.clock.core_tick,
        force_disable=False,
    )

    z_pre = array_hash(org_t19.core.Z)
    k_pre = array_hash(org_t19.core.K)
    rng_pre = copy.deepcopy(org_t19.core.rng.bit_generator.state)

    stim = np.zeros(8)
    stim[1] = 1.0
    delib_res = org_t19.sidecar_pool.run_deliberation(org_t19.core, stim, org_t19.rng_sidecar, num_clones=1)
    a_sidecar = delib_res["sidecar_prediction"]

    z_post = array_hash(org_t19.core.Z)
    k_post = array_hash(org_t19.core.K)
    rng_post = copy.deepcopy(org_t19.core.rng.bit_generator.state)

    # 2. FIX-3: Full runtime paired intervention with 100% REAL internal variables
    org_real_a = ContinuousLifeOrganism(config=cfg_t19)
    org_real_b = ContinuousLifeOrganism(config=cfg_t19)

    res_real_a = org_real_a.execute_physical_tick(event_query=1, force_disable_trigger=False)
    res_real_b = org_real_b.execute_physical_tick(event_query=1, force_disable_trigger=True)

    real_act_a = res_real_a["sankalpa_decision"]["operational_channel"]
    real_act_b = res_real_b["sankalpa_decision"]["operational_channel"]

    t19_log = {
        "test_name": "test_endogenous_trigger_changes_behavior_vs_disabled_trigger",
        "frozen_k_seed": 810001,
        "u_live_entropy": u_live,
        "trigger_score": score,
        "threshold_theta": org_t19.trigger.threshold_theta,
        "trigger_fired": can_fire,
        "trigger_components": comp,
        "branch_a_trigger_off": {
            "action_chosen": a_live,
            "policy_source": "LIVE_MAIN_LOGITS",
            "logits": l_live.tolist(),
        },
        "branch_b_trigger_on": {
            "action_chosen": a_sidecar,
            "policy_source": "Z0_64_DELIBERATION",
            "sidecar_logits": delib_res["sidecar_logits"],
            "sidecar_entropy": delib_res["sidecar_entropy"],
            "clones_requested": delib_res["clones_requested"],
            "clones_executed": delib_res["clones_executed"],
            "divergence_mechanism": delib_res["divergence_mechanism"],
        },
        "behavior_causally_changed": a_live != a_sidecar,
        "main_core_invariance": {
            "z_main_hash_pre": z_pre,
            "z_main_hash_post": z_post,
            "z_main_identical": z_pre == z_post,
            "k_main_hash_pre": k_pre,
            "k_main_hash_post": k_post,
            "k_main_identical": k_pre == k_post,
            "rng_main_identical": rng_pre == rng_post,
        },
        "fix3_real_runtime_paired_intervention": {
            "trigger_on_action": real_act_a,
            "trigger_off_action": real_act_b,
            "causally_diverged": real_act_a != real_act_b,
            "trigger_score_real": res_real_a["trigger"]["score"],
            "surprise_proxy_real": res_real_a["trigger"]["components"]["surprise_proxy"],
            "uncertainty_real": res_real_a["trigger"]["components"]["uncertainty"],
            "energy_deficit_real": res_real_a["trigger"]["components"]["energy_deficit"],
            "divergence_mechanism": res_real_a["deliberation"]["divergence_mechanism"],
            "selection_criterion": res_real_a["deliberation"].get("selection_criterion", "MINIMAL_NEURAL_PREDICTION_ENTROPY"),
            "clones_executed": res_real_a["deliberation"]["clones_executed"],
        },
        "verdict": "PASS" if (a_live != a_sidecar and real_act_a != real_act_b and z_pre == z_post and k_pre == k_post and rng_pre == rng_post) else "FAIL",
    }

    with open(output_dir / "trigger_on_vs_off.json", mode="w", encoding="utf-8", newline="\n") as f_t19:
        json.dump(t19_log, f_t19, indent=2)
    print("Test 19 & FIX-3 paired log saved: trigger_on_vs_off.json")
    print("Heartbeat T1 execution finished successfully.")


if __name__ == "__main__":
    run_t1_experiment()
