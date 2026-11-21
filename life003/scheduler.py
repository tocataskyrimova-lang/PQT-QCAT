"""Physical Tick Scheduler & Organism State Machine for LIFE-003."""
from __future__ import annotations
import copy
import hashlib
from typing import Any
import numpy as np

from pqt_qcat.core.six_echelon import SixEchelonCore, SixEchelonConfig
from pqt_qcat.memory.qnuf import QNUFAgent

from .config import load_and_validate_prereg_config
from .clock import LifeClock
from .ledger import SaksinLedger
from .hypotheses import HypothesisPool
from .trigger import EndogenousTrigger
from .sidecar_pool import SidecarPool
from .sankalpa import SankalpaGate, SankalpaDecision
from .amoeba_adapter import AmoebaWorld
from .checkpoint import compute_state_hash, save_checkpoint, load_checkpoint


def array_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of a numpy array's contiguous byte buffer."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


class ContinuousLifeOrganism:
    """Full living organism runtime orchestrating (Z, K, H, G, Plan, Pending, World, Ledger, RNG, Clock)."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | None = None,
        config_hash: str | None = None,
    ):
        if config is None:
            self.cfg, self.cfg_hash = load_and_validate_prereg_config(config_path, expected_sha=config_hash)
        else:
            self.cfg = config
            self.cfg_hash = config_hash or "EXPLICIT_CONFIG"

        # 1. Clocks
        self.clock = LifeClock()

        # 2. RNG Streams
        rng_cfg = self.cfg["rng_streams"]
        self.seed_main = rng_cfg["rng_main_seed"]
        self.seed_sidecar = rng_cfg["rng_sidecar_seed"]
        self.seed_world = rng_cfg["rng_world_seed"]

        banned = set(rng_cfg.get("banned_seeds", [101, 102, 103, 104, 105]))
        for stream, s in [("rng_main_seed", self.seed_main), ("rng_sidecar_seed", self.seed_sidecar), ("rng_world_seed", self.seed_world)]:
            if s in banned:
                raise ValueError(f"Quarantined M5 seed {s} detected in stream {stream}!")

        self.rng_sidecar = np.random.default_rng(self.seed_sidecar)

        # 3. Physical Core (48D SixEchelonCore)
        core_cfg = SixEchelonConfig(
            h_base=self.cfg["clock_and_schedule"]["dt_core"],
            enable_vortex=True,
        )
        self.core = SixEchelonCore(config=core_cfg, seed=self.seed_main)

        # 4. Frozen QNUFAgent Wrapper (t_query = 8)
        self.t_query = int(self.cfg["clock_and_schedule"]["t_query"])
        self.agent = QNUFAgent(
            self.core,
            t_query=self.t_query,
            learning_mode="CARRY_TRAIN",
            eligibility="event",
        )

        # 5. Sākṣin Ledger
        self.ledger = SaksinLedger(
            genesis_config_sha=self.cfg_hash,
            genesis_clock=self.clock,
        )

        # 6. Hypotheses Space H
        h_cfg = self.cfg["hypotheses_and_ledger"]
        self.hypotheses = HypothesisPool(
            max_active_hypotheses=h_cfg["max_active_hypotheses"],
            m_phi_uses_aging=h_cfg["m_phi_uses_aging"],
            m_phi_uses_ossified=h_cfg["m_phi_uses_ossified"],
            age_ticks_aging=h_cfg["age_ticks_aging"],
            age_ticks_ossified=h_cfg["age_ticks_ossified"],
        )

        # 7. Endogenous Trigger
        trig_cfg = self.cfg["endogenous_trigger"]
        bdg_cfg = self.cfg["deliberation_budget"]
        self.trigger = EndogenousTrigger(
            w_surprise=trig_cfg["weights"]["w_surprise"],
            w_uncertainty=trig_cfg["weights"]["w_uncertainty"],
            w_energy_deficit=trig_cfg["weights"]["w_energy_deficit"],
            w_unresolved=trig_cfg["weights"]["w_unresolved"],
            threshold_theta=trig_cfg["threshold_theta"],
            norm_scale_surprise=trig_cfg["norm_scale_surprise"],
            cooldown_ticks=bdg_cfg["cooldown_ticks_after_trigger"],
            max_total_deliberations=bdg_cfg["max_total_deliberations_t1"],
        )

        # 8. Sidecar Pool
        self.sidecar_pool = SidecarPool(
            max_sidecars_per_event=bdg_cfg["max_sidecars_per_event"],
            sidecar_rollout_ticks=bdg_cfg["sidecar_rollout_ticks"],
            max_deliberation_ticks_per_tick=bdg_cfg["max_deliberation_ticks_per_tick"],
        )

        # 9. Two-channel SANKALPA Gate
        san_cfg = self.cfg["sankalpa_policy_channels"]
        self.sankalpa = SankalpaGate(
            require_structured_witness=san_cfg["epistemic"]["require_structured_witness"],
            witness_support_min_score=san_cfg["epistemic"]["witness_support_min_score"],
            allow_zero_witness_commit=san_cfg["epistemic"]["allow_zero_witness_commit"],
            default_idle_action=san_cfg["operational"]["default_idle_action"],
            safe_probe_action=san_cfg["operational"]["safe_probe_action"],
            wait_action=san_cfg["operational"]["wait_action"],
        )

        # 10. World & Homeostasis
        self.world = AmoebaWorld(seed=self.seed_world)
        self.energy_deficit: float = 0.0
        self.unresolved_plan: float = 0.0

        # State tracking
        self.pending_event: dict[str, Any] | None = None
        self.deliberation_history: list[dict[str, Any]] = []

    def get_state_integrity(self) -> dict[str, str]:
        """Compute SHA-256 hashes of core state components."""
        return {
            "z_main_hash": array_hash(self.core.Z),
            "k_main_hash": array_hash(self.core.K),
            "rng_main_hash": hashlib.sha256(
                str(self.core.rng.bit_generator.state).encode("utf-8")
            ).hexdigest(),
        }

    def execute_physical_tick(
        self,
        event_query: int | None = None,
        event_stimulus: np.ndarray | None = None,
        force_disable_trigger: bool = False,
    ) -> dict[str, Any]:
        """Execute one complete physical scheduler cycle.
        
        Branches:
          1. IDLE tick: SixEchelonCore.step(None, write_gate=0.0); K unchanged.
          2. QUERY tick: QNUFAgent.begin_query() -> t_query physical ticks.
        """
        current_tick = self.clock.core_tick
        ledger_events_this_tick = []
        deliberation_record = None

        # Checkpoint before anything
        z_pre = self.core.Z.copy()
        k_pre = self.core.K.copy()

        if event_query is None:
            # --- BRANCH 1: IDLE TICK ---
            phase = "IDLE"
            # Exactly one core step with write_gate = 0.0
            self.core.step(x_t=None, write_gate=0.0)
            self.clock.advance_core(1)
            self.hypotheses.advance_time(1)

            # Invariance check: K must NEVER change during idle
            np.testing.assert_array_equal(self.core.K, k_pre)

            # Evaluate endogenous signals
            e2_norm = float(np.linalg.norm(self.core.Z[8:16]))
            logits_live = self.core.predict()[1:5]
            uncertainty = self.trigger.compute_entropy_uncertainty(logits_live)
            self.energy_deficit = float(1.0 - self.world.energy)

            score, fired, comp = self.trigger.evaluate(
                surprise_proxy=e2_norm / self.trigger.norm_scale_surprise,
                uncertainty=uncertainty,
                energy_deficit=self.energy_deficit,
                unresolved=self.unresolved_plan,
                current_tick=self.clock.core_tick,
                force_disable=force_disable_trigger,
            )

            if fired:
                # Deliberate on unprovoked idle thought
                deliberation_record = self.sidecar_pool.run_deliberation(
                    main_core=self.core,
                    stimulus=None,
                    rng_sidecar=self.rng_sidecar,
                    num_clones=1,
                )
                cand_action = deliberation_record["sidecar_prediction"]
                hyp = self.hypotheses.register_candidate(
                    candidate_id=f"H_idle_{self.clock.core_tick}",
                    action_candidate=cand_action,
                    v_just="UNVERIFIED",
                    witness_ref=None,
                )
                if self.hypotheses.last_eviction is not None:
                    evict_rec = self.ledger.append_entry(
                        record_type="EVICTED",
                        payload=self.hypotheses.last_eviction,
                        clock=self.clock,
                    )
                    ledger_events_this_tick.append(evict_rec)

                rec = self.ledger.append_entry(
                    record_type="HYPOTHESIS",
                    payload={
                        "candidate_id": hyp.candidate_id,
                        "action": cand_action,
                        "v_just": hyp.v_just,
                        "s_fresh": hyp.s_fresh,
                        "source": "IDLE_VIMARSA",
                    },
                    clock=self.clock,
                )
                ledger_events_this_tick.append(rec)

            decision = self.sankalpa.decide(
                candidate_action="IDLE",
                witness_ref=None,
                is_idle=True,
                ledger=self.ledger,
            )

        else:
            # --- BRANCH 2: QUERY / EVENT TICK ---
            phase = "DELIBERATION_AND_ACT"
            self.clock.advance_world(1)
            self.clock.advance_decision(1)

            # Observe and execute begin_query: exactly t_query steps
            obs_record = self.agent.begin_query(event_query, stimulus=event_stimulus)
            self.clock.advance_core(self.t_query)
            self.hypotheses.advance_time(self.t_query)

            # Record observation in ledger
            obs_rec = self.ledger.append_entry(
                record_type="OBSERVED",
                payload={
                    "event_id": obs_record["event_id"],
                    "query": event_query,
                    "prediction": obs_record["prediction"],
                    "entropy": obs_record["entropy"],
                },
                clock=self.clock,
            )
            ledger_events_this_tick.append(obs_rec)

            e2_norm = float(np.linalg.norm(self.core.Z[8:16]))
            uncertainty = float(obs_record["entropy"])
            self.energy_deficit = float(1.0 - self.world.energy)

            score, fired, comp = self.trigger.evaluate(
                surprise_proxy=e2_norm / self.trigger.norm_scale_surprise,
                uncertainty=uncertainty,
                energy_deficit=self.energy_deficit,
                unresolved=self.unresolved_plan,
                current_tick=self.clock.core_tick,
                force_disable=force_disable_trigger,
            )

            chosen_action = obs_record["prediction"]
            witness_ref = None

            if fired:
                stim_arr = np.zeros(8)
                if event_query:
                    stim_arr[event_query] = 1.0

                num_clones_to_run = int(self.cfg.get("deliberation_budget", {}).get("max_sidecars_per_event", 1))
                deliberation_record = self.sidecar_pool.run_deliberation(
                    main_core=self.core,
                    stimulus=stim_arr,
                    rng_sidecar=self.rng_sidecar,
                    num_clones=num_clones_to_run,
                )
                chosen_action = deliberation_record["sidecar_prediction"]
                hyp = self.hypotheses.register_candidate(
                    candidate_id=f"H_evt_{self.clock.core_tick}",
                    action_candidate=chosen_action,
                    v_just="UNVERIFIED",
                    witness_ref=None,
                )
                if self.hypotheses.last_eviction is not None:
                    evict_rec = self.ledger.append_entry(
                        record_type="EVICTED",
                        payload=self.hypotheses.last_eviction,
                        clock=self.clock,
                    )
                    ledger_events_this_tick.append(evict_rec)

                hyp_rec = self.ledger.append_entry(
                    record_type="HYPOTHESIS",
                    payload={
                        "candidate_id": hyp.candidate_id,
                        "action": chosen_action,
                        "v_just": hyp.v_just,
                        "s_fresh": hyp.s_fresh,
                        "source": "EVENT_VIMARSA",
                    },
                    clock=self.clock,
                )
                ledger_events_this_tick.append(hyp_rec)

            decision = self.sankalpa.decide(
                candidate_action=chosen_action,
                witness_ref=witness_ref,
                is_safe_probe=False,
                is_idle=False,
                ledger=self.ledger,
            )

            # Step world with operational action
            # Note: In T1 world events are driven from the outside; next_obs/reward/done
            # are recorded for audit but not yet used for autonomous policy learning (deferred to T2).
            next_obs, reward, done, info = self.world.step(decision.operational_channel)

            # Atomic feedback closure in T1 (audit-only, K unchanged)
            self.agent.finish_query(None)
            self.clock.advance_feedback(1)

            # Invariance check: K must NEVER change in T1
            np.testing.assert_array_equal(self.core.K, k_pre)

            fb_rec = self.ledger.append_entry(
                record_type="AUDIT_SHADOW",
                payload={
                    "event_id": obs_record["event_id"],
                    "action_taken": decision.operational_channel,
                    "reward": reward,
                    "k_unmutated": True,
                },
                clock=self.clock,
            )
            ledger_events_this_tick.append(fb_rec)

        # Build JSONL stream log item
        integrity = self.get_state_integrity()
        log_entry = {
            "tick": self.clock.core_tick,
            "clock": self.clock.snapshot(),
            "phase": phase,
            "state_integrity": integrity,
            "trigger": {
                "score": float(score),
                "threshold": float(self.trigger.threshold_theta),
                "fired": bool(fired),
                "components": comp,
            },
            "deliberation": deliberation_record,
            "sankalpa_decision": {
                "epistemic_channel": decision.epistemic_channel,
                "operational_channel": decision.operational_channel,
                "witness_ref": decision.witness_ref,
                "zero_witness_guard_tripped": decision.zero_witness_guard_tripped,
            },
            "saksin_ledger_events": ledger_events_this_tick,
        }

        return log_entry

    def capture_full_state(self) -> dict[str, Any]:
        """Capture complete serializable state for bit-exact checkpoint using official snapshots."""
        return {
            "core_snapshot": self.core.snapshot(),
            "agent_snapshot": self.agent.snapshot(),
            "rng_sidecar_state": copy.deepcopy(self.rng_sidecar.bit_generator.state),
            "clock": self.clock.snapshot(),
            "ledger": self.ledger.snapshot(),
            "hypotheses": self.hypotheses.snapshot(),
            "trigger": self.trigger.snapshot(),
            "world": self.world.snapshot(),
            "energy_deficit": float(self.energy_deficit),
            "unresolved_plan": float(self.unresolved_plan),
            "cfg_hash": self.cfg_hash,
        }

    def restore_full_state(self, state: dict[str, Any]) -> None:
        """Restore complete organism state bit-for-bit."""
        if "core_snapshot" in state:
            self.core.restore(state["core_snapshot"])
        else:
            self.core.Z = state["core_Z"].copy()
            self.core.K = state["core_K"].copy()
            self.core.B = state["core_B"].copy()
            self.core.W_readout = state["core_W"].copy()
            self.core.rng.bit_generator.state = copy.deepcopy(state["rng_main_state"])

        if "agent_snapshot" in state:
            self.agent.restore(state["agent_snapshot"])
        elif "agent_state" in state:
            ag_st = state["agent_state"]
            self.agent.event_id = int(ag_st["event_id"])
            self.agent.observed_counts = list(ag_st["observed_counts"])
            self.agent.shadow_steps = int(ag_st["shadow_steps"])
            self.agent.update_candidates = int(ag_st["update_candidates"])

        self.rng_sidecar.bit_generator.state = copy.deepcopy(state["rng_sidecar_state"])
        self.clock = LifeClock.restore(state["clock"])
        self.ledger = SaksinLedger.restore(state["ledger"])
        self.hypotheses = HypothesisPool.restore(state["hypotheses"])
        self.trigger = EndogenousTrigger.restore(state["trigger"])
        self.world = AmoebaWorld.restore(state["world"])

        self.energy_deficit = float(state["energy_deficit"])
        self.unresolved_plan = float(state["unresolved_plan"])
        self.cfg_hash = state["cfg_hash"]
