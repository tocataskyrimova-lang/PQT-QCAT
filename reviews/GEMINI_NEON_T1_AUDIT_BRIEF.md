# LIFE-003 / T1: INDEPENDENT AUDIT BRIEF FOR GEMINI NEON

- **Target Auditor:** Gemini Neon (and peer independent auditing agents)
- **Subject of Audit:** LIFE-003 / CONTINUOUS-THOUGHT-001 (Phase T1 Delivery & Epistemic Hardening)
- **Repository:** `https://github.com/tocataskyrimova-lang/PQT-QCAT`
- **Working Branch:** `exp/life-003-continuous-thought-001`
- **Target Tag:** `life-003-t1-audit-2026-11-21`
- **Frozen Reference Baseline (LIFE-002):** `06fbd5b7e6fcc224619d19b78ded716f74694a08` (`life-002-freeze-2026-11-20`)
- **Epistemic Status:** `W_CANDIDATE`, `proof_claim = false`
- **Mandate:** Strictly **T0 + T1** (T2/T3/T4 are NOT implemented or claimed).

---

## 1. INVITATION TO ADVERSARIAL CRITICAL AUDIT

Gemini Neon, you are invited to perform an **independent, adversarial, critical audit** of the continuous-thought heartbeat implementation (LIFE-003 T1).

### Rules of Engagement for Auditor:
1. **Do not rubber-stamp.** Do not simply confirm the conclusions of the authors. Actively search for failure modes, leaky abstractions, subtle non-determinisms, edge cases, semantic overstatements, or circular definitions.
2. **Do not rewrite the scientific code.** Your role is to evaluate the existing codebase as published, verify reproducibility from clean scratch, test edge cases, and report an independent verdict with reproducible counterexamples if flaws exist.
3. **Strict Epistemic Distinction:** Clearly separate:
   - What is **empirically/mathematically verified** by code and unit tests.
   - What is an **engineering scaffolding / approximation**.
   - What is **explicitly deferred or non-claimed** (e.g., reward learning, subjective awareness, open-ended autonomous adaptation).

---

## 2. SCIENTIFIC SCOPE & EXPLICIT BOUNDARIES (WHAT T1 IS AND IS NOT)

### What T1 IS:
- A deterministically reproducible, continuous physical runtime for a 48D non-unitary neural core (`SixEchelonCore`) coupled with a legacy amoeba organism interface.
- An uninterrupted continuous physical clock ($Z_{\text{main}}$ never resets to 0 during organism lifetime; ticks advance continuously during idle and event periods).
- A bounded internal deliberation pool (VIMARSA / Sidecar) executing actual isolated $Z_0\text{-}64$ rollouts without mutating the main core state ($Z_{\text{main}}, K_{\text{main}}, \text{rng}_{\text{main}}$).
- A two-channel policy gate (SANKALPA) decoupling epistemic commitment (`COMMIT`, `CLASS_ONLY`, `ABSTAIN`) from physical operational actions (`1..4`, `PROBE`, `WAIT`, `IDLE`).
- An append-only cryptographic witness ledger (Sākṣin Ledger) enforcing strict SHA-256 hash chaining, fail-closed zero-witness guards, and strict separation between ledger entry existence and evidentiary support for specific claims.
- A full-state bit-exact snapshot and resume mechanism (0-bit discrepancy over thousands of post-resume ticks, including during active pending events).

### What T1 IS NOT (Strictly Disclaimed):
- **NO reward-driven weight learning:** Plasticity is frozen (`k_write_gate_active = 0.0`). The synaptic matrix $K$ remains strictly invariant throughout the entire 10,000-tick run. Learning by reinforcement is deferred to T2.
- **NO autonomous closed-loop world planning:** In T1, external queries and world events are supplied by an external schedule. Real environmental signals (`next_obs`, `reward`, `done`) are logged to shadow audit records (`AUDIT_SHADOW`) but do NOT guide autonomous policy updates.
- **NO empirical future world modeling in Sidecar:** The Sidecar pool evaluates alternative candidate rollouts based on **minimal internal neural prediction entropy** (`MINIMAL_NEURAL_PREDICTION_ENTROPY`), NOT by predicting or simulating verified future environmental consequences.
- **NO claim of subjective consciousness:** The implementation provides verifiable dynamical and algorithmic invariants of continuous thought loops; it makes no philosophical or phenomenal claims.

---

## 3. KEY ARCHITECTURAL CHECKPOINTS FOR AUDIT

Auditors are requested to specifically scrutinize the following 6 core areas:

### Audit Point 1: Full-State Snapshot & Deterministic Resume (FIX-1)
- **Source:** `life003/checkpoint.py`, `life003/scheduler.py` (`capture_full_state`, `restore_full_state`).
- **Invariants to verify:**
  - Serialization uses recursive `pack_state()` / `unpack_state()` without unsafe Python `pickle` (`allow_pickle=False`).
  - Full snapshots capture both `SixEchelonCore.snapshot()` (step counts, numerical integrator parameters, diagnostic arrays, RNG bit generator state) and `QNUFAgent.snapshot()` (pending queries, event difference vectors, observation counts).
  - Resume during an active pending query (`test_fix1_full_state_checkpoint_and_resume_pending_event`) completes with bit-exact identical predictions.
  - Continuation from tick 5000 to tick 10000 in `scripts/run_t1_heartbeat.py` achieves 0 discrepancies across 4125+ consecutively verified steps (`step_discrepancy_count == 0`, `status == "PASS"`).

### Audit Point 2: Genuine Runtime Causal Intervention (FIX-3)
- **Source:** `tests/test_continuous_life.py` (`test_fix3_full_runtime_causal_intervention_real_variables`), `scripts/run_t1_heartbeat.py`.
- **Invariants to verify:**
  - Trigger evaluation in the paired test uses **100% real internal variables** ($S_{\text{real}}, U_{\text{real}}, E_{\text{real}}, R_{\text{real}}$) derived from live core state and shadow predictions, with NO hardcoded artificial overrides.
  - When trigger fires in Organism A ($0.4936 > 0.45$), deliberation causally alters the motor decision ($a_{\text{on}} = 1$ vs $a_{\text{off}} = 3$).
  - Deliberation leaves the main core strictly unmutated ($Z_{\text{main}}, K_{\text{main}}, \text{rng}_{\text{main}}$ have identical SHA-256 pre and post deliberation).

### Audit Point 3: Sidecar Honesty & Neural Entropy Semantics (FIX-2 & Item 6)
- **Source:** `life003/sidecar_pool.py`.
- **Invariants to verify:**
  - `clones_requested`, `clones_spawned`, and `clones_executed` strictly match the actual number of branches evaluated in code.
  - Multi-candidate deliberation does NOT claim to compare true world outcomes; it strictly compares internal neural prediction entropy (`MINIMAL_NEURAL_PREDICTION_ENTROPY`, mechanism: `MULTI_BRANCH_PREDICTION_ENTROPY_SELECTION`).

### Audit Point 4: Epistemic Gate & Witness Integrity (FIX-4 & Items 1–5)
- **Source:** `life003/sankalpa.py`, `life003/ledger.py`, `life003/hypotheses.py`.
- **Invariants to verify:**
  - **Negative Test:** An authentic `OBSERVED` or `AUDIT_SHADOW` entry in the Ledger can NEVER authorize `COMMIT` even if passed with an external score of `0.99`. `OBSERVED` yields at most `CLASS_ONLY`; `AUDIT_SHADOW` yields `ABSTAIN`.
  - **Commit Condition:** `COMMIT` strictly requires `VERIFIED_FEEDBACK` with matching provenance, matching candidate action, positive confirmation status, matching applicability domain, and a verified support score extracted directly from the signed Ledger payload.
  - **Fail-Closed:** When `require_structured_witness=True` and `ledger is None`, SANKALPA fails closed to `ABSTAIN` with `zero_witness_guard_tripped = True`.
  - **Ledger Integrity:** `validate_integrity()` strictly validates `seq == len(records)` and `prev_hash == computed final entry hash`.
  - **Eviction Audit:** Eviction in `HypothesisPool` prioritizes `REFUTED` -> `UNVERIFIED` -> oldest `SUPPORTED`, logging each eviction atomically to the Ledger as `EVICTED`.

### Audit Point 5: Zero-Leakage & Oracle Isolation
- **Invariants to verify:**
  - Organism and agent never read hidden environment labels (`rule_id`, `true_optimal_action`, future sequence tokens).
  - No future events leak into deliberation.
  - No synaptic weight updates occur without explicit external delayed feedback.

### Audit Point 6: Search for Vulnerabilities & Counterexamples
- Gemini Neon is invited to construct explicit test scripts attempting to break any of the stated invariants:
  - Can a desynchronized ledger pass `validate_integrity()`?
  - Can an organism commit without verified feedback?
  - Can a sidecar mutate the main core state?
  - Can a cloned organism diverge from an uninterrupted run given identical seeds?

---

## 4. INDEPENDENT REPRODUCTION PROTOCOL

To independently reproduce all results from scratch in a clean environment:

```bash
# 1. Clone repository into a clean directory
git clone https://github.com/tocataskyrimova-lang/PQT-QCAT.git pqt_audit_verify
cd pqt_audit_verify

# 2. Checkout the audit tag/branch
git checkout life-003-t1-audit-2026-11-21
# Verify HEAD commit SHA matches release metadata

# 3. Verify SHA-256 hashes of all 38 manifest files
python -c "
import hashlib, json
from pathlib import Path
root = Path('.')
manifest = json.loads((root / 'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
mismatches = [p for p, h in manifest['hashes'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest() != h]
assert not mismatches, f'Mismatches: {mismatches}'
print(f'ALL {len(manifest[\"hashes\"])} FILES MATCH SHA-256 MANIFEST 100%!')
"

# 4. Run full unit test suite (28/28 tests)
pytest tests/test_continuous_life.py -v

# 5. Reproduce 10,000-tick heartbeat and exact resume
# (Runs in ~2-3 minutes, writes to results/THOUGHT_LOOP_T1/)
python scripts/run_t1_heartbeat.py

# 6. Verify exact resume and paired intervention output
python -c "
import json
res = json.load(open('results/THOUGHT_LOOP_T1/resume_verification.json'))
assert res['status'] == 'PASS', 'Resume verification failed!'
assert res['step_discrepancy_count'] == 0, 'Discrepancy detected!'
print('EXACT RESUME: PASS (0 discrepancies over', res['total_verified_ticks'], 'ticks)')

t19 = json.load(open('results/THOUGHT_LOOP_T1/trigger_on_vs_off.json'))
assert t19['verdict'] == 'PASS', 'Paired intervention failed!'
assert t19['fix3_real_runtime_paired_intervention']['causally_diverged'] is True
print('CAUSAL INTERVENTION: PASS (Trigger ON action != Trigger OFF action)')
"
```

---

## 5. AUDIT REPORT FORMAT

Gemini Neon is requested to document the audit findings in:
`reviews/GEMINI_NEON_T1_AUDIT_REPORT.md` (or equivalent PR / issue review) with the following sections:
1. **Executive Verdict:** `CONFIRMED_VALID` | `DISCREPANCY_DETECTED` | `VULNERABILITY_FOUND` | `OVERSTATEMENT_IDENTIFIED`.
2. **Reproducibility Audit:** Results of clean clone, SHA-256 verification, pytest, and 10,000-tick run.
3. **Adversarial Analysis:** Evaluation of the 6 core checkpoints, including any tested edge cases or counterexamples.
4. **Epistemic Precision Assessment:** Review of disclaimers, terminology accuracy, and strict separation between T1 and future phases (T2/T3/T4).
5. **Recommendations for Phase T2 (Reinforcement & Plasticity).**
