# LIFE-003 / CONTINUOUS-THOUGHT-001: Architecture & Inventory Report (v0.2.1)
**Status:** W_CANDIDATE, proof_claim = false  
**Mandate:** T0 (Inventory + Adapter) → Config Freeze → T1 (Heartbeat Runtime & Verification) ONLY  
**Baseline Git Commit:** `06fbd5b7e6fcc224619d19b78ded716f74694a08`  
**Baseline Git Tag:** `life-002-freeze-2026-11-20`  
**Working Branch:** `exp/life-003-continuous-thought-001`  

---

## 1. Inventory of Frozen Baseline (LIFE-002)

### 1.1. Core Dynamics: SixEchelonCore
- **File:** `pqt_qcat/core/six_echelon.py` (SHA-256: `54dc6f7346a7f5c46410be637371843693b9281609956b5928536f1820e5aaab`)
- **State Representation:** 48D continuous state $Z \in \mathbb{R}^{48}$ structured into 6 cells of 8 coordinates:
  - $E_1$ (0..7): $Z_{\text{sens\_1}}$ — primary sensory receiver.
  - $E_2$ (8..15): $Z_{\text{sens\_2}}$ — prediction residual & sensory trace.
  - $E_3$ (16..23): $Z_{\text{hold\_1}}$ — write-gated cue latch (isolated when `write_gate = 0.0`).
  - $E_4$ (24..31): $Z_{\text{hold\_2}}$ — deep protected retention with low mobility $\Gamma_{\text{hold}}$ (`hold_mobility_scale = 0.001`).
  - $E_5$ (32..39): $Z_{\text{ctx\_1}}$ — endogenous context attractor basins.
  - $E_6$ (40..47): $Z_{\text{ctx\_2}}$ — rule shift detector & exploration trigger.
- **Structural Parameters:**
  - $K \in \mathbb{R}^{48 \times 48}$: plastic inter-cell coupling matrix, bounded currents $\|u_c\| \le U_{\text{max}} = 5.0$.
  - $B \in \mathbb{R}^{48 \times d_{\text{in}}}$: input sensory drive.
  - $W_{\text{readout}} \in \mathbb{R}^{d_{\text{out}} \times 48}$: frozen linear readout projection ($d_{\text{out}} = 8$).
  - Vortex dynamics: skew-symmetric matrix $A$ activated when `enable_vortex = True` (slope $\beta_v = 4.0$).
- **Tick Semantics:**
  - `step(x_t=None, write_gate=0.0, dt=0.01)`: advances continuous physics by $dt$.
  - In idle mode, `write_gate = 0.0` ensures the Hold compartment ($E_3, E_4$) is completely decoupled from writes, and $K$ remains strictly unchanged.

### 1.2. Agent Interface & Pending Constraint: QNUFAgent
- **File:** `pqt_qcat/memory/qnuf.py` (SHA-256: `11d18ceb1dc82b97c9f8166002ee334b74e8b88e39749e658d694d02cbc51582`)
- **Base Class:** `StatefulOnlineAgent` in `pqt_qcat/memory/exp_stateful_online_001.py` (default `t_query = 8`, `t_context = 4`, `eta_k = 0.8`).
- **Internal Query Mechanics (`begin_query`):**
  - Upon query $q \in \{0..6\}$, `begin_query` runs **exactly $t_{\text{query}} = 8$ physical integration steps** on `self.core` with input stimulus, and in parallel runs $t_{\text{query}} = 8$ steps on a shadow core without input.
  - `clock.core_tick` increases by exactly 8 ticks.
  - **Single Pending Rule:** `QNUFAgent` holds exactly one active pending record:
    ```python
    if self.pending is not None:
        raise RuntimeError('Resolve pending feedback before the next query')
    ```
  - The pending structure stores:
    ```python
    self.pending = {
        'prediction': record,
        'legacy': psi - self.psi_0,
        'event': psi - np.tanh(shadow.Z)
    }
    ```
  - **Frozen Constraint on Feedback (`finish_query`):**
    `finish_query` evaluates eligibility $\phi$ recorded during `begin_query`. If any intervening physical steps were executed on `self.core` between `begin_query` and `finish_query`, the post-state would drift, violating exact reproducibility.
  - **T1 Architecture Requirement:** In T1, events are processed via an atomic presentation-to-evaluation sequence (`begin_query` $\to$ evaluation $\to$ `finish_query(None)` in audit mode), ensuring full compliance with the frozen agent contract. Delayed-feedback credit assignment is strictly reserved for the T2 adapter.

### 1.3. Sidecar Deliberation: functional_sidecar_b2a
- **File:** `pqt_qcat/memory/functional_sidecar_b2a.py` (SHA-256: `0491ff9a3be053b30e3a0c056c71143baf1628bc9bd6ea036b639618f0f989ab`)
- **Sidecar Arm:** `Z0-64` (starts from canonical unperturbed initial latch state $Z_0$, adopts current $K$ from MAIN, runs for 64 physical integration steps under isolated `rng_sidecar`).
- **Invariance Rule:** The sidecar clone operates in isolated memory. Upon destruction, `MAIN Z`, `MAIN K`, and `rng_main` must remain 100% bit-exact identical to their pre-fork states.

### 1.4. Checkpoint / Exact Resume
- **File:** `pqt_qcat/memory/checkpoint_io.py` (SHA-256: `71b8bc535d35517e7bb6ab70b0ed4640fae9591d3ff4690e19033aa8f8c5ecd2`)
- Exact serialization of numpy arrays and bit generator RNG states without precision loss.

---

## 2. Inventory of Digital Amoeba Legacy & Reuse Assessment

### 2.1. Historical Artifact Analysis
- An exhaustive search across the entire repository history, commits, and local workspace confirmed that `pqt_amoeba_game_v0_1` / `AMOEBA-E0` was an early conceptual specification for an embodied agent moving in a 2D discrete grid world:
  - Actions: 4 motor directions (`UP=1, DOWN=2, LEFT=3, RIGHT=4`), plus `WAIT`, `PROBE`, `IDLE`.
  - Environmental regimes: `NORMAL → TRAP_A → NORMAL → TRAP_B → NORMAL → TRAP_A_RETURN`.
  - Brain interface: `Brain.reset()`, `Brain.act(obs)`, `Brain.on_event(event)`, `Brain.on_outcome(outcome)`, `Brain.checkpoint()`.
- Existing frozen environments in `PQT-QCAT`:
  - `RetryEnvironment` (`pqt_qcat/memory/functional_sidecar_b2a.py`): tracks query state, returns positive reward (+1.0) on match or progressive penalty ($-0.25 \times (1 + \text{streak})$) on error, shifting by an offset upon success.
  - `PartiallyObservedGraphEnv` (`pqt_qcat/dream/counterfactual_env.py`): tracks relational transition chains.

### 2.2. Reuse & Adapter Strategy (`life003/amoeba_adapter.py`)
- Rather than constructing an unvetted heavyweight game engine from scratch, we build a compact, deterministic `AmoebaWorld` and `AmoebaBrain` adapter that:
  1. Implements the standard `Brain` API:
     - `reset()`: resets the organism to initial state while preserving the immutable core structure.
     - `act(observation)`: routes observation through the 2-channel SANKALPA policy.
     - `on_event(event)`: registers external environmental stimuli.
     - `on_outcome(reward, next_obs)`: updates pending records and ledger.
     - `checkpoint()`: delegates to `life003/checkpoint.py`.
  2. Maps the 4 motor classes ($1..4$) directly to `W_readout` outputs ($1..4$).
  3. Provides a clean deterministic 2D grid world with energy homeostasis ($G_{\text{energy}} \in [0, 1]$), food resources, hazards, and sensory observation vectors (8D one-hot tokens), with zero oracle leaks.

---

## 3. Architecture of `life003` Runtime Package

### 3.1. Complete Organism State ($S_t$)
$$S_t = (Z, K, H, G, \text{Plan}, \text{Pending}, \text{World}, \text{Ledger}, \text{RNG}, \text{Clock})$$

- **Clock (`life003/clock.py`):**
  - `core_tick`: counts total physical integration steps ($dt = 0.01$).
  - `world_tick`: counts discrete environment interaction steps.
  - `decision_tick`: counts SANKALPA deliberations.
  - `feedback_tick`: counts outcome evaluation steps.
- **RNG Isolation:**
  - Three independent `numpy.random.Generator` streams: `rng_main`, `rng_sidecar`, `rng_world`.
  - Quarantined M5 seeds `[101, 102, 103, 104, 105]` are strictly rejected.
- **Hypotheses Space $H$ (`life003/hypotheses.py`):**
  - Capacity: $|H| \le 8$.
  - Two orthogonal axes:
    - Truth Justification: $v_{\text{just}} \in \{\text{UNVERIFIED}, \text{SUPPORTED}, \text{CONFLICTED}, \text{REFUTED}\}$.
    - Spanda Freshness: $s_{\text{fresh}} \in \{\text{FRESH}, \text{AGING}, \text{OSSIFIED}\}$.
  - Counters: `inertia_m_phi_uses` and `ticks_since_audit`. Transition to `AGING` occurs when uses $\ge 8$ OR ticks $\ge 50$. Transition to `OSSIFIED` occurs when uses $\ge 32$ OR ticks $\ge 200$.
  - **Invariance:** Staleness / aging **never** changes $v_{\text{just}}$ to `REFUTED`.
- **Sākṣin Ledger (`life003/ledger.py`):**
  - Append-only hash chain: `entry_hash = sha256(prev_hash + seq + clock_snapshot + record_type + payload)`.
  - Strictly deterministic: excludes wall-clock time (`time.time()`, `datetime`). Uses monotonic sequence `seq = 0, 1, 2, ...` and `LifeClock`.
  - Genesis entry (`seq = 0`): `record_type = "CONFIG_REGISTERED"`, payload contains SHA-256 of `configs/life003_t1_prereg.yaml`.
- **Endogenous Trigger (`life003/trigger.py`):**
  - Heuristic score:
    $$\text{score} = w_S \cdot S + w_U \cdot U + w_E \cdot E + w_R \cdot R$$
    where $w_S = 0.35, w_U = 0.35, w_E = 0.15, w_R = 0.15$.
  - Threshold: $\theta = 0.45$.
  - $U = H(p) / \log(4)$ (normalized policy entropy).
  - $S = \min(\|Z_{\text{sens\_2}}\| / 2.5, 1.0)$ (sensory prediction residual).
  - $E = 1.0 - G_{\text{energy}}$ (energy deficit).
  - $R$ = unresolved goal / plan obligation.
- **Sidecar Deliberation Pool (`life003/sidecar_pool.py`):**
  - Spawns $\le 3$ parallel $Z_0\text{-}64$ clones with copy of $K_{\text{main}}$ and clean $Z_0$.
  - Budget guards: $\le 192$ ticks/event, $\le 150$ total deliberations in T1, cooldown of 20 ticks.
  - Verifies bit-exact pre/post identity of $Z_{\text{main}}$, $K_{\text{main}}$, and `rng_main`.
- **Two-Channel SANKALPA (`life003/sankalpa.py`):**
  - **Epistemic Channel:** $d_{\text{ep}} \in \{\text{COMMIT}, \text{CLASS\_ONLY}, \text{ABSTAIN}\}$.
    - If `witness_ref is None`: strictly `ABSTAIN` (HonestPause, `ZeroWitnessCommit == 0`).
    - If `witness_ref` is present and support $\ge 0.60$: permits `COMMIT`.
  - **Operational Channel:** $a_{\text{act}} \in \{1, 2, 3, 4, \text{PROBE}, \text{WAIT}, \text{IDLE}\}$.
    - Epistemic `ABSTAIN` does **not** block safe `PROBE`, `WAIT`, or motor actions.
- **Physical Scheduler (`life003/scheduler.py`):**
  - Handles the tick state machine without overlapping step execution.
  - Idle: exactly 1 step of `SixEchelonCore.step(None, write_gate=0.0)`.
  - Query: `QNUFAgent.begin_query(q)` executes exactly 8 ticks.
  - Exact atomic feedback in T1.

---

## 4. Exact Checkpoint / Resume Verification Protocol
- At tick 5000 of the 10,000-tick run, an atomic checkpoint is serialized to disk (`checkpoint_5000.npz` and `checkpoint_5000_meta.json`).
- A clean, freshly initialized runtime instance restores the state from this checkpoint and runs from tick 5001 to tick 10000.
- Every state element ($Z$, $K$, $H$, $G$, Pending, World, Ledger hashes, RNG bit states, Clock) is compared against the continuous uninterrupted run.
- Zero discrepancy (0 bits) is required for `T1_VALID`.
