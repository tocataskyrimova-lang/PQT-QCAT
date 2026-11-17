# K-RESPONSE-001 — bounded exploratory protocol, 2026-11-17

Frozen before execution in this working tree. This is NOT LIFE002 M5,
NOT a nullspace audit and NOT evidence for DREAM or a specific advantage of A.
Historical runs and preregistrations remain unchanged.

- Source baseline: `2291e34ad330427bce16f9b314dff6c43a5dc8f9`.
- Only calibration seeds: `[11,12,13,42]`; first six maps from generator 170700.
- Same original wake update, fixed B/W, vortex enabled, float64, fixed Heun h=.01.
- BLOCKED: 25 epochs old, another 20 epochs old, then 20 epochs new.
  Save K0 immediately before new training (event 135), K1 after it (event 195).
  No tail, sleep, hyperparameter search or final/replication seeds.
- Intervene on K alone: K(lambda)=K0+lambda*(K1-K0), lambda=[0,.25,.5,.75,1].
  Every query starts at the same quiescent Z, lasts eight ticks, uses fixed B/W.
  Score old tokens 1..3 and new tokens 4..6 with fixed mapping labels.
- Record all four logits, accuracy, SIGNED true-target margin and NLL.
  Log raw training events and update hashes; save initial/prefix/before/after K,
  corresponding Z, B, W, parameters and RNG state. No pickle.
- Estimate only the directional logit derivative at K0 with central differences
  eps=[.01,.001,.0001]. Record convergence and the full-step linearization residual.
  This is a derivative from R^(48x48) to R^4 per token, not a 48x48 Jacobian.
- Validate finite values, invariant B/W, endpoint reconstruction, exact restored
  endpoint logits and exact K rollback. Abort on invariant failure.
- Aggregate maps WITHIN each seed first. Report four seed means and descriptive
  95% Student-t intervals (df=3); these reused calibration seeds are not a new
  confirmatory replication. No significance-based selection or stopping.
- Scientific outcome may be no harmful interference. Do not increase load just
  to force a positive result. Sensitivity to K does not establish useful dormant
  knowledge, readout bottlenecks, a guard nullspace, or a need for sleep.

Execution: `python -B -m pqt_qcat.memory.exp_k_response_001 --output <new-directory>`.
Existing output directories are refused. Source hashes and local patch are saved
before training; a `COMPLETE.json` marker is written only after all checks pass.
