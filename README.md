# PQT-QCat Cognitive Architecture (LIFE-002)

[![PQT-QCAT CI](https://github.com/tocataskyrimova-lang/PQT-QCAT/actions/workflows/ci.yml/badge.svg)](https://github.com/tocataskyrimova-lang/PQT-QCAT/actions/workflows/ci.yml)
[![Pre-registration Tag](https://img.shields.io/badge/prereg--tag-life002--prereg--v1-blue.svg)](https://github.com/tocataskyrimova-lang/PQT-QCAT/releases/tag/life002-prereg-v1)
[![Tests](https://img.shields.io/badge/pytest-60%2F60%20passed-brightgreen.svg)](pqt_qcat/tests)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)

Continuous, dynamical cognitive architecture based on **Parametric Quasi-Topological Quantum-like Continuous Attractor Trajectories (PQT-QCat)**.

---

## 1. Architectural Highlights

PQT-QCat implements an end-to-end continuous dynamical core operating in $\mathbb{R}^{48}$ with autonomous memory consolidation, conservative vortex rotation fields, and audit-verifiable provenance:

- **48D Six-Echelon Dynamic Core:**
  - $E_1$: Primary sensory receiver ($Z_{\mathrm{sens}, 1}$)
  - $E_2$: Prediction residual and eligibility trace ($Z_{\mathrm{sens}, 2}$)
  - $E_3$: Write-gated cue latching ($Z_{\mathrm{hold}, 1}$)
  - $E_4$: Protected deep retention with reduced mobility $\Gamma_{\mathrm{hold}}$ ($Z_{\mathrm{hold}, 2}$)
  - $E_5$: Endogenous context attractor basins ($Z_{\mathrm{ctx}, 1}$)
  - $E_6$: Rule shift detection & exploration trigger ($Z_{\mathrm{ctx}, 2}$)
- **Conservative Vortex Dynamics:** Continuous antisymmetric rotation operator $A_I(w_0) = -A_I(w_0)^T$ conserving energy along gradient orbits ($g^T A_I g \equiv 0$).
- **Strictly Frozen Readout Head:** Linear projection matrix $W_{\mathrm{readout}}$ is permanently frozen after initialization. All representation learning and adaptation occurs exclusively inside core dynamics $\theta_{\mathrm{core}} = \{K, B\}$.
- **Witness Ledger Provenance:** Cryptographically isolated memory separating verified real observations ($B_{\mathrm{real}}$) from unverified counterfactual dream hypotheses ($B_{\mathrm{dream}}$) with Maya uncertainty weighting.
- **Three-Gate Safe Consolidation & Anchor Guard:** Every dream candidate must pass:
  1. *Provenance & Consistency Gate* (non-contradiction of $B_{\mathrm{real}}$)
  2. *Structural Bounds Gate* (finiteness and domain validity)
  3. *Anchor Guard Gate* (shadow-copy evaluation with automatic rollback on degradation $> 0.02$).

---

## 2. Repository Structure

```text
PQT-QCAT/
├── pqt_qcat/
│   ├── core/                  # 48D core, energy, antisymmetric vortex operator, Heun integrator
│   ├── dream/                 # Symmetric counterfactual DREAM generator & 3-gate consolidation
│   ├── memory/                # Witness Ledger, provenance tracking, and M3 diagnostics
│   ├── plasticity/            # Non-local eligibility traces & feedback plasticity
│   ├── evaluator/             # Independent M5 Evaluator (H0–H6) and SHA-256 manifest utilities
│   └── tests/                 # Full pytest suite (60 unit and regression tests)
├── prereg/
│   ├── LIFE002_PREREG_001.yaml         # Formal pre-registration protocol
│   ├── MANIFEST_SHA256_LIFE002.json    # Cryptographic SHA-256 manifest of codebase
│   └── MANIFEST_SHA256_LIFE002.txt     # Plain text manifest with file checksums
├── results/                   # Benchmark artifacts and audit verification records
├── legacy_life_001/           # Historical materials from LIFE-001 phase
├── verify_manifest.py         # Automated manifest verification script
├── lock_historical.py         # Verification lock for historical files
├── run_m1_benchmarks.py       # Canonical math and Heun integrator benchmarks
├── run_m2_benchmarks.py       # 48D Six-Echelon and core plasticity benchmarks
├── run_m3_benchmarks.py       # Endogenous context and twin-histories benchmarks
├── run_m4_benchmarks.py       # DREAM generator, control models audit, and T4/T5 benchmarks
├── run_m5_evaluation.py       # Milestone M5 Independent Evaluation runner
├── M5_EVALUATION_PROTOCOL.md  # Detailed M5 independent evaluation protocol
├── ARCHITECTURE_LIFE_002.md   # Complete architectural specification
└── requirements.txt           # Python dependencies
```

---

## 3. Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/tocataskyrimova-lang/PQT-QCAT.git
cd PQT-QCAT

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Verification & Testing

### Run SHA-256 Manifest Verification
Validates that all source code, configurations, and protocols match the cryptographic manifest:
```bash
python verify_manifest.py
```

### Run Full Test Suite (60 tests)
Executes all regression tests across mathematical invariants, six-echelon core, leak-prevention, control audits, and evaluator:
```bash
pytest pqt_qcat/tests -v
```

### Run M5 Independent Evaluator (Calibration Mode)
Evaluates hypotheses H0–H6 on calibration seed 42 with raw JSONL logging:
```bash
python run_m5_evaluation.py --seed 42 --episodes 1000
```

---

## 5. Pre-registration Protocol & Sealed Seeds Gate

- **Pre-registration Protocol:** [`prereg/LIFE002_PREREG_001.yaml`](prereg/LIFE002_PREREG_001.yaml)
- **Immutable Tag:** [`life002-prereg-v1`](https://github.com/tocataskyrimova-lang/PQT-QCAT/releases/tag/life002-prereg-v1)
- **Sealed Seeds Isolation:** Seeds `[101, 102, 103, 104, 105]` are strictly sealed behind programmatic authorization barriers. Any attempt to execute them without formal unsealing authorization raises an immediate `PermissionError`.
- **Calibration Seeds:** Seeds `[42, 11, 12, 13]` are open for dry-runs, tuning, and CI smoke testing.

---

## 6. License & Citation

Distributed for research, scientific audit, and independent evaluation of continuous cognitive architectures.
