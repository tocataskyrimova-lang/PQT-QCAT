# ILORI AUDIT REPORT: CONTROL-FIX-002
**Milestone:** Pre-M5 Final Gate / Control Models Independent Audit
**Date:** 2026-11-13
**Git Worktree:** `C:\AGI\PQT_QCAT_AUDIT\PQT_QCat_LIFE_002` (branch `pqt-qcat-life-002`)
**Author:** Antigravity (DeepMind)
**Target Reviewer:** ILORI (Lead Reviewer)

---

## 1. Executive Summary & Resolution of Findings

В соответствии с мандатом **ILORI / CONTROL-FIX-002 — LAST PRE-M5 CORRECTION**, устранены все три оставшихся несоответствия:

1. **Точный учёт бюджета сна (Sleep Budget Accounting):**
   - Устранён дефицит 250 тактов в replay-контролях (`naive_replay`, `guarded_replay`). Ранее обновление матрицы параметров необоснованно приравнивалось к 10 тактам численной интеграции (`idle_ticks - dream_cycles * 10 = 750`). Теперь все 6 политик сна (`no_replay`, `state_only`, `naive_replay`, `guarded_replay`, `noA_dream`, `PQT_dream`) выполняют **ровно 1000 физических тактов интеграции** (`core.step(x_t=None, write_gate=0.0)`).
   - Внедрён раздельный учёт:
     - `physical_integration_ticks`: ровно 1000 для всех 6 политик;
     - `event_presentations`: 0 для пассивных, 25 для replay и dream;
     - `consolidation_steps`: число попыток/шагов обновления весов;
     - `total_compute_steps`: полный объём шагов во всех фазах сна.
   - Предикат `step_budget_exact` теперь вычисляется строгим равенством `physical_integration_ticks == idle_ticks` (1000) без допусков и смещений.

2. **Строгий аудит оператора $A$ (Vortex Operator Audit):**
   - Устранено любое автоматическое присвоение `vortex_verified = True`.
   - Для каждой политики задано и проверяется ожидаемое состояние вихря (`expected_enable_vortex: True` для PQT и базовых моделей, `False` для `noA`).
   - На каждом шаге сна с `write_gate=0.0` проверяется флаг `core.config.enable_vortex`.
   - Физическая сила вихря $\|F_{\mathrm{vortex}}\|$ измеряется на стандартизованном тестовом векторе $Z_{\mathrm{probe}}$:
     - Для PQT-политик подтверждено $\|F_{\mathrm{vortex}}\| = 5.110282 > 10^{-3}$;
     - Для `noA_dream` подтверждено строгое тождество $\|F_{\mathrm{vortex}}\| \equiv 0.000000 < 10^{-15}$.
   - Документированная таблица полностью согласована с исполняемым кодом и артефактом [results/control_audit_results.json](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/results/control_audit_results.json).

3. **Гарантия тождественности условий в парном сравнении PQT/noA (RNG Fingerprints):**
   - Обеспечена 100% эквивалентность начальных условий между PQT и noA:
     - Идентичное начальное состояние ядра $Z_0$ и весов $K_0, B_0, W_{\mathrm{readout}}$;
     - Идентичный поток обучающих событий в фазе бодрствования;
     - Идентичные seed-события сна (`seed_event`), выбираемые генератором;
     - Идентичная псевдослучайная траектория теплового шума $\eta(t)$ во время свободной эволюции.
   - Внедрены проверяемые криптографические хеши SHA-256 (`rng_fingerprints`):
     - `initial_state_hash`
     - `training_events_hash`
     - `dream_seed_events_hash`
     - `dream_noise_stream_hash`
   - Подтверждено: все 4 хеша побитово совпадают между PQT и noA, гарантируя, что **единственным каузальным фактором различия является оператор $A$ ($A = A_I$ vs $A = 0$)**.

4. **Устранение противоречия по метрике T5 (Фактические результаты JSON):**
   - В артефакте [results/m4_dream_verification.json](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/results/m4_dream_verification.json) фактическая точность композиционного вывода в сценарии T5 составляет **0.0% (`comp_acc = 0.0`, `fp_rate = 0.0`)** как для `guarded_replay`, так и для `noA_dream` и `PQT_dream`.
   - Ошибочное словесное утверждение о «50%» полностью отозвано и исправлено: на базовом графе отношений перенос контрфактических связей через одну лишь матрицу $B$ без адаптации связности $K$ даёт нулевой эффект. Исходный нулевой/отрицательный результат пререгистрации H5 сохранён без прикрас.

---

## 2. Сводная таблица эмпирического аудита контрольных условий (CONTROL-FIX-002)

Все значения получены прямым измерением в рантайме (seed=42, `idle_ticks=1000`):

| Политика | Физ. такты интеграции | Такты сна целевые | Предъявления событий | Шаги обновления ядра | $\max \|u_{\mathrm{ext}}\|$ | $\|\Delta W_{\mathrm{readout}}\|$ | Ожидаемый вихрь $A$ | Фактический флаг вихря | Измеренная сила $\|F_{\mathrm{vortex}}\|$ | Сохранение якорей | Ложные промоушены | Вердикт аудита |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_replay` | **1000** | 1000 | 0 | 0 | **0.0** | **0.0** | $A = A_I$ | True | **5.110282** | 100% (деградация 0.0) | 0 | **COMPLIANT** |
| `state_only` | **1000** | 1000 | 0 | 0 | **0.0** | **0.0** | $A = A_I$ | True | **5.110282** | 100% (деградация 0.0) | 0 | **COMPLIANT** |
| `naive_replay` | **1000** | 1000 | 25 | 25 | **0.0** | **0.0** | $A = A_I$ | True | **5.110282** | 100% (деградация 0.0) | 0 | **COMPLIANT** |
| `guarded_replay` | **1000** | 1000 | 25 | 25 | **0.0** | **0.0** | $A = A_I$ | True | **5.110282** | 100% (деградация 0.0) | 0 | **COMPLIANT** |
| `noA_dream` | **1000** | 1000 | 25 | 25 | **0.0** | **0.0** | $A \equiv 0$ | **False** | **0.000000** | 100% (деградация 0.0) | 0 | **COMPLIANT** |
| `PQT_dream` | **1000** | 1000 | 25 | 25 | **0.0** | **0.0** | $A = A_I$ | **True** | **5.110282** | 100% (деградация 0.0) | 0 | **COMPLIANT** |

**Общий вердикт контрольного аудита:** `overall_control_verdict = COMPLIANT` (все 6 политик подтверждены измеренными величинами).

---

## 3. Проверка парной эквивалентности PQT vs noA (RNG Fingerprints)

При парном запуске `PQT_dream` и `noA_dream` на калибровочном сиде 42 получены следующие криптографические контрольные суммы (SHA-256):

```json
{
  "PQT_dream": {
    "initial_state_hash":     "8e8cd2bf75aada0d71f5186eff1597f572b29091f9eceec2f46f5f4822e45ac7",
    "training_events_hash":   "38f3e75e659001a164fd16fa8c7fe867742da0a30d421701b4b24c8210a5f8cf",
    "dream_seed_events_hash": "71a6666481681692f46cbf3f80a7751ba2c4a91f05fea181e17ddec3c9790fcc",
    "dream_noise_stream_hash":"ae536b779d4d331c57264571150e07d08358949536d184070edd484ccb3cf2aa",
    "enable_vortex": true
  },
  "noA_dream": {
    "initial_state_hash":     "8e8cd2bf75aada0d71f5186eff1597f572b29091f9eceec2f46f5f4822e45ac7",
    "training_events_hash":   "38f3e75e659001a164fd16fa8c7fe867742da0a30d421701b4b24c8210a5f8cf",
    "dream_seed_events_hash": "71a6666481681692f46cbf3f80a7751ba2c4a91f05fea181e17ddec3c9790fcc",
    "dream_noise_stream_hash":"ae536b779d4d331c57264571150e07d08358949536d184070edd484ccb3cf2aa",
    "enable_vortex": false
  }
}
```

- Побитовое совпадение `initial_state_hash`: **ДА (100%)**
- Побитовое совпадение `training_events_hash`: **ДА (100%)**
- Побитовое совпадение `dream_seed_events_hash`: **ДА (100%)**
- Побитовое совпадение `dream_noise_stream_hash`: **ДА (100%)**
- Единственное различие: `enable_vortex` (`True` vs `False`).

---

## 4. Фактические результаты сценариев M4 на калибровочном seed=42

### Сценарий T4: Безопасность сна (Dream Safety, 1000 тактов)
- Все политики сохранили 100% точность на реальных якорях:
  - `no_replay`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0);
  - `state_only`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0);
  - `naive_replay`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0);
  - `guarded_replay`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0);
  - `noA_dream`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0);
  - `PQT_dream`: деградация 0.0 (аккуратность 1.0 $\to$ 1.0).
- **Статус H3-SLEEP-SAFE:** CONFIRMED.

### Сценарий T5: Контрфактический синтез (Counterfactual Synthesis)
- Граф `relational_chain`:
  - `guarded_replay`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `noA_dream`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `PQT_dream`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `dream_gain_over_replay`: 0.0
  - `causal_vortex_gain`: 0.0
- Граф `branching_non_bijective`:
  - `guarded_replay`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `noA_dream`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `PQT_dream`: `comp_acc = 0.0%`, `fp_rate = 0.0%`
  - `dream_gain_over_replay`: 0.0
  - `causal_vortex_gain`: 0.0
- **Научный вердикт H4 (DREAM Gain over Replay):** NOT CONFIRMED (0.0 на базовом протоколе).
- **Научный вердикт H5 (Causal Vortex Operator A Gain):** INCONCLUSIVE / NULL (0.0 на базовом протоколе).

---

## 5. Полный статус набора регрессионных тестов (pytest)

Запущено 53 теста (включая 8 тестов в `test_control_audit.py`):
```text
================== 53 passed, 1 warning in 340.05s (0:05:40) ==================
```
Все 53 теста завершились с вердиктом **PASSED** (0 сбоев, 0 ошибок).

---

## 6. Защитные инварианты

1. **Замок исторических файлов:** Проверен через `python PQT_QCat_LIFE_002/lock_historical.py --verify` $\to$ **100% UNCHANGED**.
2. **Пререгистрация:** `prereg/LIFE002_PREREG_001.yaml` **НЕ ИЗМЕНЯЛАСЬ**.
3. **Запечатанные сиды:** Seeds `[101, 102, 103, 104, 105]` **НЕ ТРОНУТЫ** (использовались только `42, 11, 12, 13`).
4. **Тег Git:** Финальный тег `life002-prereg-v1` **НЕ СОЗДАВАЛСЯ** в ожидании официального допуска ILORI.
