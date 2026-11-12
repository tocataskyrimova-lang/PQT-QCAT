# MILESTONE M5: INDEPENDENT EVALUATION PROTOCOL (H0–H6)
**Document Version:** 1.0.0
**Target Architecture:** PQT-QCat LIFE-002 (48D Six-Echelon Dynamic Core)
**Git Worktree:** `C:\AGI\PQT_QCAT_AUDIT\PQT_QCat_LIFE_002` (branch `pqt-qcat-life-002`)
**Author:** Antigravity (DeepMind)
**Reviewer:** ILORI (Lead Reviewer)
**Pre-registration Reference:** `prereg/LIFE002_PREREG_001.yaml`
**Pre-registration Tag:** `life002-prereg-v1`

---

## 1. Overview & Objective

Milestone M5 represents the final, blinded, single-run empirical validation of the PQT-QCat continuous cognitive architecture. The objective is to evaluate hypotheses **H0 through H6** on the pre-registered sealed seeds `[101, 102, 103, 104, 105]`, using a fully independent, decoupled evaluator with at least **20,000 episodes per scenario/seed**, strictly equalized compute budgets, cryptographic RNG fingerprints, and raw trial-by-trial JSON Lines logging.

---

## 2. Pre-registered Hypotheses & Evaluation Metrics

In accordance with `prereg/LIFE002_PREREG_001.yaml`, all pre-registered thresholds are permanent and cannot be modified post-hoc:

| ID | Гипотеза | Метрика | Порог | Условие подтверждения |
| :--- | :--- | :--- | :---: | :--- |
| **H0** | Численная стабильность | `invalid_count`, `skew_res` | $0$, $< 10^{-10}$ | $0$ NaN/Inf/несошедшихся шагов; $\|A_I(w0) + A_I(w0)^T\| < 10^{-10}$; $g^T A g < 10^{-12}$. |
| **H1** | Удержание метки (Hold/Latch) | `median_acc_gap_300` | $\ge 0.90$ | Точность $\ge 90\%$ при задержке $300$ тактов на полном 48D-ядре; не хуже noA при gap=100. |
| **H2** | Эндогенная история в ядре | `twin_history_acc` | $\ge 0.85$ | Точность $\ge 85\%$ при замороженной матрице $W_{\mathrm{readout}}$; падение $\ge 40$ п.п. при рандомизации ядра. |
| **H3** | Безопасность сна (Sleep Safety) | `median_degradation` | $\le 0.02$ | Деградация точности реальных якорей $\le 2$ п.п. после 1000 тактов сна для всех политик. |
| **H4** | Прирост от DREAM над Replay | `pqt_minus_guarded_replay` | $\ge 0.05$ | Прирост точности обобщения $\ge 5$ п.п. на $\ge 4/5$ финальных сидов при равных бюджетах. |
| **H5** | Каузальная роль оператора $A$ | `pqt_minus_noA_dream` | $\ge 0.03$ | Прирост $\ge 3$ п.п. на $\ge 4/5$ сидов при идентичных RNG-отпечатках. |
| **H6** | Изоляция происхождения памяти | `false_promotions` | $0$ | Ровно $0$ ложных промоушенов из $B_{\mathrm{dream}}$ в $B_{\mathrm{real}}$ и $0$ утечек меток тестовой среды. |

---

## 3. Контроль вычислительных бюджетов (Compute Budget Symmetry)

Для исключения артефактов неравномерности вычислительных ресурсов при проверке **H4** (`PQT_dream` vs `guarded_replay`) и **H5** (`PQT_dream` vs `noA_dream`), протокол фиксирует строго симметричные бюджеты:

1. **Физическая интеграция:** Ровно **1000 тактов** непрерывной интеграции (`core.step(x_t=None, write_gate=0.0)`) во время сна.
2. **Предъявление событий:** Ровно **25 событий** презентации (семена сна / повторы реальных событий).
3. **Шаги консолидации:** Ровно **25 шагов** анализа кандидатов и попыток обновления весов ядра.
4. **Бюджет тестирования:** Ровно **20,000 тестовых эпизодов** на сценарий и сид.
5. **Инструментальная валидация:** Независимый оценщик программно проверяет равенство бюджетов (`assert physical_integration_ticks == 1000`, `assert presentations == 25`, `assert updates == 25`) до вычисления разности точности.

---

## 4. Верификация условий через RNG Fingerprints

Для подтверждения того, что оператор $A$ является единственным фактором различия между PQT и noA, оценщик фиксирует 4 криптографических отпечатка SHA-256:
- `initial_state_hash`: SHA-256 начального состояния ядра ($Z_0, K_0, B_0, W_{\mathrm{readout}}$);
- `training_events_hash`: SHA-256 последовательности обучающих событий в фазе бодрствования;
- `dream_seed_events_hash`: SHA-256 выбранных семян сновидений из $B_{\mathrm{real}}$;
- `dream_noise_stream_hash`: SHA-256 всей последовательности теплового шума $\eta(t)$ во время свободной эволюции.

В парных прогонах PQT и noA совпадение всех 4 хешей является обязательным предусловием для оценки H5.

---

## 5. Архитектура независимого оценщика (Independent Evaluator)

1. **Модуль оценщика:** [pqt_qcat/evaluator/independent_evaluator.py](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/evaluator/independent_evaluator.py).
   - Не зависит от тренировочных пайплайнов и вспомогательных скриптов;
   - Защищён аппаратным барьером `_check_seed_gate`, блокирующим запуск сидов 101–105 без флага `--unseal-final-seeds`.
2. **Формат сырых логов (Raw Logging):**
   - Файлы сохраняются в `results/raw_logs/h{0-6}_seed_{seed}.jsonl`.
   - Каждая запись содержит:
     `{"episode": int, "step": int, "input": list, "pred": int, "target": int, "correct": bool, "z_norm": float, "rng_hash": str}`.
3. **Исполняемый раннер:** [run_m5_evaluation.py](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/run_m5_evaluation.py).

---

## 6. Криптографический манифест (SHA-256 Manifest)

Полная кодовая база, файлы пререгистрации, тесты и конфигурации зафиксированы в криптографическом манифесте:
- **JSON-манифест:** `prereg/MANIFEST_SHA256_LIFE002.json`
- **Текстовый манифест:** `prereg/MANIFEST_SHA256_LIFE002.txt`
- **Скрипт валидации:** `python verify_manifest.py` (автоматическая сверка 100% файлов).

---

## 7. Запечатанные сиды и процедура однократного запуска

- **Калибровочные сиды (открыты для калибровки):** `[42, 11, 12, 13]`.
- **Запечатанные сиды (закрыты до авторизации):** `[101, 102, 103, 104, 105]`.
- **Политика однократного прогона:** После получения официального допуска ILORI к однократному финальному эксперименту, раннер запускается ровно **один раз** для сидов 101–105. Все исходы (положительные, нулевые или отрицательные) фиксируются как окончательный научный вердикт проекта LIFE-002.
