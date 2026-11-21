# LIFE-003 / T1-EPISTEMIC-PATCH-001: Отчёт о семантическом патче эпистемического контура

- **Репозиторий:** `tocataskyrimova-lang/PQT-QCAT`
- **Рабочая ветка:** `exp/life-003-continuous-thought-001`
- **Дата:** 21 ноября 2026 г.
- **Основание:** Повторный независимый аудит Илори через NLA2
- **Исполнитель:** Astra/Codex (Antigravity Agent)
- **Эпистемический статус:** `W_CANDIDATE`, `proof_claim = false`
- **Мандат:** Строго **T0 + T1** (T2/T3/T4 не запускались и остаются заблокированными).

---

## 1. СТРОГИЙ ВЕРДИКТ: `EPISTEMIC_PATCH_VALID`

Все 6 пунктов семантического замечания FIX-4 устранены в строгом соответствии с требованиями PQT-KC-001 v0.3:
- Набор автотестов расширен до **28 тестов** (19 оригинальных T1 + 5 из T1-FIX-001 + 4 новых теста семантического патча), все **28/28 PASS** (`100%`).
- Замороженный базис LIFE-002 (`pqt_qcat/core/`, `pqt_qcat/memory/`, `pqt_qcat/plasticity/`) имеет **0 байт изменений** относительно коммита `06fbd5b7e6fcc224619d19b78ded716f74694a08`.

---

## 2. ПОДРОБНЫЙ РАЗБОР УСТРАНЁННЫХ ПУНКТОВ (1 — 6)

### Пункт 1: Разделение подлинности записи Ledger и доказательной поддержки утверждения
* **Проблема:** Ранее подлинность любой записи в Леджере (`OBSERVED`, `AUDIT_SHADOW`) при передаче внешнего числа `witness_support_score >= 0.60` формально пропускалась в `COMMIT`.
* **Решение:** В `SankalpaGate.evaluate_epistemic_channel` введено строгое разделение:
  * Запись `OBSERVED` подтверждает лишь факт регистрации внешнего стимула/наблюдения. Она допускает максимум эпистемический статус `CLASS_ONLY`, но **никогда не разрешает `COMMIT`**.
  * Запись `AUDIT_SHADOW` является внутренней телеметрией среды и строго переводит эпистемический канал в `ABSTAIN`.
  * Эпистемический `COMMIT` разрешён **исключительно** для записей типа `VERIFIED_FEEDBACK`.

### Пункт 2: Строгая проверка свидетельств `VERIFIED_FEEDBACK`
* **Решение:** Для записи `VERIFIED_FEEDBACK` в `SankalpaGate` выполняется 4-уровневая проверка:
  1. **Происхождение (Provenance):** Поле `provenance` / `hypothesis_id` в payload обязано строго совпадать с проверяемой гипотезой (или базисом события).
  2. **Результат (Outcome / Action):** Зафиксированное в фидбеке действие обязано совпадать с кандидатным действием гипотезы, а статус подтверждения `confirmed` обязан быть `True` (отсутствовать флаги `REFUTED` / `REJECTED` / `FAILED`).
  3. **Область применимости (Applicability Domain):** Контекст / область применимости (`applicability_domain` / `domain` / `query`) обязаны покрывать контекст гипотезы.
  4. **Оценка поддержки (Support Score):** Значение берётся **исключительно из подписанного payload записи Леджера** (`payload["support_score"]`). Произвольно переданный вызывающей стороной аргумент `witness_support_score` более не принимается как достаточное основание.
* В `SaksinLedger` добавлен метод `append_verified_feedback(...)` с валидацией всех указанных полей.

### Пункт 3: Fail-Closed при отсутствии Ledger
* **Решение:** При `require_structured_witness=True` (по умолчанию `True` в pre-reg конфигурации) отсутствие переданного экземпляра `ledger` (`ledger is None`) вызывает безусловный fail-closed: `return "ABSTAIN", True` со срабатыванием сторожевого гейта Zero-Witness.

### Пункт 4: Расширенная проверка целостности Sākṣin Ledger
* **Решение:** В `SaksinLedger.validate_integrity()` добавлены обязательные проверки:
  1. `self.seq == len(self.records)` — счётчик записей обязан строго равняться длине списка.
  2. `self.prev_hash == expected_prev` — сохранённый хвостовой хеш обязан бит-в-бит совпадать с вычисленным хешем последней записи (или `"0"*64` для пустого леджера).
* Любая рассинхронизация `seq` или `prev_hash` делает `validate_integrity() == False`.

### Пункт 5: Отрицательный тест на `OBSERVED` без feedback
* **Решение:** Добавлен тест `test_epistemic_patch_observed_and_shadow_cannot_authorize_commit`:
  * Настоящая криптографически валидная запись `OBSERVED` с искусственно переданным внешним скором `witness_support_score = 0.99` возвращает `CLASS_ONLY` и строго ассертится `d_ep != "COMMIT"`.
  * Запись `AUDIT_SHADOW` с любым скором возвращает `ABSTAIN` (`zero_witness_tripped = True`).

### Пункт 6: Исправление терминологии Sidecar Deliberation
* **Проблема:** Термин `MULTI_CANDIDATE_OUTCOME_COMPARISON` создавал ложное впечатление, что сайдкар сравнивает доказанные реальные последствия действий в мире среды.
* **Решение:**
  * Термин заменён на честный и точный: `MULTI_BRANCH_PREDICTION_ENTROPY_SELECTION`.
  * Добавлен явный критерий отбора: `selection_criterion: "MINIMAL_NEURAL_PREDICTION_ENTROPY"`.
  * В возвращаемую структуру и документацию добавлен явный эпистемический дисклеймер:  
    `epistemic_disclaimer: "Comparison of candidate neural prediction entropies; not a validated world outcome comparison."`
  * В T1 физическая среда не моделируется сайдкаром наперёд; сайдкар лишь сравнивает энтропию выходных логитов нейронного ядра при альтернативных воздействиях.

---

## 3. СВОДНАЯ ТАБЛИЦА ТЕСТОВ (28 / 28 PASS)

| № | Тест | Назначение / Проверяемый инвариант | Статус |
|---|---|---|:---:|
| 1–19 | Тесты этапа T1 (Baseline) | Часы, Z-непрерывность, K-инвариантность, SANKALPA, Леджер | **PASS** |
| 20 | `test_fix1_full_state_checkpoint_and_resume_pending_event` | Восстановление при активном `pending` event | **PASS** |
| 21 | `test_fix1_full_state_comparison_after_continuation` | Полное совпадение словарей состояния при продолжении | **PASS** |
| 22 | `test_fix2_sidecar_pool_executes_actual_clones` | Честный Sidecar: равенство запрошенных и исполненных клонов | **PASS** |
| 23 | `test_fix3_full_runtime_causal_intervention_real_variables` | Причинная интервенция на 100% реальных переменных | **PASS** |
| 24 | `test_fix4_witness_integrity_and_ledger_validation` | Крипто-валидация свидетелей и логирование вытеснения | **PASS** |
| 25 | `test_epistemic_patch_observed_and_shadow_cannot_authorize_commit` | **Отрицательный тест:** `OBSERVED`/`AUDIT_SHADOW` не разрешают `COMMIT` | **PASS** |
| 26 | `test_epistemic_patch_missing_ledger_fails_closed_abstain` | **Fail-closed:** отсутствие Ledger при структурированном свидетеле $\to$ `ABSTAIN` | **PASS** |
| 27 | `test_epistemic_patch_ledger_integrity_seq_and_prev_hash` | **Целостность Леджера:** валидация `seq == len(records)` и `prev_hash` | **PASS** |
| 28 | `test_epistemic_patch_sidecar_neural_entropy_terminology` | **Честная терминология:** отбор по энтропии нейронного прогноза | **PASS** |

---

## 4. ГРАНИЦЫ И СТРОГАЯ ОСТАНОВКА

- Замороженный код LIFE-002 не модифицировался.
- Эксперименты T2 (пластичность $K$, обучение с подкреплением), T3 и T4 **не запускались**.
- Выполнен отдельный атомарный коммит. Работа по мандату T0 + T1 + T1-FIX-001 + T1-EPISTEMIC-PATCH-001 завершена.
