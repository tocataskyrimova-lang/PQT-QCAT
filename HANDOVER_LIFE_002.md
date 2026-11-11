# ЭСТАФЕТА ИССЛЕДОВАНИЯ: PQT-QCat LIFE-002
**Дата фиксации:** 11 ноября 2026 г., 17:55  
**Статус разработки:** Milestones M0, M1, M1-FIX-001, M2, M3 **ПОЛНОСТЬЮ ЗАВЕРШЕНЫ И ЗАФИКСИРОВАНЫ В GIT**  
**Следующий этап:** Milestone M4 (DREAM & Witness Ledger)  
**Контроль:** Архитектурный протокол `ARCHITECTURE_LIFE_002.md` и директивы **ILORI**

---

## 1. Обзор проекта и принципиальные правила

Проект **LIFE-002** — это строгая научная реализация и независимая верификация непрерывной динамической нейросети на основе канонического математического формализма PQT-QCat.

### Главные инварианты (ОБЯЗАТЕЛЬНО К СОБЛЮДЕНИЮ):
1. **Запечатанные сиды (SEALED SEEDS):**
   Финальные оценочные сиды `seeds = [101, 102, 103, 104, 105]` **СТРОГО ЗАПРЕЩЕНО** использовать для подбора гиперпараметров или отладки. Они открываются **только один раз** на этапе M5 после полной заморозки кода (`git tag life002-prereg-v1`). Для разработки используется калибровочный сид `seed = 42`.
2. **Неизменность исторических файлов:**
   Все 408 файлов из `PQT_QCat_LIFE_001`, `independent_nn_audit_20261111`, `PQT_QCat_Neural_Network` и корня заблокированы по SHA-256 в `prereg/HISTORICAL_LOCK.json`. Проверка:
   ```bash
   python PQT_QCat_LIFE_002/lock_historical.py --verify
   ```
   Должен всегда возвращать: `ALL HISTORICAL FILES VERIFIED 100% UNCHANGED!`.
3. **Замороженный считывающий слой (Frozen Readout):**
   Матрица считывания $W_{\mathrm{readout}}$ инициализируется при создании сети и **навсегда замораживается**. Обучение меняет внутренние структурные параметры ядра $\theta_{\mathrm{core}} = \{K, B\}$.
4. **Абсолютный запрет оракулов и утечек (Anti-Leakage):**
   Вызов `predict(token)` принимает **только** входной токен. Никаких `context_id`, скрытых меток правил или правильных ответов. Ответ среды `target_y` передаётся **исключительно** на этапе задержанного `feedback()` строго после фиксации предсказания.
5. **Отказ при нечисленности (Fail-Fast):**
   Никакого скрытого `nan_to_num`. Любое появление NaN/Inf немедленно вызывает `NonFiniteStateError` и регистрируется как отказ.

---

## 2. Структура директорий и Git

Корневая папка: `C:\AGI\PQT_QCAT_AUDIT` (или путь, куда вы распакуете директорию).

```
C:\AGI\PQT_QCAT_AUDIT\
├── PQT_QCat_LIFE_001/          # Главный Git-репозиторий (ветка pqt-qcat-life-001, чистая, snapshot)
├── PQT_QCat_LIFE_002/          # Активный Git Worktree (ветка pqt-qcat-life-002 — ЗДЕСЬ ИДЁТ РАБОТА)
├── independent_nn_audit_20261111/ # Исторический аудит от 11.11.2026 (заморожен)
├── PQT_QCat_Neural_Network/    # Исторический исходный код v0.1 (заморожен)
├── ARCHITECTURE_LIFE_002.md    # Полное архитектурное ТЗ (461 строка)
├── HANDOVER_LIFE_002.md        # Настоящий документ передачи эстафеты
└── results/                    # Артефакты бенчмарков внутри PQT_QCat_LIFE_002/results/
```

### Git-история в `PQT_QCat_LIFE_002`:
* `88b477d` — **M3: Endogenous contexts q_ctx, Blind Twin-Histories T2, causal interventions, symmetric noA ablation, and gap=1000 physical diagnostics** *(HEAD)*
* `ae796b1` — **M2: Six-echelon 48D architecture, core plasticity, T1 hold retention, and T6 core causality**
* `8c58330` — **M1-FIX-001: Strict IntegratorConvergenceError, fixed/adaptive modes, extended invariants and validation**
* `a07e3d3` — **M0-M1: Complete mathematical engine, canonical A_I, adaptive Heun stepper, and invariant verification**
* `b8a5689` — **Build LIFE-001 continuous plasticity experiment** *(Базовый коммит)*

> [!TIP]
> **При переносе на другой компьютер:**  
> Так как `PQT_QCat_LIFE_002` — это `git worktree`, если путь к папке изменится (например, на `D:\PQT_QCAT_AUDIT`), выполните в консоли:
> ```bash
> cd PQT_QCat_LIFE_001
> git worktree repair
> ```
> Git мгновенно обновит внутренние относительные пути к ворктри.

---

## 3. Что уже сделано и проверено (Milestones M0 — M3)

### Milestone M0: Source Lock & Setup (PASSED)
- Проведена полная инвентаризация и SHA-256 хеширование 408 исторических файлов.
- Оформлен документ расхождений `prereg/PDF_CODE_DISCREPANCIES.md` (дефекты D1–D8, включая математическую ошибку PDF стр. 26).
- Оформлен черновик пререгистрации `prereg/LIFE002_PREREG_001.yaml`.

### Milestone M1 & M1-FIX-001: Math First (PASSED, 17/17 тестов)
- Реализован 8D потенциал $E(Z)$ и аналитический градиент $\nabla E(Z)$ с явным $r_0^2 = 0.8$.
- Реализован канонический антисимметричный оператор $A_I(w_0)$ ($A + A^T = 0$ с машинной точностью $0.0$, дефект ошибочной матрицы PDF равен $3.0$).
- Реализован двухстадийный интегратор Хьюна с адаптивным контролем шага и строгим `IntegratorConvergenceError` при недостижимом допуске, а также явным режимом `fixed`.
- Доказана монотонность диссипации энергии $\frac{dE}{dt} \le 0$ на 50 случайных траекториях.

### Milestone M2: Six-Echelon Core & Core Plasticity (PASSED, 26/26 тестов)
- Построена связанная 48D-система из 6 функциональных эшелонов:
  - **E1 (Fast Sense) & E2 (Slow Sense):** приём и сглаживание сенсорного потока.
  - **E3 (Hold A) & E4 (Hold B):** компартмент памяти с пониженной мобильностью $\Gamma_{\mathrm{hold}} = 10^{-4}$ и изоляцией через `write_gate`.
  - **E5 (Context-Gate) & E6 (Context-Drift):** эндогенный контекст и детектор смены правил.
- Полносвязная матрица межъячеечных связей $K \in \mathbb{R}^{48 \times 48}$ с нормированием токов $\|u_{\mathrm{inter}}\| \le 5.0$.
- Трехфакторный механизм пластичности `CorePlasticityEngine` для параметров $\theta_{\mathrm{core}} = \{K, B\}$ при строго замороженном $W_{\mathrm{readout}}$.
- Сценарий T1 (Hold Latch): удержание стимула на интервалах 0, 20, 50, 100, 300 тиков — **100.0%**.
- Сценарий T6 (Core Causality): доказано каузальное сохранение опыта в ядре (scramble снижает точность до $25\%$, restore возвращает $100\%$, swap между агентами переносит задачу с $100\%$ точностью).

### Milestone M3: Endogenous Contexts & T2 Blind Twin-Histories (PASSED, 30/30 тестов)
- **Топологические бассейны:** Рассчитаны 8 стационарных 8D-минимумов потенциала $V_8(Z)$:
  - 6 гексагональных циклических минимумов ($E = -0.5720$).
  - 2 синхронных состояния ($E = +0.0531$).
- **Эндогенный роутер $q_{\mathrm{ctx}}(Z_{\mathrm{ctx}})$:** Непрерывное вероятностное распределение по бассейнам из внутреннего состояния E5 без оракулов (уверенность $> 90.5\%$).
- **Переключение вихря по невязке:** Рост ошибки предсказания $R_t > 0.40$ переводит $w_0 \to -1.0$, активируя вращательный оператор $A_I(w_0)$ для преодоления энергетического барьера.
- **Сценарий T2 (Слепые близнецы):** Клонированные близнецы Twin 1 и Twin 2 с разной историей обучения протестированы вслепую на новых токенах:
  - Twin 1 на Задаче A: **100.0%** (на Задаче B: **0.0%**).
  - Twin 2 на Задаче B: **100.0%** (на Задаче A: **0.0%**).
  - Гипотеза H2-CORE подтверждена.
- **Каузальные интервенции:**
  - Сброс $Z \to 0$: точность сохраняется на уровне **100.0%** (знание в весах).
  - Swap $\theta_{\mathrm{core}}$: мгновенный обмен навыками (**100.0%** на противоположной задаче).
  - Swap $B$ vs Swap $K$: доказано, что $B$ задаёт перцептивное отображение токенов, а $K$ — фазовую координацию.
- **Симметричный контроль noA ($A=0$):**
  - При внезапной смене правила PQT восстанавливает $\ge 80\%$ точности на шаге 43 (итоговая точность 60%), тогда как noA не достигает порога 80% (итоговая точность 40%). Causal difference: $+20.0\%$.
- **Физический анализ памяти на gap=1000:**
  - Рассчитаны точные барьеры между бассейнами: $\Delta E = 0.6606$ (соседние) до $1.9919$ (противоположные).
  - Измельчение шага $h \in \{0.01, 0.005, 0.002, 0.001\}$ показало численное смещение всего $0.000733$ за 1000 тиков.
  - Устойчивость к шумам: 100% восстановление при $\sigma \le 0.10$. Причина затухания в M2 была в нефизической инициализации ($m=0$ вместо $-1.522$); при канонических бассейнах память стабильна.

---

## 4. Карта ключевых файлов исходного кода

Все новые исходники находятся в `PQT_QCat_LIFE_002/`:

| Файл / Модуль | Назначение |
| :--- | :--- |
| [`lock_historical.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/lock_historical.py) | Проверка SHA-256 замка 408 исторических файлов (`--verify`). |
| [`pqt_qcat/core/energy.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/core/energy.py) | 8D потенциал $E(Z)$, аналитический градиент $\nabla E(Z)$, параметры ячейки. |
| [`pqt_qcat/core/vortex.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/core/vortex.py) | Канонический антисимметричный оператор $A_I(w_0)$, сигмоида $g(w_0) = \sigma(-\beta_v w_0)$. |
| [`pqt_qcat/core/integrator.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/core/integrator.py) | Интегратор Хьюна с bisection, режимы `fixed`/`adaptive`, строгий `IntegratorConvergenceError`. |
| [`pqt_qcat/core/six_echelon.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/core/six_echelon.py) | 48D ядро (E1–E6), межъячеечная связь $K$, `write_gate`, клиппинг токов $\|u\| \le 5.0$, `predict()`. |
| [`pqt_qcat/plasticity/eligibility.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/plasticity/eligibility.py) | Локальные следы пригодности для межъячеечных связей и сенсорной проекции. |
| [`pqt_qcat/plasticity/core_update.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/plasticity/core_update.py) | Трёхфакторное правило обновления $\theta_{\mathrm{core}} = \{K, B\}$ при строго замороженном $W_{\mathrm{readout}}$. |
| [`pqt_qcat/memory/basins.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/memory/basins.py) | 8 канонических минимумов, функция $q_{\mathrm{ctx}}(Z_{\mathrm{ctx}})$, расчёт энергетических барьеров. |
| [`pqt_qcat/memory/twin_agent.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/memory/twin_agent.py) | Автономный агент с разделением `predict(token)` и задержанного `feedback()`, интервенции. |
| [`pqt_qcat/memory/retention_analysis.py`](file:///C:/AGI/PQT_QCAT_AUDIT/PQT_QCat_LIFE_002/pqt_qcat/memory/retention_analysis.py) | Диагностика измельчения шага $h$, энергетических барьеров и релаксации возмущений. |
| **Тесты:** `pqt_qcat/tests/` | `test_math_invariants.py` (17 тестов), `test_no_leakage.py` (2 теста), `test_six_echelon.py` (5 тестов), `test_t1_hold_latch.py` (1 тест), `test_t6_core_causality.py` (1 тест), `test_t2_twin_histories.py` (4 теста). |
| **Бенчмарки:** `results/` | `m1_math_verification.json`, `m2_echelon_verification.json`, `m3_context_verification.json`. |

---

## 5. Инструкция для продолжения: Как запустить тесты и бенчмарки

Откройте PowerShell в `C:\AGI\PQT_QCAT_AUDIT`:

```powershell
# 1. Проверить целостность исторических файлов
python PQT_QCat_LIFE_002\lock_historical.py --verify

# 2. Запустить все 30 тестов pytest
python -m pytest PQT_QCat_LIFE_002\pqt_qcat\tests -v

# 3. Перезапустить бенчмарки этапов
python PQT_QCat_LIFE_002\run_m1_benchmarks.py
python PQT_QCat_LIFE_002\run_m2_benchmarks.py
python PQT_QCat_LIFE_002\run_m3_benchmarks.py
```

Все 30 тестов должны завершиться со статусом `30 passed`.

---

## 6. План следующего этапа — Milestone M4 (DREAM & Witness)

После согласования от ILORI предстоит реализовать этап M4 согласно разделу 5 и 6 `ARCHITECTURE_LIFE_002.md`:

1. **Реестр происхождения памяти (`pqt_qcat/memory/witness.py`):**
   * Разделение на два изолированных регистра:
     $$M_t = (B^{\mathrm{real}}_t, B^{\mathrm{dream}}_t, \theta^{\mathrm{core}}_t, Z_t), \quad B^{\mathrm{real}} \cap B^{\mathrm{dream}} = \varnothing$$
   * $B^{\mathrm{real}}$ хранит только реально наблюдавшиеся события с подтверждённым feedback.
   * $B^{\mathrm{dream}}$ хранит сгенерированные гипотезы сна и их внутреннюю уверенность.
   * Запрет автоматического превращения гипотезы сна в реальный факт без внешнего подтверждения (защита от галлюцинаций).
2. **Генератор сна и консолидация на идентичном коде (`pqt_qcat/dream/`):**
   * Модули `generator.py` и `consolidation.py`.
   * **Симметричный дизайн:** Генератор, гейты отбора и число шагов **абсолютно идентичны** для PQT и noA. Единственное различие — матрица вихря ($A = A_I$ против $A = 0$).
3. **Сценарий T4 (Dream Safety):**
   * Проверка сохранения ранее выученных якорных правил после 1000 тиков сна/idle.
   * Деградация точности на реальных якорях не должна превышать 2 процентных пунктов ($\le 2\%$).
4. **Сценарий T5 (Counterfactual Synthesis):**
   * В среде с частично наблюдаемым графом переходов оценить способность модели синтезировать ненаблюдавшиеся композиции без оракула.
   * Оценка частоты ложноположительных гипотез.

---

## 7. Финальный Milestone M5 (Pre-registration Run)

После завершения M4:
1. Заморозка кодовой базы тегом: `git tag life002-prereg-v1`.
2. Фиксация контрольных сумм всех модулей.
3. Однократный запуск на закрытых `seeds = [101, 102, 103, 104, 105]`.
4. Публикация сырых логов и независимая оценка выполнения гипотез H0–H6.

**Эстафета готова к передаче. Кодовая база, тесты и артефакты полностью сохранены на диске.**
