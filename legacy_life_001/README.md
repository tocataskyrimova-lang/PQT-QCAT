# PQT–QCat LIFE-001

Исполняемый исследовательский прототип непрерывной пластичности. Он
сохраняет динамическое состояние, выдаёт символы до получения feedback,
меняет связи online, переносит быстрые изменения в медленную память и
продолжает ticks/replay при отсутствии внешних сообщений.

Это маленькая символическая система, не языковая модель, не доказательство
сознания и не доказательство уникальности PQT. Результаты и ограничения:
`LIFE_001_REPORT_RU.md`. Математика и границы: `ARCHITECTURE.md`.

## Запуск

Проверенная среда: Python3.14.4, NumPy2.4.4, CPU. Torch, GPU и сеть не нужны.
Для воспроизведения графиков требуется matplotlib. Зависимости — requirements.txt.

```powershell
Set-Location C:\AGI\PQT_QCAT_AUDIT\PQT_QCat_LIFE_001
python -X utf8 -B test_life.py
python -X utf8 -B quick_demo.py
python -X utf8 -B life.py --serve --checkpoint runtime/my_life.json.gz --idle-hz 20 --state-only-replay
```

Сервер печатает JSON. Пока stdin открыт и сообщений нет, он продолжает
внутренние такты. Закрытие stdin/quit завершает процесс; checkpoint сохраняется.
Перезапуск с тем же runtime checkpoint продолжает состояние и RNG.
`idle-hz` задаёт целевую частоту в свободном времени, не real-time гарантию.
Работа в одном owner-thread исключает гонки; читатель stdin — отдельный поток.

`--state-only-replay` сохраняет внутреннее воспроизведение и консолидацию,
но отключает фоновые поправки readout по старым признакам. Внешний feedback
по-прежнему обучает online. Этот консервативный режим проверен отдельно.
`--guarded-replay` — альтернативный экспериментальный фильтр фоновых
обновлений; он уменьшил вред, но не устранил его полностью. Без этих флагов
воспроизводится исходный исследовательский режим, включая обнаруженные риски.

Пример сообщений, каждое на отдельной строке:

```json
{"op":"predict","context":"new_rule","token":2}
{"op":"feedback","event_id":1,"target":5}
{"op":"generate","context":"new_rule","token":2,"length":8}
{"op":"idle","ticks":1000}
{"op":"status"}
{"op":"save","path":"runtime/my_life.json.gz"}
{"op":"quit"}
```

event_id нужно брать из фактического ответа, особенно после загрузки жизни.
Новая сеть ещё не знает правило: обучение — последовательность примеров с
feedback. Слова и фразы на естественном языке этот интерфейс не интерпретирует.
Контекст задаётся явно. Новая версия правила под новым context ID сохраняет
старую; противоречивое правило под тем же ID постепенно вытесняет прежнее.

Для продолжения проверенного экземпляра сначала скопируйте checkpoint:

```powershell
New-Item -ItemType Directory runtime -Force
Copy-Item results/PQT/101/final_life.json.gz runtime/from_experiment.json.gz
python -X utf8 -B life.py --serve --checkpoint runtime/from_experiment.json.gz --state-only-replay
```

Не передавайте файлы из results как рабочий checkpoint: сервер сохраняет
состояние при выходе, а первичные экспериментальные артефакты следует сохранять.

## Воспроизведение опытов

```powershell
python -X utf8 -B experiment.py --mode main --workers 4
python -X utf8 -B experiment.py --mode long --workers 3
python -X utf8 -B temporal.py
python -X utf8 -B validate_runtime.py
python -X utf8 -B diagnose_replay.py
python -X utf8 -B guarded_experiment.py
python -X utf8 -B state_only_experiment.py
python -X utf8 -B make_report.py
```

Все конечные результаты, включая неудачи, сохраняются. Повторный запуск
перезаписывает собственные результаты в этом каталоге. Для отдельной репликации
скопируйте проект. Seed разработки11/12 не входят в final101–105;
long201–203, temporal301–303, posthoc replay601–603 разнесены.

## Карта файлов

- life.py: persistent core, online feedback, fast/slow weights, bounded replay,
  snapshot/resume, JSONL server.
- experiment.py: 50 жизней десяти вариантов и9 длинных жизней.
- temporal.py: причинная проверка памяти о предыдущем cue при одинаковом query.
- diagnose_replay.py: дополнительная проверка механизма вредного replay.
- validate_runtime.py: уточнение интегратора,100000 idle ticks, реальный сервер.
- test_life.py: математические, причинные и lifecycle-проверки.
- PROTOCOL.md, FINAL_PROTOCOL.md: постановка до соответствующих опытов.
- main_results.json, long_results.json, temporal_results.json: все сводные данные.
- results/: полный журнал predict→feedback и snapshot каждой основной жизни.
- figures/: графики; tests.log и *.log: журналы запусков.
- runtime/: рабочие пользовательские состояния, отдельно от evidence.

Исходный v0.1 находится в соседнем каталоге и не изменён. Независимый аудит
v0.1 находится в `../independent_nn_audit_20261111/`.
