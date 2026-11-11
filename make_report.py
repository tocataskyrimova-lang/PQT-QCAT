from pathlib import Path
import csv,json,hashlib,platform,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
load=lambda name:json.loads((ROOT/name).read_text(encoding='utf8'))
def mean_query(row,key):return float(np.mean([v['accuracy'] for v in row[key].values()]))
def stat(values):return f'{100*np.mean(values):.2f} ± {100*np.std(values,ddof=1):.2f}'
def table(header,rows):return '| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'

def main():
    data=load('main_results.json');long=load('long_results.json');temporal=load('temporal_results.json');runtime=load('runtime_validation.json')
    diagnosis=load('replay_diagnosis.json');guarded=load('guarded_results.json');demo=load('demo_result.json')
    state_only=load('state_only_results.json')
    assert len(data)==50 and len(long)==9 and len(temporal)==12 and len(diagnosis)==3 and len(guarded)==3
    assert all(r['status']['finite'] and r['all_feedback_follows_prediction'] for r in data)
    assert all(r['status']['finite'] for r in long+temporal)
    kinds=['PQT','noA','leaky','frozen','no_replay','no_consolidation','reset_core','shared_context','fixed_encoder','no_replay_no_consolidation']
    full=[r for r in data if r['variant']=='PQT'];summary=[];csvrows=[]
    for k in kinds:
        rr=[r for r in data if r['variant']==k]
        metrics={'late': [np.mean([s['last100'] for s in r['stages'][:4]]) for r in rr],
                 'before':[mean_query(r,'before_idle') for r in rr],'after':[mean_query(r,'after_idle') for r in rr],
                 'paired_delta':[mean_query(r,'after_idle')-mean_query(r,'paired_noidle') for r in rr],
                 'generation':[r['generation']['exact_match'] for r in rr],
                 'retention':[mean_query(r,'old_after_new_context') for r in rr],
                 'new':[mean_query(r,'new_rule') for r in rr]}
        summary.append([k,stat(metrics['late']),stat(metrics['after']),f"{100*np.mean(metrics['paired_delta']):+.2f}",stat(metrics['generation']),stat(metrics['retention'])])
        for r in rr:csvrows.append({'variant':k,'seed':r['seed'],'late':np.mean([s['last100'] for s in r['stages'][:4]]),'before_idle':mean_query(r,'before_idle'),'after_idle':mean_query(r,'after_idle'),'paired_delta':mean_query(r,'after_idle')-mean_query(r,'paired_noidle'),'generation_exact':r['generation']['exact_match'],'old_retained':mean_query(r,'old_after_new_context'),'new_rule':mean_query(r,'new_rule'),'encoder_delta':r['encoder_delta_norm'],'ticks':r['status']['ticks']})
    summary_table=table(['Вариант','Поздние ответы, %','После idle, %','Δ idle, п.п.','Exact8, %','Старое после нового, %'],summary)
    (ROOT/'figures').mkdir(exist_ok=True)
    with (ROOT/'summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0].keys());w.writeheader();w.writerows(csvrows)
    temp_rows=[]
    for k in ('PQT','noA','leaky','reset_core'):
        rr=[r for r in temporal if r['variant']==k]
        temp_rows.append([k,*[stat([r['tests'][str(d)]['accuracy'] for r in rr]) for d in (0,20,50,100,300)]])
    temp_table=table(['Вариант','Gap0','Gap20','Gap50','Gap100','Gap300'],temp_rows)
    long_rows=[]
    for k in ('PQT','noA','fixed_encoder'):
        rr=[r for r in long if r['variant']==k]
        long_rows.append([k,stat([np.mean([s['last100'] for s in r['stages'][:10]]) for r in rr]),stat([np.mean([s['last100'] for s in r['stages'][-10:]]) for r in rr]),f"{max(r['status']['max_abs_state'] for r in rr):.3f}"])
    long_table=table(['Вариант','Этапы1–10, %','Этапы51–60, %','max abs(z) в финале'],long_rows)
    diag_table=table(['Политика idle без консолидации','Средняя точность, %'],[[mode,stat([np.mean([v['accuracy'] for v in r['policies'][mode]['queries'].values()]) for r in diagnosis])] for mode in diagnosis[0]['policies']])
    guard_table=table(['Политика, новые seeds701–703','После idle, %'],[[mode,stat([np.mean([v['accuracy'] for v in r['policies'][mode]['queries'].values()]) for r in guarded])] for mode in guarded[0]['policies']])
    guard_full=[r['full_guarded_life'] for r in guarded]
    state_only_full=[r['full_state_only_life'] for r in state_only]
    state_only_table=table(['Политика, новые seeds801–803','После idle, %'],[[mode,stat([np.mean([v['accuracy'] for v in r['policies'][mode]['queries'].values()]) for r in state_only])] for mode in state_only[0]['policies']])
    guard_rejected=[r['policies']['guarded']['status']['replay_rejections'] for r in guarded]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    kk=['PQT','noA','leaky','no_replay','no_consolidation','shared_context']
    for i,k in enumerate(kk):
        rr=[r for r in data if r['variant']==k]
        axes[0].bar(i-.17,100*np.mean([mean_query(r,'paired_noidle') for r in rr]),.32,color='#8aa9b8',label='No-idle branch' if i==0 else None)
        axes[0].bar(i+.17,100*np.mean([mean_query(r,'after_idle') for r in rr]),.32,color='#237a77',label='After idle' if i==0 else None)
    axes[0].set_xticks(range(len(kk)),kk,rotation=35,ha='right');axes[0].set(ylabel='Recall accuracy (%)',ylim=(0,103));axes[0].legend(fontsize=8)
    for k,c in [('PQT','#b33739'),('noA','#237a77'),('leaky','#2461a7'),('reset_core','#777777')]:
        rr=[r for r in temporal if r['variant']==k];ds=[0,10,20,50,100,300];a=np.array([[r['tests'][str(d)]['accuracy'] for d in ds] for r in rr])*100
        axes[1].plot(ds,a.mean(0),'o-',label=k,color=c);axes[1].fill_between(ds,a.min(0),a.max(0),alpha=.1,color=c)
    axes[1].axhline(100/7,ls=':',color='gray');axes[1].set(xlabel='Cue-to-query idle ticks',ylabel='Temporal cue recall (%)',ylim=(0,104));axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/memory.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.8))
    for k,c in [('PQT','#b33739'),('noA','#237a77'),('fixed_encoder','#2461a7')]:
        arr=np.array([[s['last100'] for s in r['stages']] for r in long if r['variant']==k])*100
        ax.plot(np.arange(1,61),arr.mean(0),label=k,color=c,alpha=.8);ax.fill_between(np.arange(1,61),arr.min(0),arr.max(0),color=c,alpha=.1)
    ax.set(xlabel='Rule change',ylabel='Last 100 prequential answers (%)',ylim=(80,101));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/plasticity.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.8))
    modes=['no_idle','naive','guarded','no_replay']
    for i,r in enumerate(guarded):ax.plot(range(4),[100*np.mean([v['accuracy'] for v in r['policies'][m]['queries'].values()]) for m in modes],'o-',label=f"seed {r['seed']}")
    ax.set_xticks(range(4),modes);ax.set(ylabel='Recall accuracy (%)',title='Replay follow-up: same learned snapshot, different idle policies',ylim=(0,103));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/guarded_replay.png',dpi=170);plt.close(fig)
    adapt=[s['events_to_rolling100_90pct'] for r in full for s in r['stages'][:4]]
    text=f'''# PQT–QCat LIFE-001 — непрерывная пластичность и внутренняя активность

**Итог: PARTIAL.** Дата: 11 ноября2026. Функциональный непрерывный прототип
работает; специфическое преимущество PQT и полноценная глубокая структурная
пластичность не доказаны. Это новая отдельная ветка `pqt-qcat-life-001`.
v0.1 сохранён неизменным; его завершённый аудит имеет вердикт PARTIAL.

## 1. Что действительно построено

Один объект Life сохраняет физическое состояние, обучаемые связи, быстрые и
медленные веса, опыт, pending eligibility, clock и RNG между обращениями.
predict выдаёт символ **до** feedback; feedback приходит спустя три события
и меняет связи без BPTT, offline optimizer или нового fit. Во время паузы
ядро продолжает ticks, консолидацию и повторное использование прошлого опыта.
Символы можно генерировать авторегрессивно, подавая предыдущий ответ как
следующий вход. Snapshot/resume сохраняет всю жизнь, включая отложенный feedback.

Это восемь символов и явные контексты, не языковое общение. Связь encoder
действительно меняется, но основной измеренный вклад в приобретённый опыт
вносит контекстный readout с быстрыми/медленными весами. Тест fixed_encoder
показывает, что малый текущий encoder-update не нужен для достигнутого качества.
Поэтому мы ещё не получили доказанную самостоятельную структурно-пластичную
PQT-систему, отличную по полезным свойствам от более простых online-механизмов.

## 2. Устройство и научные границы

Шесть восьмимерных PQT-ячеек,48 динамических координат, Heun с адаптивным
дроблением. r0_squared=.8; внешний tick=.01, одно событие20 ticks.
Readout получает97 признаков состояния и его реакции на вход; прямой
символ или правильный ответ в readout не передаются. На пять контекстов
создаются7760 fast/slow коэффициентов плюс234 обучаемых коэффициента encoder.
Физические W — координаты динамики; encoder/F/S — отдельные реальные связи.

Правило feedback: F += eta*(y−p)*eligibilityᵀ. Input update использует
локальный trace и фиксированный feedback projection; это **эвристика**,
не точный e-prop и не градиент всей рекуррентной системы. Консолидация
переносит часть F в S, мгновенно сохраняя F+S; F постепенно затухает.
Replay buffer ограничен64 событиями на контекст, pending traces —256,
число выделяемых контекстов —32. Полная спецификация — ARCHITECTURE.md.

Внутренняя деятельность подпитывается явным replay forcing. Это не
непрерывный вечный цикл замкнутой диссипативной T=0 системы и не следствие
сохранения Gibbs-меры. Соответствующие различия установлены аудитом v0.1.
«Одновременно» здесь означает общую последовательность событий с обучением
между ответами, а не гонку параллельных потоков за общей матрицей.

Eligibility-подход мотивирован работой
[Bellec et al.,2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/);
использованный локальный heuristic не воспроизводит её spiking-архитектуру.
Тест многих смен правил мотивирован
[Dohare et al.,2024](https://www.nature.com/articles/s41586-024-07711-7),
исследующей потерю способности учиться в других глубоких continual-системах.
DMN/replay используется только как функциональная аналогия с
[гипотезой cascaded memory systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC12998549/).
Ни одна из этих работ не доказывает QCat или сознание программы.

## 3. Протокол и причинный порядок

Разработка: seeds11/12, learning_rate .3/.8/1.5 на11; выбран.8, подтверждён
на12. Затем отдельные основные seeds101–105, десять вариантов,50 жизней.
Каждая получает четыре неизвестные перестановки алфавита в явных контекстах
A/B/C/D по500 событий, паузу10000 ticks, новые вопросы, свободную генерацию,
новую версию A_v2 и отдельный конфликт новой версии под прежним именем A.
Полные правила известны только среде; agent получает labels через feedback.
Контексты задаются снаружи: автоматическое открытие контекстов не показано.

Оценка prequential: сначала предсказание, затем ошибка по будущему feedback.
Контрольные queries не получают target, но физические ticks и консолидация
не выключаются. Нет скрытого model.eval()/reset между этапами главной жизни.
Копии snapshot используются только в явно обозначенных парных диагностических
ветвях. Контроль reset_core специально нарушает сохранение физического состояния.

Каждая основная жизнь включает94320 ticks и4216 предсказаний; feedback3000.
Это ускоренное модельное время, не утверждение о работе всю ночь в реальном
времени. Отдельно реально запущенный JSONL-сервер подтвердил продвижение
clock в отсутствие сообщений, пока stdin открыт.

## 4. Главный опыт: обучение и сохранение опыта

± означает SD между пятью seed. Late — последние100 предсказаний каждого
из первых четырёх правил; post-idle — среднее по четырём контекстам,64 query
на каждый. Δ idle сравнивает одинаковые query после паузы и в парной ветви
того же snapshot без паузы. Exact8 — полное совпадение восьми самостоятельно
сгенерированных символов,32 последовательности на seed.

{summary_table}
Полная PQT достигает среднего late={stat([np.mean([s['last100'] for s in r['stages'][:4]]) for r in full])}%,
после паузы —{stat([mean_query(r,'after_idle') for r in full])}%,
свободная генерация —{stat([r['generation']['exact_match'] for r in full])}%.
Новый A_v2 усваивается с{stat([mean_query(r,'new_rule') for r in full])}%
на query; старые контексты остаются доступны. Медиана достижения90%
в скользящем окне100 событий на первых правилах —{np.median(adapt):.0f} событий,
диапазон{min(adapt)}–{max(adapt)}. Это обучаемость по многим примерам,
не one-shot понимание словесного правила.

Во время10000 idle ticks во всех пяти полных жизнях: ноль новых внешних
событий и ноль новых teacher labels,500 replay updates,250 консолидаций.
Физические W при этом меняются: нормы смещения
{', '.join(f"{r['idle_metrics']['physical_W_motion']:.3f}" for r in full)}.
Encoder за жизнь изменяется на норму около{np.mean([r['encoder_delta_norm'] for r in full]):.4g}
при исходной норме порядка8.5 — небольшой, измеримый эффект, функциональная
необходимость которого не показана. Context gates критичнее: shared_context
учит текущее правило, но теряет большую часть старых несовместимых правил.

![Память и зависимость от прошлого](figures/memory.png)

Первая задача всё ещё допускает успешное решение статическим отображением
внутри каждого контекста. Поэтому результат reset_core в ней не считается
опровержением непрерывного runtime, но показывает, что этой задачи недостаточно
для доказательства использования физической рабочей памяти.

## 5. Проверка памяти между обращениями

Новый temporal assay: cue из1..7, пауза, одинаковый для всех примеров query0.
Ответ зависит от раннего cue; текущий query и context сами по себе ответа
не содержат. Три новых seed301–303,1200 online trials на случайных gaps0..20;
затем280 новых примеров на каждую задержку в парных продолжениях snapshot.
Во время проверки нет новых teacher labels. Таблица — проценты, SD по3 seed.

{temp_table}
PQT и noA сохраняют рабочую информацию между вызовами и восстанавливают
cue на обученных промежутках. Reset_core возвращает результат к уровню
примерно1/7. Простая leaky-замена с теми же интерфейсами и фиксированной
конфигурацией хуже на этом тесте; это не исчерпывающий подбор всех простых
RNN и не доказательство уникальности PQT. Уже noA показывает, что A не
обязателен. На gaps50–300 качество резко снижается: долговременное хранение
правила в связях и удержание текущего cue в динамике — разные свойства.
Долгая рабочая память пока не получена.

## 6. 60 смен правил без перезапуска

Отдельные seeds201–203, три варианта,60 меняющихся перестановок; три
повторно используемых контекста, по300 событий и100 idle ticks между
изменениями. В каждой жизни366000 ticks,18000 предсказаний и feedback.
В таблице оцениваются последние100 ответов каждого этапа, затем среднее
по первым/последним десяти этапам. Здесь требуется адаптация к новому
значению старого ID, а не одновременно сохранить все противоречивые версии.

{long_table}
![Пластичность при смене правил](figures/plasticity.png)

Катастрофической потери способности учиться за60 смен в этих опытах нет;
у PQT есть небольшое снижение качества в конце. Нельзя переносить это
на бесконечный поток, сложный язык или глубокую многослойную plasticity.
Практически совпадающий fixed_encoder подтверждает, что основной результат
обеспечен readout-пластичностью, а не доказанным обучением нового PQT-представления.

## 7. Контрпример: фоновая деятельность может вредить

В no_consolidation наивный replay дал среднее изменение качества около−25 п.п.
Отключение replay при той же отключённой консолидации гораздо безопаснее.
Это отрицательный результат, не удалённый из выборки. Само затухание всех
fast weights может долго сохранять argmax, одновременно снижая уверенность;
поэтому также сохранены cross entropy и confidence, а не только accuracy.

Дополнительная причинная диагностика на новых seeds601–603: одно online
обучение без консолидации, далее одинаковые query из ветвей одного snapshot.

{diag_table}
Вред остаётся при выключенном внутреннем forcing и при замороженном на
время idle core. В этих проверках источник — фоновое изменение readout,
а не одно лишь движение физического состояния. Интерпретация: replay по
сохранённым старым признакам может ухудшать ответы в текущем представлении.
Это согласованная с интервенциями гипотеза; все возможные причины не исчерпаны.

## 8. Защитный replay: отдельная проверка

После обнаружения дефекта добавлен **опциональный** guarded_replay, default
основного исследования остался False. Он периодически заново кодирует ранее
наблюдавшиеся context/token в shadow core и использует только уже полученные
labels. Proposed update отклоняется, если растёт BCE на этих anchors или
уменьшается число правильных anchors. Active physical state не откатывается.
Это защита на ограниченной прошлой выборке, не гарантия всех будущих ответов.

Тест заранее описан в guarded_replay_protocol.md; новые seeds701–703 не
использовались в разработке. Ни один final query target не доступен guard.
Сначала одинаковые snapshots без консолидации:

{guard_table}
![Защитный replay](figures/guarded_replay.png)

Из500 фоновых предложений guard отклонил
{', '.join(map(str,guard_rejected))} соответственно. Его дополнительные
shadow evaluations учтены в status.guard_anchor_evaluations; это реальная
вычислительная цена, не бесплатное улучшение.

В трёх отдельных полных жизнях с guard и обычной консолидацией post-idle
accuracy={stat([mean_query(r,'after_idle') for r in guard_full])}%,
Exact8={stat([r['generation']['exact_match'] for r in guard_full])}%.
Эти три seed не объединяются с первичными пятью для увеличения эффектной
цифры. Режим остаётся экспериментальным: ограниченный anchor-набор не
может гарантировать отсутствие забывания вне своего покрытия.

В стресс-ветвях без консолидации guard уменьшил вред, но среднее ухудшение
относительно no-idle осталось около5.73 п.п., то есть **не прошло** пилотный
допуск5 п.п. Это не скрыто за улучшением относительно naive replay.

Поэтому отдельно проверен консервативный **state-only replay**: выбирается
прошлый опыт, продолжается внутреннее forcing и движение z, но replay не
переписывает readout. Обычный внешний feedback продолжает online-обучение,
консолидация сохраняется. Новые seeds801–803, прежние контрпримеры не удалены:

{state_only_table}
Во всех трёх парных стресс-ветвях state-only сохранил точность no-idle,
при500 внутренних replay events и нуле replay weight updates. Физическая
динамика при этом не была остановлена. В трёх новых полных жизнях с
консолидацией post-idle={stat([mean_query(r,'after_idle') for r in state_only_full])}%,
Exact8={stat([r['generation']['exact_match'] for r in state_only_full])}%.
Это наблюдаемая сохранность в заданном опыте, не универсальная гарантия:
движение состояния всё ещё может нарушать рабочую память на длинных gaps.
Для интерактивного запуска предлагается явный флаг --state-only-replay;
основной исследовательский default оставлен неизменным для воспроизведения.

## 9. Численная и программная проверка

19 автоматических проверок: независимая конечная разность энергии,
малая закрытая диссипация, опасное состояние из аудита v0.1, persistent state,
причинный predict→feedback, изменение настоящих связей, bounded memory,
неповторяемый feedback, консолидация F+S, замороженный контроль, snapshot
с pending traces, guard rollback, guarded exact resume, autoregressive chain,
отказ при NaN logits и корректность входов/learning clock.

При1000 одинаковых driven ticks RMS к reference8substeps:
{', '.join(f"k{k}: {v['rms_to_k8']:.3g}" for k,v in runtime['refinement'].items())}.
Это наблюдаемая сходимость на фиксированном опыте, не глобальная гарантия.
Стресс100000 idle ticks завершился без nonfinite. Во всех завершённых
primary/long/temporal жизнях шаги прошли finite checks. Если tolerance
дробления не достигается или logits нечисленны, система завершает шаг
ошибкой, не выдаёт фиктивный класс.

Реальный локальный сервер: фоновый clock продвинулся при открытом stdin
без входных сообщений; feedback после паузы принят, checkpoint записан,
процесс штатно завершён и восстановлен. Портов и сетевого сервиса нет.
Сервер не оставлен бесконечно расходовать ресурсы после проверки; запуск
для пользователя указан в README.md.

## 10. Короткий демонстратор

quick_demo.py запускает одну жизнь, учит переходы forward и reverse,
делает10000 пустых ticks без новых labels, затем генерирует из символа0:

```text
forward: {demo['generated']['forward']}
reverse: {demo['generated']['reverse']}
```

Expected соответственно1,2,3,4,5,6,7,0 и7,6,5,4,3,2,1,0.
Это наглядная demonstration seed901, не часть статистики primary.
Её checkpoint лежит отдельно в runtime/demo_life.json.gz.

## 11. Вердикт и следующий научный вопрос

| Свойство | Итог |
|---|---|
| Одно сохраняющееся состояние и точный restart | PASS в выполненных тестах |
| Ответ до feedback, online-изменение связей, без offline fit | PASS |
| Усвоение новой символической функции и сохранение старой по явному context | PASS в основном опыте |
| Внутренняя активность без внешнего сообщения | PASS; источник — явный replay forcing |
| Безопасность любого фонового обучения | FAIL как универсальное утверждение; найден контрпример |
| Ограниченная защита replay на anchors | Проверена отдельно; не глобальная гарантия |
| Рабочая память между вызовами | PASS для коротких gaps0..20; FAIL для длинных gaps100..300 |
| Сохранение пластичности на60 изменениях | PASS в ограниченном символическом стрессе |
| Необходимость A или преимущество полной PQT | Не подтверждены; noA часто лучше |
| Функционально необходимая глубокая/структурная plasticity ядра | Не установлена |
| Самостоятельное открытие контекстов, язык, сознание | Не исследованы/не заявляются |

LIFE-001 дал проверяемый переход от «обучили и выключили» к непрерывному
online runtime. Но следующий шаг не должен скрывать, что сохранение правил
пока в основном реализуют явные context gates и обычные fast/slow связи.
Научно содержательная LIFE-002: вывести eligibility через PQT-динамику,
показать полезное изменение представления; отделить защищённую рабочую
память от фонового replay; проверить новые ненаблюдаемые заранее контексты,
длинные задержки и другие генераторы. Сравнение с A=0 остаётся обязательным.

Артефакты: protocols, весь код, основной журнал predict/feedback каждого
seed, JSON/CSV, snapshots, тесты и графики. Отдельные ограничения:
небольшое число seed; синтетический конечный алфавит; explicit contexts;
неравные вычислительные расходы guarded replay; local input-update heuristic;
ограниченная память и потеря очень старого pending feedback. Отрицательные
результаты не заменены удачным demo.

Все символы основного теста уже встречались при обучении. Обобщение на
новые алфавиты, неизвестные композиции правил и естественный язык не
показано. Raw journals/snapshots хранятся рядом с кодом и включены в архив
поставки; в локальный Git включены код, отчёты и сводные данные.
'''
    import re
    text=re.sub(r'(?<=[а-яА-ЯёЁ])(?=\d)|(?<=\d)(?=[а-яА-ЯёЁ])',' ',text)
    (ROOT/'LIFE_001_REPORT_RU.md').write_text(text,encoding='utf8')
    save_env={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'main_lives':len(data),'long_lives':len(long),'temporal_lives':len(temporal),'guarded_lives':len(guard_full),'state_only_lives':len(state_only_full)}
    (ROOT/'environment.json').write_text(json.dumps(save_env,indent=2),encoding='utf8')
    print('LIFE REPORT READY')

if __name__=='__main__':main()
