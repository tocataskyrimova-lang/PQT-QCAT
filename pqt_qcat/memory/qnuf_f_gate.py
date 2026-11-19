"""One-write F0 feasibility with an isolated historical witness bank."""
from . import qnuf_context_d4 as d4  # pin numerical threads before NumPy

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from .access_state import exact
from .checkpoint_io import load_state, save_state
from .codex_memory_location import atomic_json, digest, source_hashes
from .functional_sidecar import ActionJournal, read_json, state_hash
from .functional_sidecar_b2a import forecast
from .qnuf import array_hash


BASE = '774cca9ca2f4a16f83c09e0f875786ad3f43c133'
BASE_EXEC = '8ccd4ee28f5249c4b5217420e1604737f1482010'
ARCHIVE = 'c16ce209173bc49167a4559cf8f72ae185f21ab3af99803087cc1d02e1d2cce9'
NAMES = ('RAW','RAW_SMALL','MIX25','MIX50','MIX75','CTX')
WEIGHTS = {'MIX25':.25,'MIX50':.50,'MIX75':.75}
REUSED = {'RAW':'RAW@raw','RAW_SMALL':'RAW@ctx','CTX':'CTX@ctx'}
DEFAULTS = {'bank_size':3,'max_A_nll_increase':.01,'min_B_nll_gain':.10,'clip':.50,
            'selection_order':list(NAMES),'mix_weights':WEIGHTS,'workers':6}
METHOD = 'LIFE_QNUF_F_GATE_F0_METHOD.md'


def init(root,reference,archive):
    repo = d4.parent.c0.causal.b2.b1.bridge.ROOT
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=repo,check=True)
    subprocess.run(['git','diff','--quiet',BASE_EXEC,BASE,'--','pqt_qcat'],cwd=repo,check=True)
    delivery, source = (read_json(reference / name) for name in ('DELIVERY.json','SOURCE.json'))
    exact([delivery['delivery_commit'],delivery['executable_sha'],source['git']],[BASE,BASE_EXEC,BASE_EXEC])
    exact(delivery['verdicts']['direction'],'UNDECIDED')
    assert delivery['counts']['RAW@ctx']['OK']==54
    assert d4.parent.c0.causal.file_sha(archive)==ARCHIVE
    manifest = reference/'ARTIFACT_MANIFEST.sha256'
    entries = {name:sha for sha,name in (line.split('  ',1) for line in manifest.read_text(encoding='utf-8').splitlines())}
    for name,value in entries.items():
        assert d4.parent.c0.causal.file_sha(reference/name)==value,name
    with ZipFile(archive) as packed:
        prior = packed.read('FROZEN_CODE.zip')
    assert prior==subprocess.check_output(['git','archive','--format=zip',BASE_EXEC],cwd=repo)
    assert prior==(reference/'FROZEN_CODE.zip').read_bytes()
    d4_plan = read_json(reference/'PLAN.json')
    exact(d4_plan['units'],[[s,m] for s in d4.parent.c0.SEEDS for m in range(3)])
    plan = {'base':BASE,'base_executable':BASE_EXEC,'base_archive':str(archive),'base_archive_sha256':ARCHIVE,
            'base_source_sha256':hashlib.sha256(prior).hexdigest(),'base_manifest_sha256':d4.parent.c0.causal.file_sha(manifest),
            'base_inputs':entries,'reference':str(reference),'d4':d4_plan,'units':d4_plan['units'],
            'preflight_unit':d4_plan['units'][0],**DEFAULTS,
            'bank_rule':'first three distinct A tokens in confirmed A-only stream, one oldest receipt each',
            'scope':'F0 only; one B witness from each original D4 snapshot and token; no persistent write or F1',
            'bootstrap_seed':d4_plan['bootstrap_seed'],'bootstrap_replicates':d4_plan['bootstrap_replicates']}
    root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(repo/'protocol'/METHOD,root/METHOD)
    for name in ('DELIVERY.json','SOURCE.json','ARTIFACT_MANIFEST.sha256'):
        shutil.copy2(reference/name,root/('BASE_'+name))
    atomic_json(root/'PLAN.json',plan)
    atomic_json(root/'SOURCE.json',{'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                'files':source_hashes(),'plan_sha256':digest(plan),'python':sys.version,
                'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','torch','scipy','pytest','scikit-learn')}})
    atomic_json(root/'LOCK.json',{name:d4.parent.c0.causal.file_sha(root/name) for name in ('PLAN.json','SOURCE.json',METHOD)})


def verify_plan(root):
    locked = read_json(root/'LOCK.json')
    assert all(d4.parent.c0.causal.file_sha(root/name)==value for name,value in locked.items())
    plan,source = (read_json(root/name) for name in ('PLAN.json','SOURCE.json'))
    assert digest(plan)==source['plan_sha256'] and source_hashes()==source['files']
    return plan


def input_path(plan,name):
    path = Path(plan['reference'])/name
    assert d4.parent.c0.causal.file_sha(path)==plan['base_inputs'][name],name
    return path


def historical_bank(history_zip,seed,mapping,anchor,size):
    """Only selected, receipted historical witnesses cross this boundary."""
    prefix = f'single/A/lives/{seed}_{mapping}/separate/SS/'
    with ZipFile(history_zip) as store:
        rows = json.loads(store.read(prefix+'events.json'))
        actions = store.read(prefix+'pre_feedback.jsonl')
        timing = json.loads(store.read(prefix+'runtime.json'))
    ledger = anchor['qnuf']['witness_ledger']
    assert len(rows)==len(ledger)==len(timing)==48
    bank = {}
    for event,row in enumerate(rows):
        observed = row['packet']['observation']
        token = observed['query']
        if token in bank:
            continue
        assert observed=={'context':'A','query':token,'cursor':event}
        receipt = timing[event]['receipt']
        assert timing[event]['issued_ns']<=timing[event]['revealed_ns']<=timing[event]['updated_ns']
        start = receipt['offset']
        chunk = actions[start:start+receipt['length']]
        assert hashlib.sha256(chunk).hexdigest()==receipt['sha256']
        exact(json.loads(chunk),row['packet'])
        assert ledger[event]=={'event_id':event,'query':token,'target':row['target_after'],'source':'observed_stream'}
        bank[token] = {'query':token,'label':row['target_after'],'presented_at':event,'receipt':receipt}
        if len(bank)==size:
            break
    assert len(bank)==size
    return list(bank.values())


def measure(logits,label):
    z = np.array(logits)
    return {'nll':float(np.logaddexp.reduce(z)-z[label-1]),'margin':float(z[label-1]-np.delete(z,label-1).max()),
            'argmax':int(np.argmax(z))+1,'correct':int(np.argmax(z)+1==label)}


def memory_gate(before,after,bound):
    """Both arguments contain only older receipted A witnesses."""
    exact(list(before),list(after))
    diffs = [after[k]['nll']-before[k]['nll'] for k in before]
    lost = sum(bool(before[k]['correct'] and not after[k]['correct']) for k in before)
    return {'max_delta_nll':max(diffs),'lost_correct':lost,'passes':not lost and max(diffs)<=bound}


def acquisition(before,after,minimum):
    """The current B result is legal here only after its evidence is revealed."""
    gain = before['nll']-after['nll']
    lost = bool(before['correct'] and not after['correct'])
    return {'nll_gain':gain,'lost_correct':lost,'passes':gain>=minimum and not lost}


def select(feasible,order):
    selected = next((name for name in order if feasible.get(name,False)),None)
    return {'selected':selected,'decision':'ACCEPTED_RAW' if selected=='RAW' else 'REPAIRED' if selected else 'NO_FEASIBLE_UPDATE'}


def proposals(k,raw,ctx,cap):
    nr,nc = (float(np.linalg.norm(x)) for x in (raw,ctx))
    made = {'RAW':d4.match_update(k,raw,nr,cap,original=True),
            'RAW_SMALL':d4.match_update(k,raw,nc,cap),
            'CTX':d4.match_update(k,ctx,nc,cap,original=True)}
    small = made['RAW_SMALL'][0]
    for name,w in WEIGHTS.items():
        if small is None:
            made[name] = None,None,{'status':'INFEASIBLE_MATCH','reason':'raw_common_radius_unavailable'}
        else:
            # A convex combination of two clipped, physically possible vectors,
            # followed by a uniform rescale to the common radius when allowed.
            mix = (1-w)*small+w*ctx
            made[name] = d4.match_update(k,mix,nc,cap)
    return made


def observe(agent,k,query):
    ctx,token = query.split(':')
    q = int(token)
    logits,_,final_rng,steps = d4.parent.neutral_read(agent,q,d4.parent.c0.vector('separate',ctx,q),k)
    return {'logits':logits,'rng_post':final_rng,'substeps':steps}


def key_reads(bank,q):
    return [*(f"A:{w['query']}" for w in bank),f'B:{q}']


def old_scores(bank,reads):
    return {k:measure(reads[k]['logits'],w['label']) for w in bank for k in (f"A:{w['query']}",)}


def reveal(log,receipt,rules):
    packet = log.committed(receipt)
    q = packet['query']
    exact(packet['observation'],{'context':'B','query':q})
    log.last_receipt = None
    return rules['B'][q-1]


def run_unit(job):
    root,plan,seed,mapping,preflight = job
    assert [seed,mapping] in plan['units']
    d4_plan = plan['d4']
    parent = d4.parent
    _,rules = parent.c0.block(d4_plan['base_plan']['c0'],seed,mapping)
    block = f'{seed}_{mapping}'
    base = f'blocks/{block}/'
    path = root/('preflight' if preflight else 'blocks')/block
    path.mkdir(parents=True,exist_ok=False)
    archived = load_state(input_path(plan,base+'START.npz'))
    anchor = archived['agent']
    anchor_sha = state_hash(anchor)
    agent = parent.c0.new_agent('B3',seed,d4_plan['base_plan']['c0']['parent'])
    agent.restore(anchor)
    assert state_hash(agent.snapshot())==anchor_sha
    assert agent.delta_core_max==plan['clip']
    save_state(path/'START.npz',archived)
    k = anchor['core_snapshot']['K']
    bank = historical_bank(input_path(plan,'D0_D3_ORIGINAL.zip'),seed,mapping,anchor,plan['bank_size'])
    before_bank = {tag:observe(agent,k,tag) for tag in [f"A:{w['query']}" for w in bank]}
    before_a = old_scores(bank,before_bank)
    original_actions = [json.loads(row) for row in input_path(plan,base+'ACTIONS_BEFORE_FEEDBACK.jsonl').read_bytes().splitlines()]
    actions = ActionJournal(path/'ACTIONS_BEFORE_FEEDBACK.jsonl')
    decisions = ActionJournal(path/'DECISIONS_BEFORE_DIAGNOSTICS.jsonl')
    events, proposed, applied, post = [], [], [], []
    gate_steps = 0
    for q in range(1,7):
        staging = copy.deepcopy(agent)
        packet = forecast(staging,q,'Z0-64','aligned',None,{},stimulus=parent.c0.vector('separate','B',q))
        packet['observation'] = {'context':'B','query':q}
        exact(packet,original_actions[q-1])
        receipt = actions.publish(packet)
        observed_B = reveal(actions,receipt,rules)
        before_b = measure(packet['logits'],observed_B)
        staging.restore(anchor)
        assert state_hash(staging.snapshot())==anchor_sha
        _,token_z,rng,steps = parent.neutral_read(agent,q,parent.c0.vector('separate',None,q))
        external = packet['external']
        assert rng==external['rng_post']
        gate_steps += 288+steps
        raw_phi = np.array(external['eligibility'])
        ctx_phi = np.tanh(external['Z_after'])-np.tanh(token_z)
        err = parent.error_for(packet['logits'],observed_B)
        raw,ctx = (parent.candidate(agent,err,phi) for phi in (raw_phi,ctx_phi))
        made = proposals(k,raw,ctx,agent.delta_core_max)
        candidates = {}
        bank_keys = key_reads(bank,q)
        for name in NAMES:
            proposal,k_after,meta = made[name]
            candidates[name] = {'geometry':meta}
            if k_after is None:
                continue
            reads = {tag:observe(agent,k_after,tag) for tag in bank_keys}
            gate_steps += sum(item['substeps'] for item in reads.values())
            assert reads[f'B:{q}']['rng_post']==rng
            assert all(reads[tag]['rng_post']==before_bank[tag]['rng_post'] for tag in bank_keys[:-1])
            local = acquisition(before_b,measure(reads[f'B:{q}']['logits'],observed_B),plan['min_B_nll_gain'])
            protect = memory_gate(before_a,old_scores(bank,reads),plan['max_A_nll_increase'])
            candidates[name].update(reads=reads,B=local,F=protect,index=len(proposed),K_hash=array_hash(k_after))
            proposed.append(proposal)
            applied.append(k_after-k)
            post.append(k_after)
        feasible = {name:bool(row.get('B',{}).get('passes') and row.get('F',{}).get('passes'))
                    for name,row in candidates.items()}
        choice = select(feasible,plan['selection_order'])
        sealed = {'write_token':q,**choice,'source_action_sha256':receipt['sha256'],
                  'admissible':feasible,'already_confirmed_A_tokens':[w['query'] for w in bank]}
        decision_receipt = decisions.publish(sealed)
        exact(decisions.committed(decision_receipt),sealed)
        decisions.last_receipt = None
        events.append({'write_token':q,'action_receipt':receipt,'target_after':observed_B,
                       'B_before':before_b,'decision_receipt':decision_receipt,'decision':sealed,'candidates':candidates,
                       'phis':{'raw':raw_phi.tolist(),'ctx':ctx_phi.tolist()},'err':err.tolist(),'token_Z':token_z.tolist()})

    # The archived diagnostic response panels are opened only after all choices
    # have been durably fixed. They cannot be used to select a candidate.
    old = read_json(input_path(plan,base+'RESULTS.json'))
    old_proposals = load_state(input_path(plan,base+'DELTAS.npz'))['proposed']
    baseline = old['baseline_panels']['NONE']['logits']
    for key,prediction in before_bank.items():
        exact(prediction['logits'],baseline[key])
    diagnostic_steps, reused = 0,0
    all_keys = [f'{ctx}:{q}' for ctx in ('A','B') for q in range(1,7)]
    for event, previous in zip(events,old['events'],strict=True):
        q = event['write_token']
        assert q==previous['write_token']
        for name,entry in event['candidates'].items():
            if 'index' not in entry:
                continue
            if name in REUSED:
                original = previous['arms'][REUSED[name]]
                exact(proposed[entry['index']],old_proposals[original['vector_index']])
                assert entry['K_hash']==original['probe']['K_hash']
                logits = original['probe']['logits']
                reused += 12-len(entry['reads'])
            else:
                logits = {}
                for name_key in all_keys:
                    if name_key in entry['reads']:
                        logits[name_key] = entry['reads'][name_key]['logits']
                    else:
                        item = observe(agent,post[entry['index']],name_key)
                        assert item['rng_post']==old['baseline_panels']['NONE']['rng_post'][name_key]
                        logits[name_key] = item['logits']
                        diagnostic_steps += item['substeps']
            for tag,item in entry['reads'].items():
                exact(logits[tag],item['logits'])
            entry['diagnostic_logits'] = logits
        exact(event['B_before'],measure(baseline[f'B:{q}'],event['target_after']))
        selected = event['decision']['selected']
        staging = copy.deepcopy(agent)
        if selected:
            staging.core.K[:] = post[event['candidates'][selected]['index']]
        staging.restore(anchor)
        assert state_hash(staging.snapshot())==anchor_sha
    assert state_hash(agent.snapshot())==anchor_sha
    save_state(path/'DELTAS.npz',{'proposed':np.stack(proposed),'applied':np.stack(applied),'K_after':np.stack(post)})
    atomic_json(path/'RESULTS.json',{'unit':[seed,mapping],'snapshot_sha256':anchor_sha,'historical_bank':bank,
                'before_bank':before_bank,'events':events,'diagnostic_none':baseline,'restored_sha256':state_hash(agent.snapshot()),
                'cost':{'gate_substeps':gate_steps+sum(x['substeps'] for x in before_bank.values()),
                        'diagnostic_substeps':diagnostic_steps,'reused_diagnostic_reads':reused}})
    return [seed,mapping]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('phase',choices=('init','preflight','run'))
    for name in ('output','reference','archive'):
        cli.add_argument('--'+name,type=Path,required=name=='output')
    args = cli.parse_args()
    root,phase = args.output.resolve(),args.phase
    if phase=='init':
        init(root,args.reference.resolve(),args.archive.resolve())
        return
    plan = verify_plan(root)
    start = time.perf_counter()
    if phase=='preflight':
        unit = run_unit((root,plan,*plan['preflight_unit'],True))
        atomic_json(root/'STATUS_preflight.json',{'phase':'PASSED','unit':unit,'wall_seconds':time.perf_counter()-start})
        return
    assert read_json(root/'STATUS_preflight.json')['phase']=='PASSED'
    path = root/'STATUS_run.json'
    atomic_json(path,{'phase':'RUNNING','completed':0,'total':len(plan['units'])})
    failures = []
    with ProcessPoolExecutor(max_workers=plan['workers']) as pool:
        jobs = {pool.submit(run_unit,(root,plan,*unit,False)):unit for unit in plan['units']}
        for count,future in enumerate(as_completed(jobs),1):
            try:
                print(count,len(jobs),future.result(),flush=True)
            except Exception as exc:
                failures.append({'unit':jobs[future],'type':type(exc).__name__,'message':str(exc)})
                print(failures[-1],flush=True)
            atomic_json(path,{'phase':'RUNNING','completed':count,'total':len(jobs),'failures':failures})
    atomic_json(path,{'phase':'COMPLETED' if not failures else 'COMPLETED_WITH_FAILURES',
                     'total':len(jobs),'failures':failures,'wall_seconds':time.perf_counter()-start})
    if failures:
        sys.exit(2)


if __name__=='__main__':
    main()
