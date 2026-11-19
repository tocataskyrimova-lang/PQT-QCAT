"""Independent audit of the locked D0–D3 logs and scoped mechanisms."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pqt_qcat.memory import qnuf_context_bind as b
from pqt_qcat.memory.access_state import exact
from pqt_qcat.memory.checkpoint_io import load_state
from pqt_qcat.memory.codex_memory_location import atomic_json
from pqt_qcat.memory.functional_sidecar import ActionJournal,read_json,state_hash
from pqt_qcat.memory.qnuf import array_hash
from scripts.analyze_access_bridge import paired
from scripts.analyze_functional_sidecar import has_target,write_csv
from scripts.analyze_qnuf_tworule import score,summary,verify_probes,verify_train
from scripts.decode_context_bind import evaluate

import numpy as np


def check_reference(plan):
    reference = Path(plan['reference'])
    for seed,mapping in plan['units']:
        for mode,arm in b.c0.NEURAL:
            path = b.path_for(None,plan,'joint',seed,mapping,mode,arm)
            for name in ('start.npz','end.npz','PROBES.json'):
                file = path/name
                assert b.c0.causal.file_sha(file)==plan['reference_inputs'][file.relative_to(reference).as_posix()]


def d0(root,plan):
    rows, events = [], 0
    for seed,mapping in plan['units']:
        rules = plan['c0']['mappings'][mapping]
        for mode,arm in b.c0.NEURAL:
            path = b.path_for(root,plan,'joint',seed,mapping,mode,arm)
            origin = load_state(path/'start.npz')['agent']
            end = load_state(path/'end.npz')
            probes,_,_ = verify_probes(Path(plan['reference']),plan['c0'],seed,mapping,mode,arm,origin,end)
            rows.append({'seed':seed,'mapping':mapping,'mode':mode,'arm':arm,'training':'joint','probe':'z0',**probes['z0']})
            for anchor in b.CONTEXTS:
                solo = b.single_plan(plan,anchor)
                start,end,ev,_,_ = verify_train(root/'single'/anchor,solo,seed,mapping,mode,arm)
                events += len(ev)
                assert state_hash(origin)==state_hash(start)
                read = read_json(b.path_for(root,plan,anchor,seed,mapping,mode,arm)/'PROBES.json')
                assert read['snapshot_hash']==state_hash(end['agent']) and read['substeps']==12*128
                rows.append({'seed':seed,'mapping':mapping,'mode':mode,'arm':arm,'training':anchor,'probe':'z0',
                             **score(read['logits'],rules)})
    seed,mapping = plan['units'][0]
    resumes = 0
    for mode,arm in b.c0.NEURAL:
        for anchor in b.CONTEXTS:
            directory = root/'single'/anchor
            main,resume = (b.c0.where(folder,seed,mapping,mode,arm) for folder in (directory,directory/'resume'))
            assert read_json(resume/'COMPLETE.json')['exact']
            assert state_hash(load_state(main/'end.npz'))==state_hash(load_state(resume/'end.npz'))
            exact(read_json(main/'events.json')[plan['solo_cut']:],read_json(resume/'events.json'))
            assert (main/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True)[plan['solo_cut']:] == (
                    (resume/'pre_feedback.jsonl').read_bytes().splitlines(keepends=True))
            resumes += 1
    return rows,events,resumes


def norm_blocks(delta):
    blocks = delta.reshape(6,8,6,8).transpose(0,2,1,3)
    return np.linalg.norm(blocks,axis=(2,3)),blocks


def cosine(a,b):
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(np.vdot(a,b)/norm,-1,1)) if norm else None


def d1_d2(root,plan):
    effects, directions, spatial = [],[],[]
    substeps, count = 0,0
    for seed,mapping in plan['units']:
        for mode,arm in b.c0.NEURAL:
            for anchor in (('joint','A','B') if (mode,arm)==b.PRIMARY else ('joint','A')):
                ref = b.path_for(root,plan,anchor,seed,mapping,mode,arm)
                end = load_state(ref/'end.npz')['agent']
                core = end['core_snapshot']
                directory = root/'diagnostics'/anchor/f'{seed}_{mapping}'/mode/arm
                data = read_json(directory/'RESULTS.json')
                stored = load_state(directory/'DELTAS.npz')
                exact(core['K'],stored['K'])
                assert data['snapshot_sha256']==state_hash(end) and data['anchor']==anchor
                assert [r['token'] for r in data['rows']]==list(range(1,7))
                shadow_check = [seed,mapping]==plan['units'][0] and (mode,arm)==b.PRIMARY and anchor=='A'
                if shadow_check:
                    agent = b.agent_for(ref,plan,seed)
                    shadow = b.c0.read_trace(agent,None,agent.Z0,core['K'],64)['trace']['64']['Z']
                journal = ActionJournal(directory/'pre_feedback.jsonl')
                offset,num,witness_budget = 0,0,0
                for row in data['rows']:
                    q = row['token']
                    label = {'seed':seed,'mapping':mapping,'mode':mode,'arm':arm,'anchor':anchor,'token':q}
                    errors, features, base = {},{},{}
                    ftypes = tuple(row['features']['A'])
                    exact(ftypes,('raw','token_given_context','context_given_token') if (mode,arm)==b.PRIMARY else ('raw',))
                    for ctx in b.CONTEXTS:
                        item = row['before'][ctx]
                        p,receipt = item['packet'],item['receipt']
                        assert receipt['offset']==offset and not has_target(p)
                        offset += receipt['length']
                        exact(journal.read_verified(receipt),p)
                        exact(p['observation'],{'context':ctx,'query':q})
                        assert p['pre_sha256']==state_hash(end) and p['action']==np.argmax(p['logits'])+1
                        exact(p['external']['hashes'], {'K':array_hash(core['K']),'W_readout':array_hash(core['W_readout']),
                                                        'B':array_hash(core['B'])})
                        target = plan['c0']['mappings'][mapping][ctx][q-1]
                        assert target==row['truth_after'][ctx]
                        exact(p['main']['input'],b.c0.vector(mode,ctx,q).tolist())
                        external,main = p['external'],p['main']
                        raw_side = np.tanh(external['Z_after'])-np.tanh(external['Z_no_input'])
                        exact(raw_side.tolist(),external['eligibility'])
                        base[ctx] = p['logits']
                        exact(base[ctx],external['logits'])
                        raw = raw_side if arm=='SS' else np.tanh(main['Z_post'])-np.tanh(main['Z_noinput'])
                        errors[ctx] = b.error_for(base[ctx] if arm=='SS' else main['logits'],target)
                        features[ctx] = {'raw':raw}
                        if len(ftypes)>1:
                            ctl = row['controls']
                            token_x = b.c0.vector(mode,None,q)
                            exact(ctl['no_cue']['input'],token_x.tolist())
                            exact(ctl[ctx]['input'],(b.c0.vector(mode,ctx,q)-token_x).tolist())
                            assert ctl['no_cue']['rng_post']==ctl[ctx]['rng_post']==external['rng_post']
                            full = np.tanh(external['Z_after'])
                            features[ctx].update(token_given_context=full-np.tanh(ctl[ctx]['Z']),
                                                 context_given_token=full-np.tanh(ctl['no_cue']['Z']))
                        exact(row['features'][ctx],{k:v.tolist() for k,v in features[ctx].items()})
                        if shadow_check:
                            exact(shadow,external['Z_no_input'])
                        witness_budget += 288
                    if len(ftypes)>1:
                        witness_budget += 3*128
                    deltas = {}
                    assert len(row['interventions'])==2*len(ftypes)
                    for effect in row['interventions']:
                        feature,ctx = effect['feature'],effect['write_context']
                        phi = features[ctx][feature]
                        candidate = np.clip(plan['c0']['parent']['eta'] * np.outer(core['W_readout'].T@errors[ctx],phi),
                                            -plan['c0']['parent']['clip'],plan['c0']['parent']['clip'])
                        exact(candidate,stored['delta'][num])
                        num += 1
                        assert effect['K_post_hash']==array_hash(core['K']+candidate)
                        deltas[(feature,ctx)] = candidate
                        for read_ctx in b.CONTEXTS:
                            target = row['truth_after'][read_ctx]
                            original,changed = np.array(base[read_ctx]),np.array(effect['logits'][read_ctx])
                            if shadow_check and feature=='raw' and q==1:
                                predicted,_,_,_ = b.neutral_read(agent,q,b.c0.vector(mode,read_ctx,q),core['K']+candidate)
                                exact(predicted,changed.tolist())
                            nll0 = float(np.logaddexp.reduce(original)-original[target-1])
                            nll1 = float(np.logaddexp.reduce(changed)-changed[target-1])
                            effects.append({**label,'feature':feature,'write_context':ctx,'read_context':read_ctx,
                                            'target':target,'before_class':int(original.argmax())+1,
                                            'after_class':int(changed.argmax())+1,'before_correct':int(original.argmax()+1==target),
                                            'after_correct':int(changed.argmax()+1==target),'nll_before':nll0,'nll_after':nll1,
                                            'nll_delta':nll1-nll0,'logit_delta_linf':float(np.max(np.abs(changed-original)))})
                            witness_budget += 128
                    for feature in ftypes:
                        a,bc = (deltas[(feature,ctx)] for ctx in b.CONTEXTS)
                        pa,pb = (features[ctx][feature] for ctx in b.CONTEXTS)
                        common = [np.clip(plan['c0']['parent']['eta']*np.outer(core['W_readout'].T@errors['A'],v),
                                          -plan['c0']['parent']['clip'],plan['c0']['parent']['clip']) for v in (pa,pb)]
                        fixed_cos = cosine(*common)
                        saved_cos = row['fixed_error_cos'][feature]
                        if fixed_cos is None:
                            assert saved_cos is None
                        else:
                            assert abs(saved_cos-fixed_cos)<1e-13
                        directions.append({**label,'feature':feature,'delta_norm_A':float(np.linalg.norm(a)),
                                           'delta_norm_B':float(np.linalg.norm(bc)),'delta_cos':cosine(a,bc),
                                           'phi_norm_A':float(np.linalg.norm(pa)),'phi_norm_B':float(np.linalg.norm(pb)),
                                           'phi_cos':cosine(pa,pb),'same_error_cos':fixed_cos})
                        na,ba = norm_blocks(a)
                        nb,bb = norm_blocks(bc)
                        for i in range(6):
                            for j in range(6):
                                spatial.append({**label,'feature':feature,'receiving_echelon':i+1,'sending_echelon':j+1,
                                                'norm_A':float(na[i,j]),'norm_B':float(nb[i,j]),
                                                'dot':float(np.vdot(ba[i,j],bb[i,j])),'cosine':cosine(ba[i,j],bb[i,j])})
                    count += 1
                assert offset==journal.path.stat().st_size and num==len(stored['delta']) and witness_budget==data['substeps']
                substeps += witness_budget
    return effects,directions,spatial,count,substeps


def mean_effect(rows,plan,correct):
    by_block = []
    n,loss,gain = 0,0,0
    for seed,mapping in plan['units']:
        group = [r for r in rows if (r['seed'],r['mapping'])==(seed,mapping)
                 and (r['before_correct'] or not correct)]
        by_block.append(float(np.mean([r['nll_delta'] for r in group])) if group else np.nan)
        n += len(group)
        loss += sum(r['before_correct'] and not r['after_correct'] for r in group)
        gain += sum(not r['before_correct'] and r['after_correct'] for r in group)
    boot = {'parent':plan['c0']['parent'],'bootstrap_seed':plan['bootstrap_seed'],
            'bootstrap_replicates':plan['bootstrap_replicates']}
    return {**paired(by_block,boot),'queries':n,'lost_correct':loss,'gained_correct':gain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    plan = b.verify_plan(root)
    if args.reference:
        plan['reference'] = str(args.reference.resolve())
    check_reference(plan)
    phases = {p:read_json(root/f'STATUS_{p}.json') for p in ('single','diagnostics','horizon','resume')}
    assert all(v['phase']=='COMPLETED' for v in phases.values())
    single,train_events,resumes = d0(root,plan)
    effects,directions,spatial,count,steps = d1_d2(root,plan)
    decoded,predictions,folds = evaluate(root,plan)
    horizons = []
    for seed,mapping in plan['units']:
        saved = read_json(root/'horizon128'/f'{seed}_{mapping}.json')
        p = b.path_for(root,plan,'joint',seed,mapping,*b.PRIMARY)
        assert saved['snapshot_sha256']==state_hash(load_state(p/'end.npz')['agent']) and saved['substeps']==12*256
        horizons.append({'seed':seed,'mapping':mapping,'mode':b.PRIMARY[0],'arm':b.PRIMARY[1],
                         'probe':'native128',**score(saved['logits'],plan['c0']['mappings'][mapping])})
    score_keys = ('acc_A','acc_B','macro','nll_A','nll_B','nll','changed_actions','correct_pairs')
    d0_means = summary(single,('mode','arm','training'),score_keys)
    d3 = [dict(row,probe='native64') for row in single if row['training']=='joint'] + decoded + horizons
    d3_means = summary(d3,('mode','arm','probe'),score_keys)
    all_effects = []
    keys = ('mode','arm','anchor','feature','write_context','read_context')
    effect_group = {}
    for row in effects:
        effect_group.setdefault(tuple(row[k] for k in keys),[]).append(row)
    for group,rows in effect_group.items():
        all_effects.append({**dict(zip(keys,group,strict=True)),'all':mean_effect(rows,plan,False),
                            'before_correct_only':mean_effect(rows,plan,True)})
    mechanisms = {}
    for mode,arm in b.c0.NEURAL:
        solo = {r['training']:r for r in d0_means if (r['mode'],r['arm'])==(mode,arm)}
        both_solo = min(solo['A']['acc_A'],solo['B']['acc_B'])>=plan['solo_gate']
        hit = next(r for r in all_effects if tuple(r[k] for k in keys)==(mode,arm,'A','raw','B','A'))['before_correct_only']
        valid_hit = hit['ci95'] is not None and hit['ci95'][0]>0
        dec = {r['probe']:r for r in d3_means if (r['mode'],r['arm'])==(mode,arm)}
        decode_support = (min(dec['decoded']['acc_A'],dec['decoded']['acc_B'])>=plan['solo_gate']
                          and dec['decoded']['macro']>max(dec[name]['macro'] for name in
                              ('native64','decoded_K0','decoded_wrong','decoded_swap','decoded_blind')))
        long_read = dec.get('native128')
        mechanisms[f'{mode}/{arm}'] = {'H1':'SINGLE_RULE_FAILURE_NOT_OBSERVED' if both_solo else 'SINGLE_RULE_FAILURE',
                'H2':('DIRECT_WRITE_INTERFERENCE_SUPPORTED' if valid_hit else 'NOT_ESTABLISHED') if both_solo
                else 'NOT_INTERPRETED_SINGLE_RULE_GATE',
                'H3':('INACCESSIBLE_CO_STORAGE_SUPPORTED' if decode_support else
                      'HORIZON_LIMITED_ACCESS_SUPPORTED' if long_read and
                      min(long_read['acc_A'],long_read['acc_B'])>=plan['solo_gate'] else 'FULL_CO_STORAGE_NOT_ESTABLISHED'),
                'A_only_accuracy':solo['A']['acc_A'],'B_only_accuracy':solo['B']['acc_B'],'B_to_A_on_correct_A':hit}
    focus = mechanisms['/'.join(b.PRIMARY)]
    diagnosed = (focus['H1']=='SINGLE_RULE_FAILURE' or focus['H2']=='DIRECT_WRITE_INTERFERENCE_SUPPORTED'
                 or focus['H3'] in ('INACCESSIBLE_CO_STORAGE_SUPPORTED','HORIZON_LIMITED_ACCESS_SUPPORTED'))
    out = {'verdict':'CONTEXT_BINDING_DIAGNOSED' if diagnosed else 'CAUSE_NOT_IDENTIFIED',
           'mechanisms':mechanisms,'phase_status':phases,'solo_verified_actions':train_events,'exact_resumes':resumes,
           'diagnostic_states':count,'diagnostic_substeps':steps,'d0_means':d0_means,'d3_means':d3_means,
           'update_effects':all_effects,'decoder_folds':folds,'decoder_controls_exact_restore':len(folds)}
    if args.verify_only:
        exact(out,read_json(root/'ANALYSIS.json'))
    else:
        atomic_json(root/'ANALYSIS.json',out)
        atomic_json(root/'DECODER_PREDICTIONS.json',predictions)
        for name,data in [('D0_BY_BLOCK',single),('D1_D2_EFFECTS',effects),('D1_D2_DIRECTIONS',directions),
                          ('D1_D2_ECHELONS',spatial),('D3_BY_BLOCK',d3)]:
            write_csv(root/f'{name}.csv',data)
    print(json.dumps({k:out[k] for k in ('verdict','mechanisms','solo_verified_actions','exact_resumes',
                                          'diagnostic_states','decoder_controls_exact_restore')},indent=2))


if __name__=='__main__':
    main()
