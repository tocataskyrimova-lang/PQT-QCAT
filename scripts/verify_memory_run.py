"""Independent artifact checks; no model training and no metric selection."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pqt_qcat.memory.checkpoint_io import load_state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args()
    run=Path(args.run);plan=json.loads((run/'PLAN.json').read_text())
    source=json.loads((run/'SOURCE.json').read_text())
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in source['files'].items())
    freeze=json.loads((run/'READER_FREEZE.json').read_text())
    assert freeze['sha256']==hashlib.sha256((run/'FROZEN_READERS.json').read_bytes()).hexdigest()
    assert len({tuple(x) for x in plan['maps']})==24
    predictions=load_state(run/'predictions.npz')
    metrics=json.loads((run/'RESULTS.json').read_text())
    for mode,p in predictions.items():
        for key,values in p['predictions'].items():
            assert abs(float(np.mean(values==p['targets']))-metrics[mode]['accuracy'][key])<1e-14
    lives=0;events=0;checks=0;cpu=0.;steps=0
    for folder in sorted((run/'lives').iterdir()):
        row=json.loads((folder/'result.json').read_text())
        assert row['failure'] is None
        records=json.loads((folder/'events.json').read_text())
        assert len(records)==plan['events']
        for event in records:
            assert np.argmax(event['logits_before_feedback'])+1==event['prediction_before_feedback']
            assert event['correct']==(event['prediction_before_feedback']==event['target_after_prediction'])
            assert abs(event['actual_delta_k_norm']-event['delta_k_norm'])<1e-12
            assert np.isfinite(event['total_cell_energy'])
        initial=load_state(folder/'checkpoint_000.npz')['agent']['core_snapshot']
        final=load_state(folder/'checkpoint_072.npz')['agent']['core_snapshot']
        features=load_state(folder/'features.npz')
        np.testing.assert_array_equal(final['K'][32:35].reshape(-1),features['K_after'])
        np.testing.assert_array_equal(initial['K'][32:35].reshape(-1),features['K_initial'])
        assert abs(np.linalg.norm(final['K']-initial['K'])-row['actual_K_change_norm'])<1e-12
        lives+=1;events+=len(records);checks+=len(list(folder.glob('checkpoint_*.npz')))
        cpu+=row['cpu_seconds'];steps+=row['training_substeps']
    assert lives==128 and events==9216
    result={'verified_lives':lives,'verified_training_events':events,'checkpoint_count':checks,
            'total_training_substeps':steps,'life_cpu_seconds':cpu,'source_hashes_match':True,
            'reader_hash_matches':True,'predictions_and_metrics_match':True,'actual_updates_match':True}
    (run/'INDEPENDENT_CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
