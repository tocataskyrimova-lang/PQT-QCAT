import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from life import Life,Config,PhysicalCore

class LifeTests(unittest.TestCase):
    def test_independent_energy_gradient(self):
        core=PhysicalCore(Config(cells=2));z=np.random.default_rng(1).normal(0,.3,(2,8));got=core.grad(z);finite=np.zeros_like(z)
        for i in range(2):
            for j in range(8):
                d=np.zeros_like(z);d[i,j]=1e-5;finite[i,j]=(core.energy(z+d)-core.energy(z-d))/2e-5
        np.testing.assert_allclose(got,finite,atol=1e-8,rtol=1e-7)
    def test_closed_small_step_energy(self):
        core=PhysicalCore(Config(h=.002));old=core.energy()
        for _ in range(500):
            core.tick(np.zeros((6,3)));new=core.energy();self.assertLessEqual(new,old+1e-9);old=new
    def test_adaptive_guard_handles_audit_counterexample(self):
        core=PhysicalCore(Config(cells=1,h=.08));core.z[:]=0;core.z[0,0]=4;old=core.energy()
        core.tick(np.zeros((1,3)));self.assertTrue(np.isfinite(core.z).all());self.assertLess(core.energy(),old);self.assertGreater(core.subdivisions,0)
    def test_substeps_do_not_change_learning_clock(self):
        a=Life(Config(integration_substeps=1));b=Life(Config(integration_substeps=4))
        a.predict(3,'A');b.predict(3,'A');self.assertEqual(a.clock,b.clock)
        self.assertLess(np.linalg.norm(a.core.z-b.core.z),.005)
    def test_target_is_unavailable_before_prediction(self):
        a=Life();b=Life.restore(a.snapshot());ra=a.predict(2,'A');rb=b.predict(2,'A')
        self.assertEqual(ra,rb)
        a.feedback(ra['event_id'],1);b.feedback(rb['event_id'],7)
        self.assertGreater(np.linalg.norm(a.contexts['A']['fast']-b.contexts['A']['fast']),0)
    def test_learning_changes_actual_synapses(self):
        a=Life();theta=a.encoder.copy();r=a.predict(2,'A');a.idle(23);a.feedback(r['event_id'],5)
        self.assertGreater(np.linalg.norm(a.encoder-theta),0);self.assertGreater(np.linalg.norm(a.contexts['A']['fast']),0)
    def test_state_persists(self):
        a=Life();a.predict(3,'A');z=a.core.z.copy();a.predict(4,'A')
        self.assertFalse(np.array_equal(z,a.core.z));self.assertEqual(a.stats['logical_initializations'],1);self.assertEqual(a.stats['core_resets'],0)
    def test_idle_has_no_external_event_or_new_teacher(self):
        a=Life();r=a.predict(1,'A');a.feedback(r['event_id'],2);n=a.stats['external_events'];f=a.stats['feedback'];z=a.core.z.copy();a.idle(100)
        self.assertEqual(a.stats['external_events'],n);self.assertEqual(a.stats['feedback'],f)
        self.assertGreater(a.stats['replay_updates'],0);self.assertFalse(np.array_equal(z,a.core.z))
    def test_consolidation_preserves_current_function(self):
        a=Life();r=a.predict(1,'A');a.feedback(r['event_id'],2);b=a.contexts['A'];old=b['fast']+b['slow'];a.consolidate()
        np.testing.assert_allclose(old,b['fast']+b['slow'],atol=1e-15)
    def test_checkpoint_exact_continuation_with_pending_feedback(self):
        a=Life();r=a.predict(2,'A');a.idle(21)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'snapshot.json.gz';a.save(p);b=Life.load(p)
            a.feedback(r['event_id'],4);b.feedback(r['event_id'],4);a.idle(127);b.idle(127)
            self.assertEqual(a.predict(1,'B'),b.predict(1,'B'));np.testing.assert_array_equal(a.core.z,b.core.z);np.testing.assert_array_equal(a.encoder,b.encoder)
    def test_feedback_cannot_be_reused(self):
        a=Life();r=a.predict(1);a.feedback(r['event_id'],2)
        with self.assertRaises(KeyError):a.feedback(r['event_id'],3)
    def test_memory_is_bounded(self):
        a=Life(Config(replay_capacity=4))
        for i in range(12):r=a.predict(i%8,'A');a.feedback(r['event_id'],i%8)
        self.assertEqual(len(a.memory),4)
    def test_frozen_control_does_not_change_synapses(self):
        a=Life(Config(learning=False));old=a.encoder.copy();r=a.predict(1,'A');a.feedback(r['event_id'],2);a.idle(100)
        np.testing.assert_array_equal(old,a.encoder);self.assertEqual(np.linalg.norm(a.contexts['A']['fast']),0);self.assertEqual(a.stats['replay_updates'],0)
    def test_invalid_inputs_rejected(self):
        a=Life()
        for token in (-1,8,float('nan')):
            with self.assertRaises(ValueError):a.predict(token)
    def test_nonfinite_readout_is_not_an_answer(self):
        a=Life();a._context('A')['fast'][0,0]=float('nan')
        with self.assertRaises(FloatingPointError):a.predict(1,'A')
    def test_generation_uses_previous_output(self):
        a=Life();seq=a.generate(3,'A',5);self.assertEqual(len(seq),5)
        self.assertEqual(a.pending[seq[0]['event_id']]['token'],3)
        for prev,cur in zip(seq,seq[1:]):self.assertEqual(a.pending[cur['event_id']]['token'],prev['token'])
    def test_guard_rejects_anchor_degrading_update(self):
        a=Life(Config(guarded_replay=True));a._context('A');e=np.zeros(a.features);e[-1]=1
        a.guard_cache={'A':{'x':e[None,:],'y':np.array([1])}};a.guard_cache_clock=a.clock
        before=a.contexts['A']['fast'].copy();accepted=a._replay_update({'context':'A','eligibility':e,'target':2})
        self.assertFalse(accepted);np.testing.assert_array_equal(before,a.contexts['A']['fast']);self.assertEqual(a.stats['replay_rejections'],1)
    def test_guarded_checkpoint_continuation(self):
        a=Life(Config(guarded_replay=True))
        for i in range(16):r=a.predict(i%8,'A');a.feedback(r['event_id'],(i+1)%8)
        a.idle(35)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'state.gz';a.save(p);b=Life.load(p);a.idle(200);b.idle(200)
            np.testing.assert_array_equal(a.contexts['A']['fast'],b.contexts['A']['fast']);self.assertEqual(a.predict(2,'A'),b.predict(2,'A'))
    def test_state_only_replay_has_activity_without_replay_weight_writes(self):
        a=Life(Config(replay_weight_updates=False,fast_decay=0,consolidation=False))
        for i in range(16):r=a.predict(i%8,'A');a.feedback(r['event_id'],(i+1)%8)
        weights=a.contexts['A']['fast'].copy();z=a.core.z.copy();a.idle(100)
        np.testing.assert_array_equal(weights,a.contexts['A']['fast']);self.assertEqual(a.stats['replay_updates'],0)
        self.assertGreater(a.stats['internal_replay_events'],0);self.assertGreater(np.linalg.norm(a.core.z-z),0)

if __name__=='__main__':unittest.main(verbosity=2)
