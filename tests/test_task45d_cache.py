"""Same-process inference preflight then real training, including legacy cache."""
import unittest
import numpy as np
from src.giada_teacher import task40_causal_reinsertion as core
from src.giada_teacher import task45d_learnability_execution as d
from tests import test_task45c as fixtures
class CacheTransition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.PreservingKinetics.setUpClass();cls.torch=fixtures.PreservingKinetics.torch
    def test_cold_preflight_then_backward_preserves_values_and_cache(self):
        t=self.torch
        for dtype in (t.float32,t.float64):
            core._CONSTANTS.clear();v=t.full((3,4),-60.,dtype=dtype);g=t.full((3,4,18),.5,dtype=dtype);mult=t.ones(3,4,11,dtype=dtype)
            with t.inference_mode():
                before=core.readout(t,v,g,mult)[1].clone()
            d.prepare_training_readout(t,'cpu');cached=core.constants(t,v)
            self.assertTrue(all(not t.is_inference(x) for x in cached))
            train_g=g.clone().requires_grad_();actual=core.readout(t,v,train_g,mult)[1];actual.square().sum().backward()
            self.assertTrue(t.equal(before,actual));self.assertTrue(bool(t.isfinite(train_g.grad).all()));self.assertIs(core.constants(t,v)[0],cached[0])
    def test_repairs_legacy_inference_cache(self):
        t=self.torch;core._CONSTANTS.clear();v=t.full((2,3),-60.)
        with t.inference_mode():bad=tuple(t.tensor(x,dtype=v.dtype) for x in (core.ionic.GBAR,core.ionic.REVERSALS,(0.,0.,-80.,-97.)))
        core._CONSTANTS[(str(v.device),v.dtype)]=bad
        d.prepare_training_readout(t,'cpu');repaired=core.constants(t,v);self.assertTrue(all(not t.is_inference(x) for x in repaired))
        for old,new in zip(bad,repaired):self.assertTrue(t.equal(old,new))
    def test_actual45d_loss_after_inference_readout(self):
        t=self.torch;base=fixtures.PreservingKinetics.baseline;p=d.spec(fixtures.ROOT)[0];p['fitting'].update(points=128,duration_ms=.4);p['matrix']['held_ticks']=[1,10]
        core._CONSTANTS.clear()
        with t.inference_mode():core.readout(t,t.full((3,2),-65.),t.full((3,2,18),.5),t.ones(3,2,11))
        d.prepare_training_readout(t,'cpu')
        data=d.c.histories(p,'fit');po=d.pool(t,base,p,data,0,'cpu')
        tensors={k:t.tensor(po[k],dtype=t.float32 if k in ('v','ca') else t.float64) for k in ('v','ca','u','z','raw','label')};tensors['initial']=t.full((128,18),.5,dtype=t.float64)
        for adapt in (False,True):
            view=d.training_view(t,base,po,p,adapt,'cpu');opt=t.optim.Adam([q for q in view.parameters() if q.requires_grad],lr=1e-5)
            for _ in range(2):
                opt.zero_grad();loss=d.train_loss(t,view,tensors,t.arange(16),p).sum();self.assertTrue(bool(t.isfinite(loss)));loss.backward();opt.step()
            self.assertTrue(all(bool(t.isfinite(q).all()) for q in view.parameters()))
if __name__=='__main__':unittest.main()
