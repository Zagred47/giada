import unittest
import numpy as np
import torch
from src.giada_teacher import calcium_gate_transfer as c
from src.giada_teacher import single_gate_transfer as e

class Task20Tests(unittest.TestCase):
    def test_canonical_branch_units(self):
        self.assertEqual(float(c.canonical_inf(.00043)),.5)
        self.assertEqual(float(c.canonical_inf(0)),float(c.canonical_inf(1e-7)))
        below=np.nextafter(1e-7,0)
        self.assertGreater(float(c.canonical_inf(below)),float(c.canonical_inf(1e-7)))
        with self.assertRaises(ValueError):c.canonical_inf(-1.)

    def test_affine_semigroup_grid(self):
        self.assertFalse(np.isin(c.grid()[:,0],c.grid(True)[:,0]).any())
        x=c.data(1,100);i,t=c.rates('',x[:,0]);y=e.update(x[:,1],i,t,x[:,2])
        self.assertTrue(np.all((y>=0)&(y<=1)))
        np.testing.assert_allclose(e.update(y,i,t,x[:,2]),e.update(x[:,1],i,t,2*x[:,2]),atol=1e-15)
        self.assertEqual(c.measurements(x,y,i,t,'')['gate_rmse'],0.)

    def test_vectorization_and_control_information(self):
        torch.set_num_threads(1)
        descriptors=[(ch,o,s) for ch in c.CHANNELS for o in ('transition_only','rate_supervised') for s in (17,29,43)]
        for w in (16,32):
            m=c.model_factory(torch,w,descriptors)
            self.assertEqual(sum(p.numel() for p in m.parameters())//18,c.parameter_count(w))
            x=torch.tensor(c.data(2,16),dtype=torch.float32)[None].expand(18,-1,-1).clone()
            self.assertTrue(c.equivalent(torch,m,w,descriptors,x)['valid'])
            with torch.no_grad():
                before=m(x)[1];x[...,0]+=1.;after=m(x)[1]
            torch.testing.assert_close(before[12:],after[12:],rtol=0,atol=0)
            self.assertGreater(float((before[:6]-after[:6]).abs().max()),0.)

if __name__=='__main__':unittest.main()
