import json,unittest
from pathlib import Path
import numpy as np
from src.giada_teacher import task42_axial_exploration as x
ROOT=Path(__file__).resolve().parents[1]

class AxialPort(unittest.TestCase):
    def test_console_notebook(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        import task42_kaggle as k
        doc=k.notebook('a'*40,'b'*64);code=''.join(doc['cells'][1]['source'])
        compile(code,'42-notebook','exec')
        self.assertIn('neuron==8.2.7',code);self.assertIn('run_task42.py',code)
        self.assertIn('SelfishGene/neuron_as_deep_net',code)
    def test_geometry_units_and_reciprocity(self):
        for geo in [x.geometry(k) for k in (2,4,8,16)]+[x.geometry(5,True)]:
            area,lap=x.matrix(geo,1.)
            self.assertTrue((area>0).all());np.testing.assert_allclose(lap,lap.T,atol=0)
            np.testing.assert_allclose(lap.sum(0),0,atol=1e-15)
            self.assertGreaterEqual(np.linalg.eigvalsh(lap).min(),-1e-15)
            np.testing.assert_array_equal(x.matrix(geo,0)[1],0)
            v=np.arange(len(area),dtype=float)-70;current=lap@v
            self.assertAlmostEqual(float(current.sum()),0,places=13)
            a=lap+np.diag(area*1e-5/.1+.0001*area*.01);rhs=area*1e-5*v/.1
            solved=np.linalg.solve(a,rhs);per=np.arange(len(area))[::-1]
            permuted=np.linalg.solve(a[per][:,per],rhs[per]);np.testing.assert_allclose(permuted,solved[per],atol=1e-12)
    def test_bifurcation_shared_junction(self):
        geo=x.geometry(3,True);area,lap=x.matrix(geo,1.)
        resistance=np.array([.01*geo['ra_ohm_cm']*geo['lengths_um'][i]/2/(np.pi*(geo['diameters_um'][i]/2)**2) for i in range(3)])
        h=1/resistance;v=np.array([-80.,-65.,-50.]);junction=(h*v).sum()/h.sum()
        np.testing.assert_allclose(lap@v,h*(v-junction),atol=1e-15)
        self.assertLess(lap[1,2],0)
    def test_design_and_unchanged_contract(self):
        config,cfg,ca,base=x.spec(ROOT)
        self.assertEqual(config['freeze']['stress']['passed'],66)
        for geo in config['geometries']:
            rows,inj,mult=x.design(geo,config)
            self.assertEqual(len(rows),12);self.assertEqual(inj.shape,(400,12,len(geo['parents'])))
            self.assertEqual(mult.shape,(12,len(geo['parents']),11))
            np.testing.assert_array_equal(inj[:,0],0)
            np.testing.assert_array_equal(inj[:,1],inj[:,5])

    def test_frozen_one_step_and_short_rollout(self):
        import os,torch
        torch.set_num_threads(1)
        os.environ['GIADA_TASK41N_FROZEN']=str(ROOT/'artifacts/task41n_frozen.zip.bin')
        config,cfg,ca,base=x.spec(ROOT);geo=x.geometry(2)
        net=x.n.load_models(torch,ROOT,'cpu')[0]['dual']
        with np.load(ROOT/config['freeze']['heads_path']) as z:net=x.repair.materialize(torch,net,dict(z))
        self.assertEqual(x.n.weights_hash(net),config['freeze']['logical_weights_sha256'])
        with torch.inference_mode():
            for model in (None,net):
                run,args,inj,core,info=x.runner(torch,model,geo,config,cfg,ca,base,'cpu')
                actual=core(*args)
                canonical=x.n.m.core(torch,model,(cfg,ca,base),float(x.matrix(geo,0)[0][0]),'voltage_solve64')(*args)
                # First four cases have zero coupling; retain original local core.
                torch.testing.assert_close(actual[0][:,:8],canonical[0][:,:8],rtol=0,atol=1e-10)
                for a in actual:self.assertTrue(torch.isfinite(a).all())
                short=dict(config,duration_ms=.3)
                data,_=x.rollout(torch,model,geo,short,cfg,ca,base,'cpu')
                self.assertEqual(data['voltage'].shape,(4,3,12,2))

if __name__=='__main__':unittest.main()
