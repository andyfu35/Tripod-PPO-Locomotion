"""MuJoCo-independent correctness tests for the original-history H2-K6 port."""
from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from h2k6_tripod_core import (
    ACT, OBS, PARAM_COUNT, GRU26Policy, PrivatePsi, Evidence, Gene, PAESArchive, unpack_parameters,
    q2_parent_probabilities, mutate, propose_relative, random_parameters,
    seeded_direction,
)
from train_segb_h2k6 import (
    Config, eval_pair, commit_results, prepare_batch, save_checkpoint,
    load_checkpoint, train,
)


class FakeEnv:
    def __init__(self, die_on_step=3):
        self.die_on_step = die_on_step
        self.n = 0
        self.reset_seeds = []
        self.step_calls = 0

    def reset(self, *, seed):
        self.n = 0
        self.reset_seeds.append(seed)
        return np.zeros(OBS, np.float32), {}

    def step(self, action):
        assert action.shape == (ACT,)
        self.n += 1
        self.step_calls += 1
        terminated = self.n >= self.die_on_step
        return np.zeros(OBS, np.float32), float(1.0), terminated, False, {
            "net_distance_m": float(self.n) * 0.01,
            "alive": not terminated,
        }

    def close(self):
        pass


class FakePool:
    def __init__(self, die_on_step=3):
        self.steps=0
        self.die_on_step = die_on_step
        self.seeds=[]
        self.tasks=0

    def map(self, function, tasks):
        results=[]
        for task in tasks:
            a,b=FakeEnv(self.die_on_step),FakeEnv(self.die_on_step)
            result=function(task,envs=(a,b))
            self.seeds.append((a.reset_seeds[-1],b.reset_seeds[-1]))
            self.steps+=a.step_calls+b.step_calls
            self.tasks+=1
            results.append(result)
        return results


class TestOriginalH2K6Tripod(unittest.TestCase):
    def test_original_gru_serialization_and_actions(self):
        rng=np.random.default_rng(3)
        p=random_parameters(rng)
        self.assertEqual(p.shape,(PARAM_COUNT,))
        policy=GRU26Policy(p)
        a,_=policy.predict(np.ones(OBS,np.float32))
        self.assertEqual(a.shape,(ACT,))
        self.assertTrue(np.all(np.abs(a)<=1))
        policy.reset(seed=17)
        b,_=policy.predict(np.ones(OBS,np.float32))
        self.assertTrue(np.allclose(a,b))
        self.assertTrue(np.isfinite(a).all())

    def test_gru_is_equivalent_to_original_torch_cell(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("Optional torch is not installed")
        rng=np.random.default_rng(40)
        p=random_parameters(rng)
        obs=rng.normal(0,0.2,OBS).astype(np.float32)
        policy=GRU26Policy(p)
        actual,_=policy.predict(obs)
        weights={k:torch.as_tensor(v.copy()) for k,v in unpack_parameters(p).items()}
        x=torch.as_tensor(obs)
        with torch.no_grad():
            encoded=F.silu(F.linear(x,weights["input_weight"],weights["input_bias"]))
            ig=F.linear(encoded,weights["gru_weight_ih"],weights["gru_bias_ih"])
            hidden=torch.zeros(26)
            hg=F.linear(hidden,weights["gru_weight_hh"],weights["gru_bias_hh"])
            ir,iz,inn=ig.chunk(3)
            hr,hz,hn=hg.chunk(3)
            r=torch.sigmoid(ir+hr)
            z=torch.sigmoid(iz+hz)
            n=torch.tanh(inn+r*hn)
            hidden=(1-z)*n+z*hidden
            features=F.silu(F.linear(hidden,weights["hidden_weight"],weights["hidden_bias"]))
            expected=torch.tanh(F.linear(features,weights["action_weight"],weights["action_bias"]))
        self.assertTrue(np.allclose(actual,expected.numpy(),atol=1e-7))

    def test_signed_seed_replay_inverse_direction(self):
        p=random_parameters(np.random.default_rng(1))
        e=seeded_direction(11)
        self.assertAlmostEqual(float(np.linalg.norm(e)),1.0,places=5)
        plus=mutate(p,11,2.5)
        minus=mutate(p,11,-2.5)
        self.assertTrue(np.allclose(plus+minus,2*p,atol=1e-6))

    def test_private_psi_both_good_and_bad_are_inherited(self):
        psi=PrivatePsi("a","founder")
        psi.evidence.append(Evidence(1,1.5,(0.1,0.05),100,120,False,True,555))
        psi.evidence.append(Evidence(2,-2.5,(-0.5,-0.05),100,50,False,False,666))
        descendant=psi.inherited_by("child","a")
        self.assertEqual(len(descendant.evidence),2)
        self.assertEqual(descendant.lineage,"founder")
        descendant.evidence[0].admitted=False
        self.assertTrue(psi.evidence[0].admitted)
        self.assertEqual(descendant.evidence[1].delta,(-0.5,-0.05))
        with self.assertRaises(ValueError):
            descendant.inherited_by("new","wrong_parent")

    def test_relative_history_rejects_unsafe_movement_shortcut(self):
        p=PrivatePsi("a","a")
        p.evidence=[Evidence(18,5.0,(0.6,-0.4),100,20,False,False,111),
                    Evidence(19,-3.0,(0.03,0.1),100,110,False,True,111)]
        self.assertEqual(p.evidence[0].useful_axes(),(False,False))
        self.assertEqual(p.evidence[1].useful_axes(),(True,True))
        for i in range(8):
            seed,signed,source=propose_relative(p,np.random.default_rng(i),random_fraction=0)
            self.assertEqual(seed,19)
            self.assertLess(signed,0)
            self.assertEqual(source,"relative_guided")

    def test_q2_parent_selection_and_nondominated_bank(self):
        bank=PAESArchive(capacity=3)
        rng=np.random.default_rng(3)
        def gene(k,a,b):
            g=Gene(str(k),random_parameters(rng),PrivatePsi(str(k),str(k)))
            g.progress=a;g.longevity=b
            return g
        a,b,c=gene(1,.3,.9),gene(2,.7,.4),gene(3,.1,.2)
        admitted=bank.commit_batch([a,b,c],rng)
        self.assertEqual(admitted,{"1","2"})
        p=q2_parent_probabilities(bank.ordered())
        self.assertAlmostEqual(float(p.sum()),1.)
        self.assertTrue(np.all(p>0))

    def test_physical_pair_seeds_steps_and_early_fall(self):
        rng=np.random.default_rng(8)
        p=random_parameters(rng)
        task={"theta":(p,mutate(p,18,.2)),"reset_seed":312,
              "budget_cap":2000,"horizon":1000,"parent_id":"a","child_id":"b"}
        a,b=FakeEnv(3),FakeEnv(2)
        r=eval_pair(task,envs=(a,b))
        self.assertEqual(r["used"],5)
        self.assertEqual((a.reset_seeds,b.reset_seeds),([312],[312]))
        self.assertEqual((r["results"][0]["steps"],r["results"][1]["steps"]),(3,2))
        self.assertTrue(all(x["completed"] for x in r["results"]))

    def test_budget_censor_does_not_create_fake_completed_genes(self):
        p=random_parameters(np.random.default_rng(10))
        task={"theta":(p,p),"reset_seed":111,"budget_cap":3,
              "horizon":1000,"parent_id":"a","child_id":"b"}
        a,b=FakeEnv(1000),FakeEnv(1000)
        r=eval_pair(task,envs=(a,b))
        self.assertEqual(r["used"],3)
        self.assertTrue(all(x["censored"] for x in r["results"]))

    def test_de_novo_charged_parent_replay_does_not_overwrite_archive_parent(self):
        rng=np.random.default_rng(119)
        bank=PAESArchive()
        parent=Gene("g0",random_parameters(rng),PrivatePsi("g0","g0"))
        parent.progress,parent.longevity,parent.reward,parent.steps=.8,.95,.75,950
        bank.commit_batch([parent],rng)
        before=(parent.progress,parent.longevity,parent.reward,parent.steps)
        child=Gene("g1",random_parameters(rng),PrivatePsi("g1","g1"))
        entries=[dict(parent=parent,child=child,seed=None,signed_scale=None,
                      mode="de_novo",reset_seed=88)]
        results=[{"used":42,"pair_seed":88,"parent_id":"g0","child_id":"g1",
                  "results":[{"completed":True,"censored":False,"distance":0.,"steps":20,"fell":True,"reward":3.8},
                             {"completed":True,"censored":False,"distance":2.,"steps":22,"fell":True,"reward":5.2}]}]
        history=[]
        commit_results(entries,results,bank=bank,rng=rng,histories=history,
                       steps_used=42,best_reward=None)
        self.assertEqual((parent.progress,parent.longevity,parent.reward,parent.steps),before)
        self.assertEqual(len(parent.psi.evidence),0)
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]["mode"],"de_novo")
        self.assertEqual(history[0]["gene_id"],"g1")
        self.assertEqual(child.psi.lineage,"g1")

    def test_rejected_offspring_updates_parent_private_psi(self):
        rng=np.random.default_rng(20)
        bank=PAESArchive()
        parent=Gene("g0",random_parameters(rng),PrivatePsi("g0","g0"))
        parent.progress,parent.longevity=.5,.8
        bank.commit_batch([parent],rng)
        child=Gene("g1",mutate(parent.theta,5,.1),
                   PrivatePsi("g1","g0",parent="g0"),parent_id="g0")
        entries=[dict(parent=parent,child=child,seed=5,signed_scale=.1,
                      mode="relative_guided",reset_seed=99)]
        # Child worse on both axes -> bank rejects, parent STILL remembers it.
        results=[{"used":1000,"pair_seed":99,"parent_id":"g0","child_id":"g1",
                  "results":[{"completed":True,"censored":False,"distance":10.0,"steps":800,"fell":True,"reward":300},
                             {"completed":True,"censored":False,"distance":0.0,"steps":200,"fell":True,"reward":10}]}]
        hist=[]
        commit_results(entries,results,bank=bank,rng=rng,histories=hist,
                       steps_used=1000,best_reward=None)
        self.assertEqual(len(parent.psi.evidence),1)
        self.assertEqual(parent.psi.evidence[0].child_steps,200)
        self.assertFalse(parent.psi.evidence[0].admitted)
        self.assertEqual(hist[0]["admitted"],0)

    def test_admitted_offspring_inherits_all_parent_records(self):
        rng=np.random.default_rng(21)
        bank=PAESArchive()
        parent=Gene("g0",random_parameters(rng),PrivatePsi("g0","g0"))
        parent.progress,parent.longevity=.0,.1
        parent.psi.evidence.append(Evidence(3,1.,(-.1,-.1),100,50,False,False,2))
        bank.commit_batch([parent],rng)
        child=Gene("g1",mutate(parent.theta,5,.1),
                   PrivatePsi("g1","g0",parent="g0"),parent_id="g0")
        entries=[dict(parent=parent,child=child,seed=5,signed_scale=.1,
                      mode="bootstrap_random",reset_seed=99)]
        results=[{"used":1200,"pair_seed":99,"parent_id":"g0","child_id":"g1",
                  "results":[{"completed":True,"censored":False,"distance":0.,"steps":100,"fell":True,"reward":20},
                             {"completed":True,"censored":False,"distance":2.,"steps":900,"fell":True,"reward":200}]}]
        commit_results(entries,results,bank=bank,rng=rng,histories=[],steps_used=1200,best_reward=None)
        self.assertIn("g1",bank.genes)
        self.assertEqual(len(bank.genes["g1"].psi.evidence),2)
        self.assertEqual(bank.genes["g1"].psi.owner,"g1")
        self.assertTrue(bank.genes["g1"].psi.evidence[-1].admitted)

    def test_exact_step_ledger_and_resume_without_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Config(total_steps=101,checkpoint_every=25,output=str(Path(tmp)/"run"))
            fake=FakePool(die_on_step=3)
            used=train(cfg,pool=fake,verbose=False)
            self.assertEqual(used,101)
            self.assertEqual(fake.steps,101)
            self.assertGreater(fake.tasks,0)
            self.assertTrue(all(a==b for a,b in fake.seeds))
            self.assertTrue((Path(cfg.output)/"checkpoint.npz").is_file())
            self.assertTrue((Path(cfg.output)/"best_stable_progress.npz").is_file())
            before=fake.steps
            used=train(cfg,pool=fake,verbose=False)
            self.assertEqual(used,101)
            self.assertEqual(fake.steps,before)


if __name__=="__main__":
    unittest.main()
