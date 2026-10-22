"""Executable tests of rolling leakage guards, routing, year budgets and metric accounting."""
import unittest
import numpy as np
from protocol import *
from metrics import classification,cascade

class Contracts(unittest.TestCase):
    def test_dedup(self):
        a=[dict(sha256='a',year=2018,label=1,order=0),dict(sha256='a',year=2017,label=1,order=1)]
        k,e=deduplicate(a);self.assertEqual(k[0]['year'],2017);self.assertEqual(len(e),1)
        a[0]['label']=0
        with self.assertRaises(RuntimeError):deduplicate(a)
    def test_scope(self):
        r=[dict(sha256=str(y),year=y,label=y%2,order=y) for y in range(2017,2022)]
        for w in WINDOWS:
            tr,va=split_window(r,w);self.assertTrue(all(r[t]['year']<r[va[0]]['year'] for t in tr))
        with self.assertRaises(RuntimeError):audit_scope(r,np.array([1]),np.array([1]),[2018],2018)
    def test_router(self):
        m=np.array([[.2,.3,.4,.1],[.8,.6,.9,.8]])
        self.assertTrue(np.allclose(mean_opposition(m,m[:,3]),[.25,.225]))
        self.assertEqual(top_budget([.5]*10,[str(i) for i in range(10)]).sum(),2)
        self.assertTrue(top_budget([.5]*10,[str(i) for i in range(10)])[0])
        with self.assertRaises(RuntimeError):mean_opposition(m,[.2,.8])
    def test_sampling(self):
        y=np.array([2017]*21+[2018]*5+[2019]*2);a=year_balanced_order(y,np.random.RandomState(1));self.assertEqual(len(a),len(y));c=np.unique(y[a],return_counts=True)[1];self.assertLessEqual(c.max()-c.min(),1)
        one=np.array([2017]*10);self.assertTrue(np.array_equal(year_balanced_order(one,np.random.RandomState(4)),np.random.RandomState(4).permutation(10)))
    def test_metrics(self):
        r=cascade(np.array([0,1,0,1]),np.array([.8,.2,.2,.8]),np.array([.1,.8,.9,.9]),np.ones(4,bool))
        self.assertEqual((r['Fixed'],r['Broken'],r['Net']),(2,1,1));self.assertEqual(r['ClassFlips'],3)

if __name__=='__main__':unittest.main()
