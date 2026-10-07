import copy
import unittest
from scripts.summarize_decode_profile import summarize, NAMES


def fixture():
    trials=[]
    for repeat,length in [(-1,32), *[(r,n) for r in range(3) for n in [32,128,512]]]:
        steps=[]
        for step in range(3 if repeat==-1 else 9):
            active=repeat>0 and step>0 and (repeat+step)%2==0
            calls=[{'name':n,'seconds':0.07} for n in NAMES] if active else []
            entry={'step':step,'seconds':1.1 if active else 1.,'profiled':active,
                   'scores_exact':True,'scores_sha256':f'{length}/{step}','calls':calls}
            if active:entry.update(component_seconds=sum(c['seconds'] for c in calls),outside_predict_seconds=1.1-sum(c['seconds'] for c in calls))
            steps.append(entry)
        trials.append({'repeat':repeat,'prefix_length':length,'warmup':repeat==-1,'disposed':True,'steps':steps})
    return {'status':'completed','passed':True,'trials':trials}

class SummaryTests(unittest.TestCase):
    def test_complete_accounting_and_warmup_exclusion(self):
        s=summarize(fixture())
        self.assertEqual((s['exact_score_comparisons'],s['matched_pairs'],s['profiled_component_calls']),(84,24,336))
        self.assertAlmostEqual(s['paired_profile_over_control_ratio']['median'],1.1)
        self.assertAlmostEqual(sum(v['share'] for v in s['grouped_cost'].values()),1.)
        self.assertEqual(s['by_prefix']['32']['control_seconds']['n'],8)
    def test_incomplete_rejected(self):
        r=fixture();r['status']='running'
        with self.assertRaises(ValueError):summarize(r)
    def test_missing_step_rejected(self):
        r=fixture();r['trials'][1]['steps'].pop()
        with self.assertRaises(ValueError):summarize(r)
    def test_wrong_condition_rejected(self):
        r=fixture();r['trials'][4]['steps'][1]['profiled']=False
        with self.assertRaises(ValueError):summarize(r)
    def test_pair_score_drift_rejected(self):
        r=fixture();r['trials'][4]['steps'][1]['scores_sha256']='changed'
        with self.assertRaises(ValueError):summarize(r)
    def test_wrong_component_order_rejected(self):
        r=fixture();r['trials'][4]['steps'][1]['calls'].reverse()
        with self.assertRaises(ValueError):summarize(r)
    def test_disposal_rejected(self):
        r=fixture();r['trials'][2]['disposed']=False
        with self.assertRaises(ValueError):summarize(r)
    def test_exact_score_failure_rejected(self):
        r=fixture();r['trials'][2]['steps'][2]['scores_exact']=False
        with self.assertRaises(ValueError):summarize(r)

if __name__=='__main__':unittest.main()
