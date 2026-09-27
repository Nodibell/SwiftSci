import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Tools'))
from sweeps import metrics

class SweepMetrics(unittest.TestCase):
    def test_memory_metrics_are_distinct_and_dtype_aware(self):
        p=dict(rows=128,columns=8,dtype='float32',device='gpu',stage='prepared')
        s=dict(case_id='case',engine='pandas',median_ns=1000,status='passed',timing_resolved=True,peak_rss_bytes=12345678)
        r=metrics(p,s)
        self.assertEqual(r['source_feature_payload_bytes'],8192)
        self.assertEqual(r['selected_tensor_payload_bytes'],3072)
        self.assertEqual(r['process_lifetime_peak_rss_bytes'],12345678)
        self.assertEqual(r['comparator_device'],'cpu')
        self.assertEqual(metrics(dict(p,dtype='float64'),s)['selected_tensor_payload_bytes'],6144)
    def test_failed_or_unresolved_duration_has_no_throughput(self):
        p=dict(rows=128,columns=8,dtype='float32',device='cpu',stage='pipeline')
        for status,resolved in [('failed',True),('passed',False)]:
            s=dict(case_id='case',engine='swiftsci',median_ns=10,status=status,timing_resolved=resolved,peak_rss_bytes=None)
            r=metrics(p,s);self.assertIsNone(r['median_ns']);self.assertIsNone(r['selected_elements_per_second'])
