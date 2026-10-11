"""Check partition and comparison evidence without GPU dependencies."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'PersistentModelTrial/report.py'
spec = importlib.util.spec_from_file_location('persistent_report', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def sample(variant, quantum, mode, iteration):
    extra = {'tight': 0, 'balanced': 64, 'wide': 128, 'cancel-active': 0}[mode]
    lanes = []
    for name in ('cpu', 'gpu', 'coreml'):
        cancelled = mode == 'cancel-active' and name == 'coreml'
        lanes.append(dict(lane=name, cancelled=cancelled, completed=16 if cancelled else 32,
                          admissionSeconds=[.001, .001], serviceSeconds=[.01, .01],
                          completionSeconds=[.011, .022], hashes=['reference'] if name == 'coreml' else []))
    s = dict(mode=mode, iteration=iteration, wallSeconds=.1, budgetBytes=(113+extra)*2**20,
             peakReservedBytes=113*2**20, finalReservedBytes=0, finalQueued=0,
             peakResidentBytes=1024, thermalState=0, lanes=lanes)
    return dict(variant=variant, quantum=quantum, poolScopes=1 if variant == 'persistent' else 32//quantum,
                startupSeconds=.001 if variant == 'persistent' else None,
                retainedBytes=49*2**20 if variant == 'persistent' else 0,
                workBudgetBytes=(64+extra)*2**20 if variant == 'persistent' else s['budgetBytes'], sample=s)


def valid_record():
    samples = [sample(v,q,m,i) for v,q in [('reload',8), ('whole-run',32), ('persistent',8), ('persistent',32)]
               for m in ('tight','balanced','wide') for i in range(3)]
    return dict(samples=samples, cancellation=[sample('persistent', q, 'cancel-active', 0) for q in (8,32)],
                undersizedPartitionRejected=True, queuedCancellation=dict(queuedBeforeCancellation=3,
                queuedAfterCancellation=2, cpuCompleted=32, gpuCompleted=32, finalReservedBytes=0, finalQueued=0))


class PersistentModelReportTests(unittest.TestCase):
    def test_complete_record(self):
        report.validate(valid_record(), 32)

    def test_rejects_broken_evidence(self):
        mutations = [
            lambda r: r.update(undersizedPartitionRejected=False),
            lambda r: r['samples'].pop(),
            lambda r: r['samples'][18].update(poolScopes=4),
            lambda r: r['samples'][18].update(workBudgetBytes=1),
            lambda r: r['samples'][18]['sample'].update(finalReservedBytes=1),
            lambda r: r['samples'][18]['sample'].update(peakResidentBytes=2**30),
            lambda r: r['samples'][0]['sample']['lanes'][0].update(completed=31),
            lambda r: r['samples'][1]['sample']['lanes'][2].update(hashes=['changed']),
            lambda r: r['cancellation'][0]['sample']['lanes'][2].update(cancelled=False),
            lambda r: r['queuedCancellation'].update(cpuCompleted=0),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record=copy.deepcopy(valid_record()); mutate(record)
                with self.assertRaises(ValueError): report.validate(record, 32)

    def test_report_includes_tradeoffs(self):
        text = report.render([(dict(rows=1024, policy='neural', batches=32), valid_record())])
        self.assertIn('cannot lend unused model capacity', text)
        self.assertIn('does not validate a general multi-model scheduler', text)
        self.assertIn('Admission waits exclude startup', text)


if __name__ == '__main__':
    unittest.main()
