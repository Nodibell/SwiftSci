"""Reject incomplete or incorrectly labelled contention evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('mixed_report', Path(__file__).resolve().parents[1] / 'MixedWorkloads/report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def valid_record():
    samples = []
    modes = [(mode, i) for mode in ('isolated', 'tight', 'balanced', 'wide') for i in range(3)]
    modes.append(('cancel-active', 0))
    for mode, iteration in modes:
        lanes = []
        for name in ('cpu', 'gpu', 'coreml'):
            cancelled = mode == 'cancel-active' and name == 'coreml'
            lanes.append(dict(lane=name, completed=16 if cancelled else 32, cancelled=cancelled,
                              admissionSeconds=[.001], serviceSeconds=[.01], completionSeconds=[.011],
                              hashes=['reference'] if name == 'coreml' else []))
        samples.append(dict(mode=mode, iteration=iteration, finalReservedBytes=0, finalQueued=0,
                            peakReservedBytes=100, budgetBytes=200, peakResidentBytes=1024,
                            wallSeconds=.1, lanes=lanes, peakQueued=0, thermalState=0))
    return dict(samples=samples, queuedCancellation=dict(queuedBeforeCancellation=3,
                queuedAfterCancellation=2, cpuCompleted=32, gpuCompleted=32, finalReservedBytes=0, finalQueued=0))


class MixedReportTests(unittest.TestCase):
    def test_accepts_complete_record(self):
        report.validate(valid_record(), 32)

    def test_rejects_bad_evidence(self):
        mutations = [
            lambda r: r['samples'].pop(),
            lambda r: r['samples'][0].update(finalReservedBytes=1),
            lambda r: r['samples'][0].update(peakReservedBytes=201),
            lambda r: r['samples'][0].update(peakResidentBytes=2**30),
            lambda r: r['samples'][0].update(wallSeconds=float('nan')),
            lambda r: r['samples'][0]['lanes'][0].update(completed=31),
            lambda r: r['samples'][1]['lanes'][2].update(hashes=['different']),
            lambda r: r['samples'][-1]['lanes'][2].update(cancelled=False),
            lambda r: r['queuedCancellation'].update(queuedAfterCancellation=3),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                record = copy.deepcopy(valid_record())
                mutate(record)
                with self.assertRaises(AssertionError):
                    report.validate(record, 32)

    def test_report_keeps_memory_and_device_limits(self):
        q = dict(rows=1024, policy='neural', quantum=8)
        text = report.render([(q, valid_record()['samples'])])
        self.assertIn('does not prove Neural Engine placement', text)
        self.assertIn('not a hard physical-memory cap', text)
        self.assertIn('FIFO prevents overtaking', text)
        self.assertIn('1.000', text)


if __name__ == '__main__':
    unittest.main()
