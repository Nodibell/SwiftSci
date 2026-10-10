"""Evidence validation for the bounded concurrency controller."""
import copy
import unittest

from concurrency import percentile_upper, report, validate
from pool_trial import report as pool_report


def sample():
    request = dict(fixture='fixture.json', rows=8192, policy='cpu', callers=1, instances=1,
                   admissionSlots=1, seconds=1, rssLimitBytes=1_073_741_824)
    buckets = [0] * 256
    buckets[100] = 1
    latency = dict(count=1, totalSeconds=.002, minimumSeconds=.002,
                   maximumSeconds=.002, buckets=buckets)
    record = dict(request=request, executionTraceVerified=False, predictorsReleased=True,
                  retainedOutputsVerified=True, finalReservedBytes=0, finalQueuedCount=0,
                  cancelledRequestsVerified=2, requestReservationBytes=100, budgetPeakBytes=100,
                  budgetLimitBytes=100, outputHashes=['a' * 64, 'b' * 64], elapsedSeconds=1.01,
                  callers=[dict(caller=0, latency=latency)],
                  memory=[dict(residentBytes=1000, reservedBytes=100)])
    return record, request


class ConcurrencyEvidenceTests(unittest.TestCase):
    def test_complete_record(self):
        record, request = sample()
        validate(record, request)

    def test_persistent_scope_holds_and_releases_whole_quota(self):
        record, request = sample()
        request.update(execution='persistent', retainedBytes=33_554_432)
        record['cancelledRequestsVerified'] = 0
        limit = request['retainedBytes'] + record['requestReservationBytes']
        record.update(budgetLimitBytes=limit, budgetPeakBytes=limit)
        record['memory'][0]['reservedBytes'] = limit
        validate(record, request)
        record['memory'][0]['reservedBytes'] = limit - 1
        with self.assertRaises(ValueError):
            validate(record, request)
        record['memory'][0]['reservedBytes'] = limit
        record['finalReservedBytes'] = limit
        with self.assertRaises(ValueError):
            validate(record, request)

    def test_rejects_incomplete_ownership_and_admission(self):
        for key, value in [('predictorsReleased', False), ('retainedOutputsVerified', False),
                           ('finalReservedBytes', 1), ('finalQueuedCount', 1),
                           ('cancelledRequestsVerified', 1), ('budgetPeakBytes', 101),
                           ('executionTraceVerified', True), ('elapsedSeconds', .5)]:
            with self.subTest(key=key):
                record, request = sample()
                record[key] = value
                with self.assertRaises(ValueError):
                    validate(record, request)

    def test_rejects_missing_or_malformed_latency(self):
        record, request = sample()
        record['callers'][0]['latency']['buckets'][0] = 1
        with self.assertRaises(ValueError):
            validate(record, request)
        record, request = sample()
        record['callers'][0]['latency']['totalSeconds'] = float('nan')
        with self.assertRaises(ValueError):
            validate(record, request)

    def test_rejects_excess_rss_and_unbounded_telemetry(self):
        record, request = sample()
        record['memory'][0]['residentBytes'] = request['rssLimitBytes'] + 1
        with self.assertRaises(ValueError):
            validate(record, request)
        record, request = sample()
        record['memory'] *= 4
        with self.assertRaises(ValueError):
            validate(record, request)

    def test_combines_histograms_and_reports_overflow(self):
        record, _ = sample()
        callers = record['callers']
        self.assertAlmostEqual(percentile_upper(callers, .95), 1e-6 * 1.08 ** 100)
        other = copy.deepcopy(callers[0])
        other['latency']['buckets'] = [0] * 255 + [1]
        self.assertIsNone(percentile_upper(callers + [other], .95))

    def test_pool_report_uses_declared_transport_precision(self):
        record, request = sample()
        request.update(execution='synchronous', modelPackage='program.mlpackage')
        record['transportElementBytes'] = 2
        self.assertIn('Float16 I/O', pool_report([record]))
        self.assertNotIn('Float32 I/O', pool_report([record]))
        record['transportElementBytes'] = 4
        self.assertIn('Float32 I/O', pool_report([record]))
        del record['transportElementBytes']
        self.assertIn('Float32 I/O', pool_report([record]))

    def test_report_preserves_measurement_limits(self):
        record, _ = sample()
        text = report([record])
        self.assertIn('not separately instrumented', text)
        self.assertIn('not numerical certification', text)
        self.assertIn('not allocation traces', text)


if __name__ == '__main__':
    unittest.main()
