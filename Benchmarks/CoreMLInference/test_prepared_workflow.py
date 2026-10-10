import unittest
from prepared_workflow import schedule, validate, report


def record():
    cases = [dict(mode=m, reuse=4) for m in ['direct', 'retained']]
    samples = [dict(mode=c['mode'], reuse=4, preparationSeconds=.001, predictionSeconds=.003,
                    totalSeconds=.004, releaseSeconds=.0001, residentBytes=1_000_000,
                    inputReservedBytes=65536 if c['mode']=='retained' else 0) for c in cases]
    return dict(request=dict(rows=1024, policy='cpu', callers=1, slots=1, schedule=cases),
                poolReleased=True, finalPoolReservedBytes=0, finalInputReservedBytes=0,
                transportElementBytes=2, outputHash='a'*64, samples=samples)


class PreparedWorkflowTests(unittest.TestCase):
    def test_schedule_has_balanced_adjacent_pairs_and_is_reproducible(self):
        cases = schedule(5, 17)
        self.assertEqual(cases, schedule(5, 17))
        self.assertNotEqual(cases, schedule(5, 18))
        for a, b in zip(cases[::2], cases[1::2]):
            self.assertEqual(a['reuse'], b['reuse'])
            self.assertEqual({a['mode'], b['mode']}, {'direct', 'retained'})
        for reuse in [1, 2, 4, 16, 64]:
            self.assertEqual(sum(c['reuse']==reuse for c in cases), 10)

    def test_valid_and_report(self):
        r = record()
        validate(r)
        text = report([r])
        self.assertIn('| 1024 | cpu | 1/1 | 4 | 4.000 | 4.000 | 1.00x |', text)
        self.assertIn('0.062', text)
        self.assertIn('not physical-memory savings', text)

    def test_rejects_incomplete_runs_and_unreleased_owners(self):
        for key, value in [('poolReleased', False), ('finalInputReservedBytes', 1), ('finalPoolReservedBytes', 1)]:
            r = record(); r[key] = value
            with self.assertRaises(ValueError): validate(r)
        r = record(); r['samples'].pop()
        with self.assertRaises(ValueError): validate(r)
        r = record(); r['samples'].reverse()
        with self.assertRaises(ValueError): validate(r)

    def test_rejects_excluded_preparation_and_invalid_measurements(self):
        for key, value in [('totalSeconds', .003), ('residentBytes', 1_073_741_824),
                           ('inputReservedBytes', -1), ('predictionSeconds', 0), ('releaseSeconds', -1)]:
            r = record(); r['samples'][0][key] = value
            with self.assertRaises(ValueError): validate(r)
        for value in [float('nan'), float('inf')]:
            r = record(); r['samples'][0]['totalSeconds'] = value
            with self.assertRaises(ValueError): validate(r)
        r = record(); r['samples'][1]['inputReservedBytes'] = 0
        with self.assertRaises(ValueError): validate(r)


if __name__ == '__main__':
    unittest.main()
