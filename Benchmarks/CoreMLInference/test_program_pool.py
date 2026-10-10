"""Do not mistake requested output backings for observed reuse."""
import unittest
from program_pool import validate_backing_counts


class ProgramPoolEvidenceTests(unittest.TestCase):
    def sample(self):
        return dict(request=dict(execution='persistent'), callers=[dict(latency=dict(count=10))],
                    poolPredictions=12, poolOutputBackingIdentityMatches=12)

    def test_adoption_and_fallback_are_both_valid_observations(self):
        record = self.sample()
        validate_backing_counts(record)
        record['poolOutputBackingIdentityMatches'] = 0
        validate_backing_counts(record)

    def test_rejects_missing_or_impossible_observations(self):
        for key, value in [('poolPredictions', 11), ('poolPredictions', None),
                           ('poolOutputBackingIdentityMatches', 13), ('poolOutputBackingIdentityMatches', -1)]:
            record = self.sample()
            record[key] = value
            with self.assertRaises(ValueError):
                validate_backing_counts(record)

    def test_fresh_predictions_cannot_claim_pool_reuse(self):
        record = self.sample()
        record['request']['execution'] = 'asynchronous'
        with self.assertRaises(ValueError):
            validate_backing_counts(record)
        del record['poolPredictions'], record['poolOutputBackingIdentityMatches']
        validate_backing_counts(record)
