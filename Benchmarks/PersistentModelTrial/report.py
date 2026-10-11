"""Validate persistent-model evidence without treating admitted bytes as process memory."""
import math
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(record, batches):
    require(record['undersizedPartitionRejected'] is True, 'Undersized partition accepted')
    samples = record['samples']
    variants = [('reload', 8), ('whole-run', batches), ('persistent', 8), ('persistent', 32)]
    expected = {(variant, quantum, mode, i) for variant, quantum in variants
                for mode in ('tight', 'balanced', 'wide') for i in range(3)}
    actual = {(s['variant'], s['quantum'], s['sample']['mode'], s['sample']['iteration']) for s in samples}
    require(len(samples) == len(expected) and actual == expected, 'Incomplete comparison matrix')
    require(len(record['cancellation']) == 2, 'Missing active cancellations')
    require({s['quantum'] for s in record['cancellation']} == {8, 32}, 'Missing cancellation quantum')
    reference = None
    for outer in samples + record['cancellation']:
        s = outer['sample']
        require(s['finalReservedBytes'] == s['finalQueued'] == 0, 'Undrained reservation')
        require(0 < s['peakReservedBytes'] <= s['budgetBytes'], 'Reservation exceeds limit')
        require(0 < s['peakResidentBytes'] < 2**30, 'RSS cutoff exceeded')
        require(math.isfinite(s['wallSeconds']) and s['wallSeconds'] > 0, 'Invalid wall time')
        require(sorted(l['lane'] for l in s['lanes']) == ['coreml', 'cpu', 'gpu'], 'Missing lane')
        if outer['variant'] == 'persistent':
            require(outer['poolScopes'] == 1, 'Persistent model reloaded')
            require(outer['retainedBytes'] == 49 * 2**20, 'Unexpected retained partition')
            require(outer['workBudgetBytes'] + outer['retainedBytes'] == s['budgetBytes'], 'Partition sum differs')
            require(outer['workBudgetBytes'] >= 64 * 2**20, 'Work cannot fit')
            require(0 <= outer['startupSeconds'] <= s['wallSeconds'], 'Invalid startup time')
        else:
            require(outer['poolScopes'] == batches // outer['quantum'], 'Reference load count differs')
        for lane in s['lanes']:
            cancelled = s['mode'] == 'cancel-active' and lane['lane'] == 'coreml'
            require(lane['cancelled'] == cancelled, 'Cancellation label differs')
            require(lane['completed'] == (16 if cancelled else batches), 'Missing work')
            for key in ('admissionSeconds', 'serviceSeconds', 'completionSeconds'):
                require(all(math.isfinite(x) and x >= 0 for x in lane[key]), 'Invalid duration')
            if not cancelled:
                require(len(lane['admissionSeconds']) == len(lane['serviceSeconds']) == len(lane['completionSeconds']) > 0,
                        'Incomplete timing')
                require(lane['completionSeconds'] == sorted(lane['completionSeconds']), 'Unordered completion')
            if lane['lane'] == 'coreml' and not cancelled:
                require(bool(lane['hashes']), 'Missing prediction hashes')
                if reference is None:
                    reference = lane['hashes']
                require(lane['hashes'] == reference, 'Predictions changed')
    q = record['queuedCancellation']
    require(q['queuedBeforeCancellation'] == 3 and q['queuedAfterCancellation'] == 2, 'Queue cancellation failed')
    require(q['cpuCompleted'] == q['gpuCompleted'] == batches, 'Peers did not recover')
    require(q['finalReservedBytes'] == q['finalQueued'] == 0, 'Queue did not drain')


def render(records):
    lines = ['# Persistent model admission trial', '',
        'One scoped Core ML model is retained while each chunk reacquires shared workspace. A fixed model partition and a work partition sum to the same total allowance as the references. The smallest work partition fits every request. No task acquires the model partition while holding work capacity.', '',
        'Each configuration has three rotated samples after one warmup of every variant. Timing includes model setup, preprocessing, prediction, verification, and cleanup. Persistent startup precedes peer launch and is included in total time and peer completion times. Admission waits exclude startup.', '',
        '| Rows | Policy | Budget | Variant | Chunk | Work-set ms | Speedup vs reload-8 | Speedup vs whole run | Max peer wait ms | Max peer first completion ms | Max peer gap ms | RSS MiB |',
        '| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for request, record in records:
        normal = record['samples']
        for mode in ('tight', 'balanced', 'wide'):
            subset = [s for s in normal if s['sample']['mode'] == mode]
            times = {(variant, quantum): statistics.median(s['sample']['wallSeconds'] for s in subset
                     if (s['variant'], s['quantum']) == (variant, quantum))
                     for variant, quantum in {(s['variant'], s['quantum']) for s in subset}}
            for variant, quantum in [('reload', 8), ('whole-run', request['batches']), ('persistent', 8), ('persistent', 32)]:
                group = [s['sample'] for s in subset if (s['variant'], s['quantum']) == (variant, quantum)]
                wall = times[variant, quantum]
                peers = [lane for sample in group for lane in sample['lanes'] if lane['lane'] != 'coreml']
                wait = max(x for lane in peers for x in lane['admissionSeconds'])
                first = max(lane['completionSeconds'][0] for lane in peers)
                gaps = [b-a for lane in peers for a,b in zip(lane['completionSeconds'], lane['completionSeconds'][1:])]
                rss = max(s['peakResidentBytes'] for s in group) / 2**20
                lines.append(f"| {request['rows']} | {request['policy']} | {mode} | {variant} | {quantum} | {wall*1000:.2f} | {times['reload',8]/wall:.3f} | {times['whole-run',request['batches']]/wall:.3f} | {wait*1000:.2f} | {first*1000:.2f} | {max(gaps)*1000:.2f} | {rss:.2f} |")
    thermals = sorted({s['sample']['thermalState'] for _, r in records for s in r['samples'] + r['cancellation']})
    lines += ['', f'All {len(records)*2} active cancellation checks and {len(records)} queued cancellation checks passed. Retained model capacity remained charged until peers completed and the pool closed. Every partition drained and reacquired full capacity. Undersized partitions were rejected before loading. Thermal states: {thermals}.', '',
        'The model partition is 49 MiB. Work partitions are 64, 128, and 192 MiB; combined totals are 113, 177, and 241 MiB. Fixed fixtures are outside these reservations and inside process RSS. The persistent variant cannot lend unused model capacity to CPU/GPU work, even after its inference lane finishes. This is an intentional fixed-partition tradeoff.', '',
        'This single-model comparison uses scoped retained sessions and the existing rolling batch scheduler. It does not validate a general multi-model scheduler. Retained inference wait includes metadata validation before the first load. Logical pool scopes do not measure native framework cache misses. Neural policy permits CPU fallback. Results do not prove exclusive placement, physical memory limits, leak freedom, or starvation freedom for unbounded arrivals.']
    return '\n'.join(lines) + '\n'
