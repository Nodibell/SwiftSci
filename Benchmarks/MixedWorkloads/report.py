"""Validate bounded mixed-workload results and report matching work totals."""
import math
import statistics


def validate(record, batches):
    samples = record['samples']
    queued = record['queuedCancellation']
    assert queued['queuedBeforeCancellation'] == 3
    assert queued['queuedAfterCancellation'] == 2
    assert queued['cpuCompleted'] == queued['gpuCompleted'] == batches
    assert queued['finalReservedBytes'] == queued['finalQueued'] == 0
    expected = {(mode, i) for mode in ('isolated', 'tight', 'balanced', 'wide') for i in range(3)}
    expected.add(('cancel-active', 0))
    assert len(samples) == len(expected)
    assert {(s['mode'], s['iteration']) for s in samples} == expected
    reference = None
    for s in samples:
        assert s['finalReservedBytes'] == s['finalQueued'] == 0
        assert 0 < s['peakReservedBytes'] <= s['budgetBytes']
        assert 0 < s['peakResidentBytes'] < 2**30
        assert math.isfinite(s['wallSeconds']) and s['wallSeconds'] > 0
        assert sorted(lane['lane'] for lane in s['lanes']) == ['coreml', 'cpu', 'gpu']
        for lane in s['lanes']:
            cancelled = s['mode'] == 'cancel-active' and lane['lane'] == 'coreml'
            assert lane['cancelled'] == cancelled
            assert lane['completed'] == (16 if cancelled else batches)
            for key in ('admissionSeconds', 'serviceSeconds', 'completionSeconds'):
                assert all(math.isfinite(x) and x >= 0 for x in lane[key])
            if not cancelled:
                assert len(lane['admissionSeconds']) == len(lane['serviceSeconds']) == len(lane['completionSeconds']) > 0
                assert lane['completionSeconds'] == sorted(lane['completionSeconds'])
            if lane['lane'] == 'coreml' and not cancelled:
                assert lane['hashes']
                if reference is None:
                    reference = lane['hashes']
                assert lane['hashes'] == reference


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def render(records):
    lines = ['# Mixed-workload qualification', '',
             'The same finite work set runs in isolation and with three competing lanes: CPU imputation and scaling, MLX GPU linear regression, and Core ML inference. Every output is checked against its isolated reference. Medians use three interleaved samples after an isolated warmup.', '',
             'Core ML admissions contain either eight batches or the complete run. CPU and GPU admissions contain one batch each. Model loading occurs inside every Core ML admission. Fitting, fixed source preparation, model compilation, and reference checks are outside timing.', '',
             '| Rows | Core ML policy | Batches per admission | Mode | Work-set ms | Speedup vs isolated | Peak RSS MiB | Peak queued |',
             '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |']
    detail = ['', '## Admission and service', '',
              'Wait is time inside shared-budget acquisition. Service includes computation and result checking after acquisition. Completion gaps are measured between completed admissions, so Core ML gaps describe chunks, not individual predictions.', '',
              '| Rows | Policy | Chunk | Mode | Lane | p95 wait ms | Max wait ms | Service ratio vs isolated | Max completion gap ms |',
              '| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for q, samples in records:
        base = [s for s in samples if s['mode'] == 'isolated']
        base_wall = statistics.median(s['wallSeconds'] for s in base)
        for mode in ('isolated', 'tight', 'balanced', 'wide'):
            group = [s for s in samples if s['mode'] == mode]
            wall = statistics.median(s['wallSeconds'] for s in group)
            rss = max(s['peakResidentBytes'] for s in group) / 2**20
            queued = max(s['peakQueued'] for s in group)
            prefix = f"{q['rows']} | {q['policy']} | {q['quantum']} | {mode}"
            lines.append(f'| {prefix} | {wall*1000:.2f} | {base_wall/wall:.3f} | {rss:.2f} | {queued} |')
            for name in ('cpu', 'gpu', 'coreml'):
                lanes = [next(l for l in s['lanes'] if l['lane'] == name) for s in group]
                baseline = [next(l for l in s['lanes'] if l['lane'] == name) for s in base]
                waits = [x for lane in lanes for x in lane['admissionSeconds']]
                service = statistics.median(sum(l['serviceSeconds']) for l in lanes)
                reference = statistics.median(sum(l['serviceSeconds']) for l in baseline)
                gaps = [b-a for lane in lanes for a,b in zip([0]+lane['completionSeconds'], lane['completionSeconds'])]
                detail.append(f'| {prefix} | {name} | {percentile(waits,.95)*1000:.3f} | {max(waits)*1000:.3f} | {service/reference:.3f} | {max(gaps)*1000:.3f} |')
    thermals = sorted({s['thermalState'] for _, samples in records for s in samples})
    lines += detail + ['', f'All {len(records)} active Core ML cancellation trials stopped after 16 consumed batches while peer lanes completed. Queued cancellation also removed the blocked inference request while both numeric peers subsequently completed. All scopes drained and reacquired their full budget. Thermal states: {thermals}.', '',
        'Budgets are conservative admission estimates. Tight permits one lane at a time; balanced permits some overlap; wide fits all three. FIFO prevents overtaking but does not promise equal latency or preemption. These finite runs test progress, not unlimited-time starvation freedom.', '',
        'Inputs and fitted state are prepared before timing and retained outside the participating-operation budget. Their memory remains included in process RSS. Model and allocator caches persist across samples. RSS is sampled at 10 ms and completion boundaries, not a hard physical-memory cap.', '',
        'The Core ML neural policy permits CPU fallback and does not prove Neural Engine placement. GPU regression uses its explicit GPU route and returns materialized values before releasing admission. No production scheduler or numerical tolerance changed.']
    return '\n'.join(lines)+'\n'
