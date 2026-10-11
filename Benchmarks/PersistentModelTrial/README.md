# Persistent model admission trial

This prototype asks whether one loaded Core ML model can serve bounded chunks while CPU and MLX GPU peers make progress under a constrained memory allowance. It builds on [mixed-workload qualification](../MixedWorkloads/README.md).

The experiment compares four strategies at the same total allowance:

| Strategy | Model lifetime | Batches per work admission |
| --- | --- | ---: |
| Reload | One pool per chunk | 8 |
| Whole run | One pool for the run, all capacity held | 256 |
| Persistent, short | One pool for the run, model partition held | 8 |
| Persistent, medium | One pool for the run, model partition held | 32 |

The smoke test uses 32 total batches. The full comparison uses 256 batches for each CPU, GPU, and Core ML lane. It covers 1,024 and 8,192 rows, CPU-only and CPU-and-Neural-Engine Core ML policies, and total allowances of 113, 177, and 241 MiB.

## Ownership and progress contract

The persistent strategy divides its allowance before model loading. A private 49 MiB partition covers the model pool and context. The remaining 64, 128, or 192 MiB forms the shared work queue. Every numeric request and inference window requires 64 MiB.

The private model partition has two known reservations: one for context and one for the pool. Their sum fits exactly, and neither waits for work capacity. All computing callers acquire only the work partition. The constructor rejects a partition that cannot admit the largest request. This structure prevents model owners from exhausting the capacity needed for their own next work request in this single-model experiment.

The model remains charged until all peers finish and the pool closes. No task borrows unused model capacity. Fixed source fixtures remain outside these operation reservations and inside sampled process residency. These estimates do not enforce an allocator or RSS limit.

The retained strategy uses `CoreMLBatchPipeline.withSession` and `CoreMLBatchSession.run`, which reuse the rolling batch scheduler. The existing whole-run API and numerical operations remain unchanged. The historical prototype is preserved on `codex/persistent-model-admission-trial`. This comparison still qualifies one model, not a general multi-model scheduling contract.

## Run the comparison

Build the uninstrumented Release worker using the existing benchmark build recorder. The controller verifies its source, executable, Metal resources, and toolchain.

```sh
python3 Benchmarks/PersistentModelTrial/run.py \
  --worker /absolute/path/SwiftSciBenchmarkWorker \
  --models /absolute/path/coreml-models \
  --raw /absolute/path/raw-covertype.json \
  --output /absolute/path/new-results-directory
```

Add `--smoke` for one small configuration. Reuse the [frozen Covertype artifacts](../BoundedInference/README.md). The controller starts one bounded process at a time with a six-minute timeout. The worker rejects sampled RSS at or above 1 GiB.

Each configuration warms every strategy and allowance once, then records three rotated samples. Timing includes model loading, preprocessing, prediction, result checking, and shutdown. Persistent model setup precedes peer launch. It is included in total elapsed time and peer completion times, but excluded from workspace-acquisition waiting. For the retained variant, admission timing ends at the first loader callback and includes metadata validation. Numeric-peer wait measurements still measure budget acquisition directly. The report shows both waiting and first completion to preserve this distinction.

Run the report controls with:

```sh
python3 -m unittest discover -s Benchmarks/Tests -p 'test_persistent_model_trial.py'
```

## Interpret correctness and cancellation

All strategies must match the isolated references used by the existing mixed-workload benchmark. Every batch is checked. Cancellation tests stop active inference after 16 consumed batches for both chunk sizes. A queued cancellation removes inference behind a full work-budget blocker while the model remains loaded and two numeric peers wait. Peers must finish, the retained charge must survive until pool closure, both partitions must drain, and full capacity must be reacquired.

The experiment rejects an allowance one byte below the minimum before loading. Every normal result checks the sum of partition peaks against the total allowance. These finite runs qualify the declared workload. They do not prove indefinite fairness, exclusive Neural Engine placement, physical memory limits, or leak freedom.
