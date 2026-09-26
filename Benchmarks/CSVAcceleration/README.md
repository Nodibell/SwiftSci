# CSV CPU and Metal experiment

This optional benchmark tests byte classification and field-index construction. It does not change the production parser or measure complete CSV reading. It requires an Apple silicon Mac with Xcode and Metal. No package dependency is added.

```sh
Benchmarks/CSVAcceleration/run.sh /path/to/input.csv
```

The script builds in the temporary directory, runs correctness checks, and prints raw JSON timings for each supplied file. It reads the parser snapshot at commit `dc31b5f1afc7f92810cfd472f7269f0d05533438` from local Git history. Fetch that history first if using a shallow clone. The snapshot is a historical baseline, not the latest CPU implementation.

## Measurements

M4 Max, 128 GiB, macOS 27.0, Swift 6.4, native ARM64 Release. Median milliseconds from nine rotating repetitions after two warmups per method. The 1M-row input was the production comparison fixture, 16,521,578 bytes. The 4M-row input repeated its data section four times with one header, 66,086,267 bytes.

| Method | 1M rows | 4M rows |
| --- | ---: | ---: |
| Serial NEON classification only | 1.861 | 7.425 |
| Resident GPU classification only | 0.756 | 1.375 |
| Historical scalar parser, full index | 21.128 | 84.325 |
| Scalar bitmask classifier, full index | 22.762 | 88.272 |
| Serial NEON, full index | 7.949 | 31.647 |
| Parallel NEON, full index | 6.485 | 25.427 |
| GPU with copied input, full index | 8.723 | 39.085 |
| GPU with mapped input, full index | 7.671 | 30.824 |

GPU index creation was 18% slower than parallel NEON at 1M rows and 21% slower at 4M rows. GPU classification was faster than the serial NEON classifier, but buffer creation, synchronization, and CPU output construction removed that advantage. Classification-only timings reuse buffers and are lower bounds. They do not establish a GPU advantage over parallel CPU classification.

Keep GPU execution opt-in until a complete CSV benchmark beats the current production CPU path. These results do not justify a default GPU parser.

## Algorithm and correctness

Each GPU SIMD group classifies 32 bytes into comma, LF, quote, and CR bitmasks. The CPU creates the existing field offsets and row boundaries. The ARM NEON classifier emits the same masks. The parallel CPU variant splits classification into 1 MiB chunks. Any quote falls back to the original scalar parser, preserving quoted newlines and escaped quotes. This prototype supports the default comma and quote characters only.

The checks compare every field offset, length, escaped-quote flag, and row boundary. Eight edge cases and 100 randomized inputs pass, including empty input, CRLF, quoted newlines, escaped quotes, unfinished quotes, missing final newline, and trailing delimiter. Every method also agrees on both full benchmark fixtures. The existing parser's behavior is the oracle, including its known EOF behavior; this does not prove RFC compliance.

## What timing includes

All inputs use a warm filesystem cache. Full-index methods include output allocation and destruction. GPU methods also include Metal buffer creation, command submission, and completion wait. The copied method includes copying the input. File mapping, Metal pipeline compilation, and fixture validation occur outside timing. Column inference, numeric conversion, and DataFrame construction are not measured.

The mapped method requires page-aligned memory from a file mapping and rounds the exposed buffer length to the VM page size. The shader bounds every read by the actual file length. The mapped `Data` remains alive inside its borrowed pointer scope until GPU completion. Unsupported input alignment fails the prototype explicitly; a production implementation would fall back. Apple documents the lifetime and alignment requirements of [wrapping existing memory](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)). [Shared storage](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) still requires producer completion before the other processor accesses the data.

The installed Metal compiler rejects `double`. The check prints that diagnostic. Float64 parsing remains on the CPU; replacing it with Float32 would change scientific data semantics. A future GPU decimal converter would need a separate accuracy proof.
