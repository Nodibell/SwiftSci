# Bounded vision preprocessing conformance

Eight original image fixtures exercise `YOLOPreprocessor.preprocess` through `ImageDataset`. Swift requests MLX CPU explicitly. There are no models, weights, downloaded images, image decoders or CoreGraphics operations.

The cases cover CHW RGB to NHWC conversion, grayscale replication, spatial row and column order, odd image dimensions, asymmetric target dimensions, default and custom padding, odd leftover padding, positive half rounding, and normalized Float32 output. Four images have spatially varying pixels and retain their original dimensions during preprocessing. Four resize constant channel planes, which have an analytic answer independent of the interpolation kernel.

All input and target dimensions are between 1 and 32. Images have one or three channels. Pixels must be finite, normalized and exactly representable in Float32. The adapter rejects malformed buffers, unsupported layouts, unknown fields and nonconstant images that require resizing. These are benchmark boundary checks. The production API itself accepts a wider and less strictly validated domain.

The scalar oracle computes dimensions with exact rational arithmetic. It rounds positive half ties upward, places floor-half of leftover padding before the image, and leaves the extra pixel after the image. The output contains its four observed dimensions followed by every channel of every output pixel. Every value participates in validation. The NumPy comparator independently constructs the same bounded image transformation on CPU. It does not implement or measure a general resampler.

Absolute and relative tolerances are both 2e-6. Expected image values use exact Float32 conversion, including the default padding of 114/255. The small nonzero budget allows Float32 filter arithmetic when Accelerate resamples constant planes. There is no color-space, image-codec or CoreGraphics tolerance. This is a bounded numerical budget, not a promise about arbitrary Lanczos images. Negative controls change each output value by .001 and require rejection.

The source lock pins the generator, scalar oracle, input validator and NumPy comparator. Each dataset pins the lock, input and complete reference. Regenerate with `Benchmarks/.venv-standardized/bin/python Benchmarks/Fixtures/vision/generate.py`.

Swift timing includes image and preprocessor construction, preprocessing, output evaluation, CPU stream synchronization and full extraction. Input decoding stays outside timing. The NumPy comparator uses constant broadcasting for resized inputs, so these timings are diagnostics rather than matched resampler performance comparisons.

## Source review and limits

The production docstring calls resizing bilinear, but the implementation requests `kvImageHighQualityResampling`. Apple documents that flag as [Lanczos-5](https://developer.apple.com/documentation/accelerate/kvimagehighqualityresampling). This suite certifies resize dimensions and constant preservation. It does not certify arbitrary Lanczos interpolation, boundary extension, antialiasing or fidelity to the bilinear docstring.

The production docstring also says the preprocessor normalizes output, but its code copies input values into Float32 planes without division by 255 or clipping. This contract supplies already normalized input. Passing it cannot establish raw-byte normalization. A separate contract would need a clear input-range requirement before treating that behavior as a numerical failure.

The adapter rejects malformed inputs before calling production. Passing these tests does not establish that production safely rejects negative target dimensions, empty images, truncated buffers or unsupported channel counts. The suite does not test detection, segmentation, model quality, NMS, color management, alpha, GPU execution or training.
