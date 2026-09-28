# ml-pipes

Build explicit ML pipelines you can validate, run, inspect, trace, and
benchmark.

`ml-pipes` treats the pipeline as the primary artifact: an ordered sequence
of small operators with clear input and output boundaries. That makes the
whole data path—from loading and preprocessing through inference,
postprocessing, and delivery—visible, composable, and testable.

## Get Started

`ml-pipes` requires Python 3.10 or later. Install the umbrella package for the
core framework, then choose the optional package profiles your pipeline needs:

```bash
pip install ml-pipes
pip install 'ml-pipes[onnx,vision]'
```

See [Packages](PACKAGES.md) for the package matrix, install profiles, and
public import surfaces. The repository’s [examples](https://github.com/trained-by-humans/ml-pipes/tree/main/examples)
provide runnable starting points.

## Explore the Framework

- [Design](DESIGN.md) explains the composition-first model behind
  `ml-pipes`.
- [Architecture](ARCHITECTURE.md) maps that model to the framework’s runtime
  and ownership boundaries.
- [Build Pipelines](OPERATORS.md) covers operators, composition, execution
  regions, and model scaffolding.
- [Tooling](VALIDATION.md) covers validation, inspection, tracing,
  benchmarking, and performance guidance.
