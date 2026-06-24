# Pipeline Composition

ml-pipes provides two fundamental ways to compose pipelines — **merge** and
**join** — each available in two syntactic forms depending on where the
composition happens (inside or outside a pipeline definition).

## Concepts

### Join

Joining two pipelines keeps the original pipelines as distinct, self-contained
blocks. The joined pipeline treats each block as an opaque step. Two
consequences follow directly:

- **Isolated context** — each block runs with a fresh `Context`. Internal
  `Store`/`Recall` keys are invisible outside the block, and outer keys are
  invisible inside.
- **Live reference** — the joined pipeline holds a reference to the original
  pipeline object. Mutating the source after composition is reflected at
  runtime — the block evolves with its source.

```
Before

 Pipeline A                Pipeline B
┌──────────────────┐      ┌──────────────────┐
│ Op1 → Op2 → Op3  │  >>  │ Op4 → Op5 → Op6  │
└──────────────────┘      └──────────────────┘

After

 Pipeline (A >> B)
┌───────────────────────────────────────────────┐
│  ┌──────────────────┐   ┌──────────────────┐  │
│  │ Op1 → Op2 → Op3  │ → │ Op4 → Op5 → Op6  │  │
│  └──────────────────┘   └──────────────────┘  │
└───────────────────────────────────────────────┘
```

### Merge

Merging two pipelines produces a single uniform pipeline. The original
pipelines lose their individual identity — there is no boundary between them
at runtime. Two consequences follow directly:

- **Shared context** — operators across the merged boundary share the same
  `Context`, so `Store`/`Recall` keys are visible on both sides.
- **Snapshot** — the source pipeline's operators are copied into the new flat
  list at build time. Mutating the source afterwards has no effect.

```
Before

 Pipeline A                Pipeline B
┌──────────────────┐      ┌──────────────────┐
│ Op1 → Op2 → Op3  │  +   │ Op4 → Op5 → Op6  │
└──────────────────┘      └──────────────────┘

After

 Pipeline (A + B)
┌────────────────────────────────────┐
│ Op1 → Op2 → Op3 → Op4 → Op5 → Op6  │
└────────────────────────────────────┘
```

## API reference

### Inside a pipeline definition

Use these inside the operator list passed to `Pipeline([...])`.

#### Join `embed(p)` / `Embed(p)`

Joins `p` as a self-contained block. Isolated context, live reference.

```python
preprocess = Pipeline([Resize(), Normalize()])
infer      = Pipeline([Infer("model.onnx"), Extract("output0")])

detection = Pipeline([
    Decode(),
    embed(preprocess),
    embed(infer),
    NMS(...),
])

result = detection("image.jpg")
```

#### Merge `inline(p)` / `Inline(p)`

Merges `p`'s operators into the parent list at construction time. After
`__init__` returns, no `Inline` marker exists in `self.operators` — it is
always a plain flat list.

```python
preprocess = Pipeline([Resize(), Normalize()])

detection = Pipeline([
    Decode(),
    inline(preprocess),   # expands to Resize(), Normalize() here
    Infer(...),
])
# detection.operators == [Decode(), Resize(), Normalize(), Infer(...)]

result = detection("image.jpg")
```

Nested `Inline` markers are fully expanded recursively.

---

### Outside a pipeline definition

Use these to compose existing named pipelines outside a list literal.
Each mirrors its inside-definition counterpart.

#### Join `a >> b`

Mirrors `embed(p)`. Returns a **new** pipeline where both `a` and `b` are
isolated blocks held by live reference. Equivalent to
`Pipeline([Embed(a), Embed(b)])`.

Chaining is flat: `a >> b >> c` produces `[Embed(a), Embed(b), Embed(c)]`,
not a nested structure.

```python
decode     = Pipeline([Decode()])
preprocess = Pipeline([Resize(), Normalize()])
infer      = Pipeline([Infer("model.onnx"), Extract("output0"), NMS(...)])

detection = decode >> preprocess >> infer

result = detection("image.jpg")
```

Neither `a` nor `b` is mutated as a result of `>>`. Both are held as live
references — mutating either after composition is reflected at runtime.

#### Merge `a + b`

Mirrors `inline(p)`. Returns a **new** pipeline with `a`'s and `b`'s operators
merged into one flat list with shared context. Equivalent to
`Pipeline([*a.operators, *b.operators])`.

```python
resize_stage   = Pipeline([Resize(), Store("transform", index=1), Pick(0)])
project_stage  = Pipeline([Recall("transform"), ProjectBoxes()])

detection = resize_stage + infer_stage + project_stage

result = detection("image.jpg")
```

Neither `a` nor `b` is mutated.

#### Merge in place `pipeline.extend([Op1, Op2, ...])`

No inside-definition equivalent. Mutates `pipeline` by merging operators
directly into its existing list and returns `self`. Use this when you already
hold the pipeline handle and want to keep it. `Inline` markers inside the list
are expanded at this point.

```python
pipeline = Pipeline([Decode(), Resize()])
pipeline.extend([Normalize(), Infer("model.onnx")])
pipeline.extend([NMS(...), ToDetections()])

result = pipeline("image.jpg")
```

## Summary table

| Syntax                    | Operation      | Where                    | Result object          |
|---------------------------|----------------|--------------------------|------------------------|
| `embed(p)` / `Embed(p)`   | join           | inside `Pipeline([...])` | outer pipeline         |
| `inline(p)` / `Inline(p)` | merge          | inside `Pipeline([...])` | outer pipeline         |
| `a >> b`                  | join           | outside definition       | new pipeline           |
| `a + b`                   | merge          | outside definition       | new pipeline           |
| `p.extend([...])`         | in-place merge | outside definition       | same pipeline (`self`) |

## Use-cases

### Join

**Reusing shared preprocessing across two independent models** — each model
pipeline stores intermediate values under its own keys. Because join gives
each block a fresh context, there is no risk of one model's keys leaking into
the other even if they happen to use the same name.

```python
# Both detector and classifier cache intermediate tensors under "features".
# Joining keeps those caches isolated — neither block sees the other's value.
detector   = Pipeline([Backbone(), Store("features"), DetectionHead()])
classifier = Pipeline([Backbone(), Store("features"), ClassificationHead()])

inference = detector >> classifier
result = inference(frame)
```

**Registering a new model variant at deploy time** — a team ships the base
pipeline and adds the new model head after the fact. Because both blocks are
live references, the composed pipeline picks up the change without being
rebuilt.

```python
preprocess = Pipeline([Decode(), Resize(640), Normalize()])
detector   = Pipeline([Infer("yolo_v8.onnx"), Extract("output0")])

detection_pipeline = preprocess >> detector

# Later, the model is swapped for a quantised variant and NMS is added — no
# need to touch detection_pipeline.
detector.extend([NMS(iou_threshold=0.5)])

detection_pipeline.validate()  # confirm the contract still holds before running ✅
result = detection_pipeline("frame.jpg")
```

**Wrong mutation — output type changes silently break downstream blocks** —
because blocks are live references, extending a pipeline with an operator that
changes its output type will corrupt the contract for everything that follows.
`validate()` catches this before it reaches production.

```python
preprocess  = Pipeline([Decode(), Resize(640), Normalize()])  # -> np.ndarray
detector    = Pipeline([Infer("yolo.onnx"), Extract("output0")])  # -> Detections
postprocess = Pipeline([NMS(), ScaleBoxes(), ToJSON()])  # expects Detections

detection_pipeline = preprocess >> detector >> postprocess

# A developer serialises detections inside the detector block for debugging,
# changing its output type from Detections to dict.
detector.extend([SerializeToDict()])  # output is now dict, not Detections ❌

detection_pipeline.validate()  # raises PipelineValidationError: contract mismatch
```

### Merge

**Projecting bounding boxes back to the original image space** — the resize
transform is computed once during preprocessing and needs to be available
several steps later to map model output coordinates back to the source frame.
Merging keeps a single shared context across all three stages.

```python
# Resize stores the scale/offset transform so ProjectBoxes can undo it later.
preprocess  = Pipeline([Decode(), Resize(640), Store("transform", index=1), Pick(0), Normalize()])
infer       = Pipeline([Infer("yolo.onnx"), Extract("output0"), NMS()])
postprocess = Pipeline([Recall("transform"), ProjectBoxes(), DrawAnnotations()])

detection_pipeline = preprocess + infer + postprocess
# "transform" written in preprocess is visible to postprocess ✅

result = detection_pipeline("frame.jpg")
```

## Validation

All composition forms are compatible with `validate()`.

- `inline` and `+` expand to flat operator lists — the validator sees a single
  unbroken chain with no special handling needed.
- `embed` and `>>` produce an `Embed` operator that validates the boundary type
  against the inner pipeline's contract during the outer pipeline's validation.

Because `>>` and `embed` hold live references, mutating a joined pipeline
after composition changes the contract of the composed pipeline. Always call
`validate()` after any such mutation and before the next execution.

```python
# Type contract is validated end-to-end across all composition forms
detection = decode_pipeline >> preprocess_pipeline + infer_pipeline
detection.validate()
```
