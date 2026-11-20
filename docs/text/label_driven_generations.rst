# Label-Driven Classifier Generations (tsfresh)

The `tsfresh.labeldriven` package builds binary classifier **generations**
whose only training signal is the set of *currently accepted human labels*.
It exists to enforce one rule end-to-end:

> A label reviewed against one version of a sequence can never silently
> travel onto a later, modified version of that sequence.

## Core concepts

| Concept | Meaning |
| --- | --- |
| Business ID | The ID callers know a sample by (may be reused). |
| Lineage | The durable internal identity behind a business ID. Deleting a sample retires its lineage; re-registering the same business ID mints a **new** lineage that must be reviewed again. |
| Revision | One immutable, content-hashed version of the lineage's **full** sequence. Revisions are numbered 1..n per lineage. |
| Review record | A human 0/1 decision bound to an exact `(lineage, revision)` with its own review version. States: `pending → accepted`, or `withdrawn`; reviews on superseded revisions become `expired`. |
| Generation | An immutable, fully provenanced bundle: training source, features, fitted model and ID-aligned probabilities. |

## Label lifecycle rules

* Only **accepted** reviews bound to the lineage's **current** revision
  produce a training label.
* `pending`, `withdrawn` and `expired` reviews — and samples with no review —
  are **excluded with an explicit reason**. They are never treated as class 0.
* Editing a sequence appends a revision and expires the prior live review, so
  re-review is mandatory before the sample can train again.
* Deleting a sample retires the lineage; ID reuse starts a fresh lineage.
* Training requires at least one accepted class-0 and one accepted class-1
  sample; otherwise `NoTrainingPopulationError` is raised.

## Building a generation

```python
from tsfresh.labeldriven import (
    LabelDrivenStore, ExtractionSpec, build_and_publish,
)
from tsfresh.feature_extraction.settings import MinimalFCParameters

store = LabelDrivenStore("./model_store")
store.create()

# 1. Register full raw sequences (long-format tsfresh DataFrames).
store.register_sample("S1", sequence_df_s1, created_at="2026-01-01T00:00:00Z")

# 2. Human review: submit then accept (0/1). Reviews bind to the revision.
review = store.submit_review("S1", label=1, created_at="2026-01-02T00:00:00Z")
store.accept_review(review.lineage_id, review.review_version,
                    decided_at="2026-01-02T01:00:00Z")

# 3. Build + publish a generation.
spec = ExtractionSpec(
    column_id="id", column_sort="time",
    default_fc_parameters=MinimalFCParameters(),
)
generation = build_and_publish(
    store, generation_id="gen-0001", created_at="2026-01-03T00:00:00Z",
    extraction_spec=spec,
)
generation.probabilities          # class-1 probability, aligned by lineage ID
generation.source.as_frame()      # audit table: who trained and who was excluded
```

### What happens inside a build

1. Resolve the effective training population from current lifecycle state.
2. Verify every raw sequence's content hash against its registered revision.
3. Extract features from the **actual current sequences** via native
   `tsfresh.extract_features`.
4. Fit imputation (`PerColumnImputer`) and native feature selection
   (`FeatureSelector`, Benjamini–Hochberg) **on the training population
   only**, then fit a fresh binary classifier.
5. Run the same fitted pipeline over every current sample to produce
   probabilities aligned to the full set of lineage IDs (unlabeled samples
   are scored but never trained on).

## Failure semantics

* Conflicts (stale revision, deleted lineage, illegal review transition,
  duplicate active review, hash drift) raise `ConflictError`/`StoreError` and
  reject the operation.
* Extraction, fitting or save failures raise without side effects; the
  previously published generation stays current. Results can never mix new
  labels with an old model.
* All artifacts write to a staging directory and are renamed into place only
  after checksums and a reload check pass; each generation carries a
  `checksums.json`.

## Reopening

```python
store = LabelDrivenStore("./model_store")
store.open()                              # verifies every stored sequence
generation = store.load_generation()      # verifies checksums + provenance
store.list_generations()
store.current_generation_id()
```

Loading a generation whose bound revisions or labels no longer match the
registry raises `StoreError`; rebuild and publish a new generation instead.
Generations are immutable and never overwritten.

## On-disk layout

```
model_store/
  registry.pkl                       # lineages, revisions, review records
  sequences/<lineage_id>/rev_<n>.parquet
  sequences/<lineage_id>/current.json
  generations/index.json             # ordered generations + current pointer
  generations/<generation_id>/
      generation.pkl                 # full Generation (model + provenance)
      features.parquet
      probabilities.csv
      checksums.json
```
