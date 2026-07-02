# Local Examples And Experiments

This directory is the single home for local sample data, ad-hoc experiments and large test outputs. Keep the repository root limited to product code, deployment assets and documentation.

## Local Data

`local-data/` is ignored by Git because it contains recorded videos, generated labels, model experiments and large comparison outputs. The current workstation keeps these existing items there:

```text
local-data/
  sampleInput/          small annotator workspace
  sampleInput3/         segmented workspace sample
  vid1/                 video/tracking experiment
  vid2/                 video/tracking experiment
  OPD/                  OPD input data
  OPD_method_tests/     tracker/fusion comparison outputs
  InterVL/              local external-model experiment
  platform_tasks/       local platform smoke-test state
  annotation_test.json  legacy annotation sample
```

These names are descriptive only; code must not depend on their presence. New examples that are suitable for Git should be small, anonymized and placed outside `local-data/` with a short README explaining how to run them.
