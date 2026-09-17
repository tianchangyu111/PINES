# Data Preparation

## Binary Classification and Localization

Place before images in `data/susbtrat/`, PBS after images in `data/pbs/`, analyte
after images in `data/analyte/`, and illumination frames in `data/led_field/`.
Before and after filenames must match. The legacy spelling `susbtrat` is part
of the loader interface and is intentionally retained.

Image content, experimental concentrations, wavelengths, and batch identity are
not inferred reliably from filenames. Preserve an external, verified sample
manifest. No dataset or sample manifest is supplied in this source release.

## Gas Transfer

Use the same layout under `data/gas/`. `gas_transfer.py` reverses analyte pairs
by default for fluorescence-decrease responses; pass `--no-reverse-analyte`
when this is not appropriate. `localization_comparison.py` does not reverse
pairs by default; its `--reverse-analyte` flag explicitly enables reversal.
These defaults are retained from the research source.

## Four-Class Workflow

```text
data/concentration_archives/
  pbs.zip
  susbtrat.zip
data/concentration/
  1uM/analyte/
  1uM/susbtrat/
  10uM/analyte/
  10uM/susbtrat/
  100uM/analyte/
  100uM/susbtrat/
```

Archive image basenames must match for PBS/before pairing. File basenames must
match within each concentration directory. Provide verified class labels from
experimental metadata. Historical local concentration folders were created by
ranking response area; they must not be treated as independently verified
experimental concentration labels. The ranking utility is deliberately excluded.

The four-class script has its own preprocessing/feature implementation. Do not
assume it performs the same LED correction or every feature operation as the
binary pipeline merely because both have 29 input channels.

## Unknown Images

Provide trusted dataset ZIP files in `data/unknown_archives/`. Consult the
archive-pairing loader in `scripts/resnet_inference.py` for supported archive
names. Run inference in a fresh work directory and supply a compatible model
directory from the corresponding training workflow.
