# Recovered DeepConvLSTM experiment outputs

The uploaded executed notebook contains all 13 completed experiment logs, all
13 printed class reports, the displayed final comparison table, and two original
reference figures. This bundle was recovered from that notebook without training
any model again.

## What is included

- `executed_notebook.ipynb`: the uploaded notebook, copied without modification.
- `source/deepconvlstm/`: the original embedded workshop package, extracted without executing it.
- `workshop_comparison.csv`: the 13-row displayed comparison table reconstructed as CSV.
- `runs/<run>/training_log.txt`: that run's printed training output.
- `runs/<run>/classification_report.txt`: the printed test class report.
- `runs/<run>/history_recovered.csv`: all printed epoch records.
- `runs/<run>/per_class_recovered.csv`: class precision, recall, F1 and support parsed from the report.
- `runs/<run>/metrics_recovered.json`: the displayed final metrics and logged metadata, with provenance.
- `runs/<run>/learning_curves_reconstructed.png`: a new plot of the recovered epoch records.
- `runs/ref_s0/learning_curves_original.png`: the original embedded reference learning-curve image.
- `runs/ref_s0/confusion_matrix_original.png`: the original embedded reference confusion-matrix image.
- `figures/`: combined comparison plots for all runs, Part D, Part E and Part F.
- `per_class_comparison.csv`: reference, focal and two-convolution class F1 values.
- `RESULTS_NOTES.md`: experiment findings that can support the workshop report.
- `recovery_manifest.json`: recovery counts, source hashes and precision limitations.

## Recovery precision

There are 743 contiguous epoch records across 13 runs.
The comparison table retains the values displayed in the notebook, usually six
decimals. Epoch losses and validation F1 were printed to four decimals. Class
precision, recall and F1 were printed to three decimals. Reconstructed CSVs and
plots retain those displayed values rather than pretending to reproduce hidden
full-precision files. Raw printed logs are included for traceability.

The two `*_original.png` files preserve the decoded embedded PNG bytes. The
other figures are newly plotted from recovered values and named accordingly.
Training focal loss and validation cross entropy are different objectives and
are shown in separate panels. The focal run's training cross entropy was not
printed and has not been invented.

## Files not recovered

The original live-session ZIP and full-precision history/metrics files were not
embedded. The other 12 original confusion matrices, predictions, checkpoints and
processed dataset are unavailable in this notebook. Class reports cannot determine
the full confusion matrix, so none of those matrices has been fabricated.

The recovered metric JSONs are deliberately named `metrics_recovered.json`; they
are not optimizer-state checkpoints and do not enable interrupted training to
resume. The source code can reproduce the experiment plan in a new environment.
The original export cell also records a later `NameError` for `ZipFile`, consistent
with that cell being executed without the setup variables in the current session.

## Source of the observations

All performance values, histories and reports come from the uploaded
`notebook4721f39ce7.ipynb`. The setup output identifies PyTorch 2.11.0+cu128,
CUDA 12.8 and a Tesla T4. All runs use seed 0 and the notebook requests a maximum
of 100 epochs with early stopping patience 20.

## Future exports

`export_kaggle_results.py` is a self-contained export cell for future Kaggle
training sessions. It imports its own dependencies and checks that completed-run
files are present before creating a ZIP. Paste its contents into a cell while
the trained files are still available. It does not recover deleted live files.
Preserve a completed run's output files with a saved notebook version or download
them before ending the session.
