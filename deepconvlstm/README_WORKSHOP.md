# Workshop 2: DeepConvLSTM - start here

This package extends the supplied course code to cover the experiments requested
in Workshop.pdf. Start with the reference run. Train all final comparisons with
the same seed, maximum epochs, data split, early-stopping rule, batch size, and
learning rate, except for the particular setting being studied.

## Google Colab, step by step

1. Download and extract the updated `deepconvlstm.zip` on your computer.
2. Upload the updated ZIP itself to the top level of Google Drive, `MyDrive`.
   Keep its filename exactly `deepconvlstm.zip`.
3. Open Google Colab and choose **File > Upload notebook**. Select
   `deepconvlstm/workshop_colab.ipynb` from the extracted folder.
4. Choose a GPU runtime (T4 if available).
5. Run notebook cells in order through **Train the reference**. Approve your own
   Google Drive connection when Colab requests it.
6. Read the reference scores and inspect the plots. Then run Parts A through F,
   one part at a time. You do not have to train all parts in one session.
7. At the end of every session, run the summary cell. Your results are stored in
   `MyDrive/deepconvlstm_workshop/runs/`, and the comparison table is stored in
   `MyDrive/deepconvlstm_workshop/workshop_comparison.csv`.

The data download and preprocessing happen once. Subsequent sessions reuse the
processed data on Drive. Completed runs are skipped only when their saved
settings match the requested settings. An interrupted training run restarts from
epoch 1; this version does not implement resuming the optimizer state.

## The experiment plan

There are **13 runs**, including the reference. The extra run compared with the
12 originally counted is the necessary Part D comparison baseline explained below.

| Part | Run name before `_s0` | Change | Compare with |
| --- | --- | --- | --- |
| 0 | ref | Supplied default DeepConvLSTM | Reference |
| A | A_cnn | Dense layers instead of LSTM | ref |
| A | A_gru | GRU instead of LSTM | ref |
| B | B_conv1 | One convolutional layer | ref |
| B | B_conv2 | Two convolutional layers | ref |
| C | C_w20 | 20 samples = 0.67 seconds | ref |
| C | C_w60 | 60 samples = 2.0 seconds | ref |
| D | D_ref | Dropout 0.5, weight decay 0.0001, augmentation on | D comparison baseline |
| D | D_no_dropout | Dropout off; retain D_ref's other settings | D_ref |
| D | D_no_weight_decay | Weight decay off; retain D_ref's other settings | D_ref |
| D | D_no_augmentation | Augmentation off; retain D_ref's other settings | D_ref |
| E | E_warmup5 | Learning rate increases over first 5 epochs | ref |
| F | F_focal | Focal Loss, gamma=2, instead of cross entropy | ref |

All runs use gesture recognition with the Null class, all 113 channels, seed 0,
RMSProp, learning rate 0.001, batch size 100, and a maximum of 100 epochs unless
the listed variation changes a setting. Early stopping uses validation weighted
F1 with patience 20. Window step stays at 12 samples when window length changes.

The workshop tables also list zero convolutions and a 42-sample window. Its
explicit run instructions request one/two convolutions and 20/60 samples; those
are the default plan. Optional extra commands are given below.

## Why Part D has its own baseline

In the supplied `train.py`, weight decay is already 0 and augmentation is already
off. Removing either from the supplied reference would produce the same settings.
For a meaningful removal experiment, Part D first trains `D_ref` with all three
regularizers enabled. Each D variant then removes exactly one regularizer.
The choice of weight decay 0.0001 is an experimental setting, not a value
specified by the workshop. Report this choice and compare D variants with
`D_ref`, not with the paper reference.

## Commands for a local machine

```bash
pip install -r requirements.txt
python download_data.py
python preprocess.py
python workshop.py --list
python workshop.py --part ref
python workshop.py --part A
python workshop.py --part B
python workshop.py --part C
python workshop.py --part D
python workshop.py --part E
python workshop.py --part F
python summarize.py
python workshop.py --summary
```

The original `ablations.sh` is a much larger A-H experiment suite, and its part
letters do not match the current workshop. Use `workshop.py` for this assignment.

Optional extras, if requested:

```bash
python train.py --name extra_conv0_s0 --n-conv 0
python train.py --name extra_w42_s0 --window 42
```

These optional runs are collected by `summarize.py`; `workshop_comparison.csv`
contains only the 13 defined workshop variants.

## Understanding the outputs

Every completed run saves:

- `metrics.json`: settings, number of parameters, best epoch, training time,
  test weighted F1, test macro F1, and test accuracy.
- `history.csv`: epoch-by-epoch learning rate, optimized training loss, training
  cross entropy, validation cross entropy, and validation scores.
- `classification_report.txt`: precision, recall, F1, and support per gesture.
- `confusion_matrix.csv` and `.png`: which gestures were confused.
- `learning_curves.png`: training/validation cross entropy and validation F1.
- `best_model.pt`: the checkpoint selected on validation weighted F1.

`workshop_comparison.csv` includes each run's correct comparison baseline and
score differences. A delta of 0.01 means one percentage point when F1 is
expressed as a percentage. It does not mean a one-percent relative improvement.

For Part D, inspect training/validation curves for overfitting. For Part E,
inspect losses and the `lr` column: the first five rates should be 0.0002,
0.0004, 0.0006, 0.0008, 0.001. Compare loss spikes and final scores; warm-up is
not guaranteed to improve them. For Part F, focus on macro F1 and the rare
gestures' F1, as well as weighted F1. Keep checkpoint selection the same as the
reference so only the loss changes.

With Focal Loss, `train_loss` is the focal objective and is not directly
comparable with cross entropy. The learning-curve plot uses the separate
`train_cross_entropy` field so both displayed loss curves have the same meaning.
Training curves are recorded during training, while validation uses evaluation
mode; dropout and augmentation can therefore affect their gap.

## Checks and reporting

```bash
python -m unittest discover -s tests -v
```

These checks verify Focal Loss, warm-up, valid model shapes, recording boundaries,
and that every ablation changes exactly one factor from its stated baseline.
They do not replace training and evaluation on OPPORTUNITY.

Keep code, result tables, plots used in your report, and the joint report on
GitHub. The included `.gitignore` excludes the raw dataset and model weights.
If there is time for extra seeds, run the same plan with `--seed 1`, then
`--seed 2`. One seed supplies a result but does not estimate run-to-run variation.

## Changes to the supplied package

- Added `losses.py`, with multiclass Focal Loss.
- Extended `train.py` with `--loss`, `--focal-gamma`, and `--warmup-epochs`.
  Defaults retain ordinary cross entropy and no warm-up.
- Logged learning rates and comparable cross-entropy curves.
- Added `workshop.py`, this guide, `workshop_colab.ipynb`, and focused checks.
- Left model architecture, data splitting, preprocessing, and the original
  general-purpose experiment scripts unchanged.

## Validation completed before delivery

Checked on CPU with PyTorch 2.6.0: nine focused tests; short end-to-end training
on synthetic data for ordinary cross entropy and for Focal Loss plus warm-up;
saved checkpoint reloads and output generation; notebook code-cell compilation;
and the actual UCI download/preprocessing pipeline. Both full-size default
models also completed a forward pass on real OPPORTUNITY windows.

The reference preprocessing produced 41,399 training windows, 5,076 validation
windows, and 9,890 test windows. The supplied PyTorch model implementations have
3,965,778 parameters for DeepConvLSTM and 7,486,546 for the baseline CNN.
These are implementation/data checks, not recognition-performance results.
Full OPPORTUNITY training and execution inside Colab have not been run here.
