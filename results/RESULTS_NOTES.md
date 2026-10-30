# DeepConvLSTM workshop: recovered results notes

These notes use the executed notebook outputs. They describe one seed (seed 0). Learning-curve evidence comes from printed epoch values rounded to four decimals. Class-level F1 values are rounded to three decimals in the original printed reports.

## Experiment setup

The recorded environment was a Tesla T4 with PyTorch 2.11.0+cu128 and CUDA 12.8. The notebook requests 100 maximum epochs. Early stopping uses validation weighted F1 with patience 20. The source reference uses 113 channels, 18 classes including Null, 24-sample windows, four convolutional layers, two LSTM layers, dropout 0.5, RMSProp, learning rate 0.001 and batch size 100. The reference has 41,399 training windows, 5,076 validation windows and 9,890 test windows.

## Final test results

| Configuration | Weighted F1 (%) | Macro F1 (%) | Accuracy (%) | Parameters | Best epoch | Training minutes |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 88.73 | 59.12 | 88.98 | 3,965,778 | 38 | 17.78 |
| CNN | 86.29 | 44.61 | 87.48 | 7,486,546 | 29 | 13.11 |
| GRU | 89.38 | 60.93 | 89.49 | 2,990,418 | 22 | 12.40 |
| 1 convolution | 88.79 | 57.55 | 88.69 | 3,904,146 | 25 | 6.98 |
| 2 convolutions | 89.89 | 63.47 | 90.10 | 3,924,690 | 23 | 9.67 |
| Window 20 | 89.08 | 58.14 | 88.95 | 3,965,778 | 25 | 9.88 |
| Window 60 | 89.86 | 62.08 | 89.81 | 3,965,778 | 16 | 42.68 |
| D baseline | 88.73 | 57.39 | 88.97 | 3,965,778 | 42 | 19.34 |
| D: no dropout | 88.81 | 58.61 | 89.08 | 3,965,778 | 46 | 20.37 |
| D: no weight decay | 88.96 | 57.62 | 89.10 | 3,965,778 | 43 | 19.42 |
| D: no augmentation | 88.43 | 56.57 | 88.41 | 3,965,778 | 37 | 17.47 |
| Warm-up 5 | 88.97 | 60.09 | 89.20 | 3,965,778 | 57 | 23.67 |
| Focal loss | 89.44 | 58.73 | 89.42 | 3,965,778 | 97 | 30.65 |

## A. CNN and GRU

The CNN scores below the LSTM reference on both F1 metrics and has more parameters (7,486,546 versus 3,965,778). GRU improves weighted F1 to 89.38% and macro F1 to 60.93%, using 2,990,418 parameters: approximately 24.6% fewer than LSTM.

## B. Convolutional depth

Two convolutional layers have the highest observed weighted F1 (89.89%) and macro F1 (63.47%) among these runs. The improvements over the reference are 1.16 and 4.35 percentage points. One convolution slightly improves weighted F1 while lowering macro F1. These tested settings do not establish a universally optimal network depth.

## C. Window length

The 20-sample window improves weighted F1 but lowers macro F1 relative to the 24-sample reference. The 60-sample window improves both metrics, reaching 89.86% weighted F1 and 62.08% macro F1. Its recorded training time is 42.68 minutes versus 17.78 minutes for the reference. Parameters are unchanged because the recurrent weights are shared across time; longer windows increase sequence length. Changing the window length also changes the number of valid windows, as recorded in each recovered metric file.

## D. Regularization

Part D must be compared with D_ref, which has dropout, weight decay 0.0001 and augmentation enabled. The supplied paper-style reference already has weight decay and augmentation disabled, so a separate D baseline was necessary.

| Ablation versus D_ref | Delta weighted F1 (percentage points) | Delta macro F1 (percentage points) |
|---|---:|---:|
| D: no dropout | +0.09 | +1.22 |
| D: no weight decay | +0.23 | +0.23 |
| D: no augmentation | -0.30 | -0.82 |

Removing augmentation lowers both final F1 metrics. Removing dropout or weight decay produces small increases in this seed. Use `figures/part_d_regularization_reconstructed.png` to inspect training and validation losses. A training/validation loss gap alone is not a causal measure of overfitting here: dropout and augmentation act during training and alter the conditions under which the two losses are measured.

### Matched-epoch loss observations

At epoch 40, the recovered losses are:

| Part D setting | Training CE | Validation CE |
|---|---:|---:|
| D baseline | 0.3218 | 0.4448 |
| No dropout | 0.1482 | 0.5264 |
| No weight decay | 0.3012 | 0.4897 |
| No augmentation | 0.1853 | 0.4504 |

The no-dropout run fits the training data more closely while validation cross
entropy is higher at this matched epoch. Its later loss curves show a widening
gap, a pattern consistent with overfitting. Nevertheless, its selected
checkpoint has slightly better test F1 than D_ref. Training-mode regularization
affects training loss, and cross entropy and F1 measure different aspects of
predictions, so a larger loss gap does not establish a lower test F1.

## E. Learning-rate warm-up

The recovered first five learning rates are 0.0002, 0.0004, 0.0006, 0.0008 and 0.001. Warm-up reaches 88.97% weighted F1 and 60.09% macro F1, small gains over the reference. `figures/part_e_warmup_reconstructed.png` compares the learning rates and the first 20 epochs of training/validation cross entropy. Final-score improvements alone do not establish that warm-up reduces every loss spike.

## F. Focal Loss

Focal Loss increases weighted F1 by 0.71 percentage points, while macro F1 decreases by 0.38 points. Null accounts for 83.27% of the reference test windows. Among the 17 non-Null activities, the printed class reports show 10 F1 increases and 7 decreases. Improvements are therefore mixed across classes; the weighted-F1 gain is not evidence that all rare classes improve.

| Activity | Support | Reference F1 | Focal F1 | Change |
|---|---:|---:|---:|---:|
| Null | 8235 | 0.946 | 0.950 | +0.004 |
| Open Fridge | 230 | 0.598 | 0.673 | +0.075 |
| Open Drawer 1 | 39 | 0.424 | 0.354 | -0.070 |
| Close Drawer 2 | 27 | 0.400 | 0.310 | -0.090 |
| Close Drawer 3 | 60 | 0.537 | 0.438 | -0.099 |
| Drink from Cup | 318 | 0.532 | 0.641 | +0.109 |

Full class comparisons are available in `per_class_comparison.csv` and `figures/part_f_class_f1_comparison.png`. The training focal objective and validation cross entropy are plotted separately. Training cross entropy for the focal run was not printed and cannot be reconstructed.

## Time and interpretation

The recorded training times total 243.41 minutes (4 hours 3.4 minutes). These times are affected by both computational cost per epoch and how many epochs each run completed before early stopping. They do not isolate per-epoch architecture speed.

All comparisons are from one seed. No statistical significance or run-to-run stability has been established. Original confusion matrices are available only for the reference as an embedded PNG. See README_RECOVERY.md for the complete recovery provenance and unavailable files.
