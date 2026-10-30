"""Fast correctness checks; these do not measure recognition performance."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from data import make_windows
from losses import FocalLoss
from models import build_model
from train import epoch_learning_rate, model_kwargs, parse_args
from workshop import RUNS, collect_summary, training_args


class WorkshopChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_focal_gamma_zero_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, -0.5, 0.1], [-0.5, 0.2, 1.3]], requires_grad=True)
        labels = torch.tensor([0, 1])
        focal = FocalLoss(gamma=0)(logits, labels)
        torch.testing.assert_close(focal, F.cross_entropy(logits, labels))
        focal.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_downweights_easy_examples_more(self):
        label = torch.tensor([0])
        easy, hard = torch.tensor([[5.0, -5.0]]), torch.tensor([[0.1, 0.0]])
        easy_ratio = FocalLoss()(easy, label) / F.cross_entropy(easy, label)
        hard_ratio = FocalLoss()(hard, label) / F.cross_entropy(hard, label)
        self.assertLess(easy_ratio.item(), hard_ratio.item())

    def test_focal_extreme_logits_have_finite_loss_and_gradients(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
        loss = FocalLoss()(logits, torch.tensor([0, 0]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_warmup_reaches_requested_rate(self):
        rates = [epoch_learning_rate(i, 0.001, 100, 5) for i in range(1, 8)]
        np.testing.assert_allclose(rates, [0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.001, 0.001])
        self.assertEqual(epoch_learning_rate(50, 0.001, 100), 0.001)

    def test_warmup_then_cosine_is_nonnegative(self):
        rates = [epoch_learning_rate(i, 0.001, 100, 5, True) for i in range(1, 101)]
        self.assertTrue(all(0 <= value <= 0.001 for value in rates))
        self.assertLess(rates[-1], rates[5])

    def test_each_variant_changes_only_one_factor_from_its_baseline(self):
        configs = {run[0]: vars(parse_args(training_args(run))) for run in RUNS}
        self.assertEqual(len(configs), 13)
        for name, _, baseline, _ in RUNS:
            if baseline:
                changed = {key for key in configs[name] if key != "name"
                           and configs[name][key] != configs[baseline][key]}
                self.assertEqual(len(changed), 1, (name, baseline, changed))

    def test_all_planned_architectures_accept_their_window(self):
        for run in RUNS:
            args = parse_args(training_args(run))
            cfg = vars(args)
            cfg.update(filters=2, rnn_units=4)
            model = build_model(args.model, 3, 18, args.window, **model_kwargs(cfg))
            result = model(torch.randn(2, args.window, 3))
            self.assertEqual(tuple(result.shape), (2, 18), run[0])
            self.assertTrue(torch.isfinite(result).all())

    def test_windows_do_not_cross_recordings_and_use_last_label(self):
        x = np.concatenate([np.zeros((4, 2)), np.ones((4, 2))])
        y = np.arange(8)
        windows, labels = make_windows(x, y, [4, 4], 3, 1)
        np.testing.assert_array_equal(labels, [2, 3, 6, 7])
        self.assertTrue(all(np.all(w == w[0]) for w in windows))

    def test_part_d_summary_uses_its_own_baseline(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for variant, score in [('ref', 0.70), ('D_ref', 0.80), ('D_no_dropout', 0.82)]:
                    run = next(item for item in RUNS if item[0] == variant)
                    folder = Path('runs') / f'{variant}_s0'
                    folder.mkdir(parents=True)
                    payload = {'config': vars(parse_args(training_args(run))),
                               'test_f1_weighted': score, 'test_f1_macro': score,
                               'test_accuracy': score, 'params': 1, 'best_epoch': 1,
                               'train_minutes': 1}
                    (folder / 'metrics.json').write_text(json.dumps(payload))
                with contextlib.redirect_stdout(io.StringIO()):
                    collect_summary()
                with open('workshop_comparison.csv') as stream:
                    row = next(row for row in csv.DictReader(stream) if row['run'] == 'D_no_dropout_s0')
                self.assertEqual(row['compare_with'], 'D_ref_s0')
                self.assertAlmostEqual(float(row['delta_weighted_F1']), 0.02)
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
