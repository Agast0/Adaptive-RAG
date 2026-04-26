"""Helpers used only by the 4-way (A/B/C/D) classifier trainer.

Kept separate from the original ``classifier/utils.py`` so the existing
3-way pipeline is not touched. Only adds a per-class accuracy helper that
includes class D.
"""

from __future__ import annotations


def calculate_accuracy_perClass_plus(gold_answers, predictions):
    classes = ("A", "B", "C", "D")
    out = {}
    for cls in classes:
        gold_num = sum(1 for g in gold_answers if g == cls)
        pred_num = sum(1 for p in predictions if p == cls)
        correct = sum(1 for g, p in zip(gold_answers, predictions) if g == p == cls)
        acc = (correct / gold_num) * 100 if gold_num != 0 else -1
        label = {"A": "zero", "B": "single", "C": "multi", "D": "global"}[cls]
        out[f"{cls} ({label}) acc"] = acc
        out[f"{cls} ({label}) pred num"] = pred_num
        out[f"{cls} ({label}) gold num"] = gold_num
    return out
