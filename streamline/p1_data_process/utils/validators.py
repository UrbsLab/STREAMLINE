import os
import re
import pandas as pd

PAIR_RE = re.compile(r"^(?P<name>.+)_CV_(?P<fold>.+)_(?P<split>Train|Test)\.csv$")

def find_cv_pairs(cv_dataset_dir):
    pairs = {}
    for fname in os.listdir(cv_dataset_dir):
        m = PAIR_RE.match(fname)
        if not m:
            continue
        fold = str(m.group("fold"))
        split = m.group("split")
        pairs.setdefault(fold, {})
        pairs[fold][split] = os.path.join(cv_dataset_dir, fname)
    return {k: v for k, v in pairs.items() if "Train" in v and "Test" in v}

def validate_cv_pair(train_df, test_df, outcome_label, instance_label=None):
    # columns/order identical
    if list(train_df.columns) != list(test_df.columns):
        raise ValueError("Train/Test columns differ or are in different order")
    # outcome present
    if outcome_label not in train_df.columns or outcome_label not in test_df.columns:
        raise ValueError("Outcome column '%s' missing in Train or Test" % outcome_label)
    # disjoint instances
    if instance_label and instance_label in train_df.columns and instance_label in test_df.columns:
        a = set(train_df[instance_label].astype(str).tolist())
        b = set(test_df[instance_label].astype(str).tolist())
        if a & b:
            raise ValueError("Instance leakage across Train/Test for label '%s'" % instance_label)
