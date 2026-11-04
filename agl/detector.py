"""Detector training/evaluation with splits by RING / AGENCY (no group on both sides)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .graph import ACCOUNT_ONLY, HARDENING_ONLY, features


def lgbm(seed=0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=2)


def split_groups(df, frac_train=0.5, seed=0):
    rng = np.random.default_rng(seed)
    groups = df.group.unique()
    train_groups = set(g for g in groups if rng.random() < frac_train)
    return df.group.isin(train_groups).to_numpy()


def threshold_at_fpr(scores_benign, fpr=0.01):
    return float(np.quantile(scores_benign, 1 - fpr))


def ring_recall(df, flagged, mask):
    """Share of fraud rings (in mask) with at least half their accounts flagged."""
    d = pd.DataFrame({"g": df.group, "f": flagged, "k": df.kind})[mask & (df.kind == "fraud").to_numpy()]
    r = d.groupby("g").f.mean()
    return float((r >= 0.5).mean()), r


class Detector:
    def __init__(self, cols=None, seed=0, er=True):
        self.cols, self.seed, self.er = cols, seed, er

    def fit(self, df, train_mask, extra=None, k=5):
        """
        Known-bad features ("how many already-enforced accounts share this card") must be built
        OUT-OF-FOLD for training rows: otherwise every training fraud account sees its own ring-mates
        as known-bad, the model leans on that, and it fails on new rings that have no enforced
        neighbours yet.
        """
        y = df.is_fraud.to_numpy()
        seed_bad = (y * train_mask).astype(float)
        self.seed_bad = seed_bad
        groups = df.group.to_numpy()
        tg = np.unique(groups[train_mask])
        fold_of = dict(zip(tg, np.random.default_rng(self.seed).integers(0, k, len(tg))))
        fold = np.array([fold_of.get(g, -1) for g in groups])
        parts = []
        for f in range(k):
            sb = seed_bad * (fold != f)
            Xf, _ = features(df, sb, self.er)
            parts.append(Xf[fold == f])
        X = pd.concat(parts).sort_index()
        self.cols = self.cols or [c for c in X.columns if c not in HARDENING_ONLY]
        tm = train_mask[X.index]
        Xt, yt = X.loc[X.index[tm], self.cols], y[X.index[tm]]
        if extra is not None:                                         # adversarial examples (features, labels)
            Xt = pd.concat([Xt, extra[0][self.cols]], ignore_index=True)
            yt = np.r_[yt, extra[1]]
        # threshold: 1% false positives on benign training accounts, using out-of-fold scores
        from sklearn.model_selection import GroupKFold
        g = np.r_[groups[X.index[tm]], np.full(len(yt) - tm.sum(), "adversarial")] if extra is not None else groups[X.index[tm]]
        oof = np.zeros(len(yt))
        for a, b in GroupKFold(3).split(Xt, yt, g):
            oof[b] = lgbm(self.seed).fit(Xt.iloc[a], yt[a]).predict_proba(Xt.iloc[b])[:, 1]
        self.threshold = threshold_at_fpr(oof[yt == 0], 0.01)
        self.m = lgbm(self.seed).fit(Xt, yt)
        return self

    def score(self, df):
        X, _ = features(df, self.seed_bad, self.er)
        return self.m.predict_proba(X[self.cols])[:, 1]

    def evaluate(self, df, test_mask):
        s = self.score(df)
        y = df.is_fraud.to_numpy()
        flagged = s >= self.threshold
        rr, _ = ring_recall(df, flagged, test_mask)
        ben = test_mask & (y == 0)
        return {"ap": float(average_precision_score(y[test_mask], s[test_mask])),
                "account_recall_at_threshold": float(flagged[test_mask & (y == 1)].mean()),
                "false_positive_rate": float(flagged[ben].mean()),
                "agency_fpr": float(flagged[ben & (df.kind == "agency").to_numpy()].mean()),
                "ring_recall": rr}


def account_only():
    return Detector(cols=ACCOUNT_ONLY)


def hardened_cols(base_cols, include_spoofable=False):
    from .graph import SPOOFABLE
    return list(base_cols) + [c for c in HARDENING_ONLY if include_spoofable or c not in SPOOFABLE]
