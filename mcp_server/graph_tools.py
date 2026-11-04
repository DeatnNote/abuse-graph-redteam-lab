"""
Analyst tools over the abuse graph (plain functions, unit-tested; served over MCP by server.py).
Raw cards, devices and e-mails are never returned: only cluster statistics and masked values.
"""
import hashlib

import numpy as np
import pandas as pd

from agl.graph import derive, features


def _mask(v):
    return "h:" + hashlib.sha256(str(v).encode()).hexdigest()[:10]


class GraphStore:
    def __init__(self, df, detector):
        self.d = derive(df).reset_index(drop=True)
        X, self.lab = features(df, detector.seed_bad, detector.er)
        self.score = detector.m.predict_proba(X[detector.cols])[:, 1]
        self.thr = detector.threshold
        self.idx = {a: i for i, a in enumerate(self.d.account_id)}

    def account_risk(self, account_id):
        i = self.idx.get(account_id)
        if i is None:
            return {"error": "unknown account"}
        return {"account_id": account_id, "risk_score": round(float(self.score[i]), 4),
                "flagged": bool(self.score[i] >= self.thr), "cluster_id": int(self.lab[i])}

    def cluster_summary(self, account_id, max_members=20):
        i = self.idx.get(account_id)
        if i is None:
            return {"error": "unknown account"}
        m = np.where(self.lab == self.lab[i])[0]
        sub = self.d.iloc[m]
        shared = {}
        for c in ("card", "device", "email_stem", "ip_block", "text_sig"):
            vc = sub[c].value_counts()
            shared[c] = [{"value": _mask(v), "accounts": int(n)} for v, n in vc[vc > 1].head(5).items()]
        return {"cluster_id": int(self.lab[i]), "accounts": int(len(m)),
                "flagged_accounts": int((self.score[m] >= self.thr).sum()),
                "created_span_days": round(float(sub.created_day.max() - sub.created_day.min()), 1),
                "median_domain_age_days": round(float(sub.domain_age_days.median()), 0),
                "shared_attributes": shared,
                "members": list(sub.account_id.head(max_members))}

    def top_clusters(self, n=10, min_size=3):
        s = pd.DataFrame({"c": self.lab, "flag": self.score >= self.thr, "score": self.score})
        g = s.groupby("c").agg(accounts=("flag", "size"), flagged=("flag", "sum"), mean_score=("score", "mean"))
        g = g[g.accounts >= min_size].sort_values(["flagged", "mean_score"], ascending=False).head(n)
        return [{"cluster_id": int(c), **{k: (int(v) if k != "mean_score" else round(float(v), 4)) for k, v in r.items()}}
                for c, r in g.iterrows()]
