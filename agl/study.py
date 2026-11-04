"""
Red-team study over independent simulated worlds.

  python -m agl.study        -> results/world_<seed>.json

Per world: train detectors on half the rings/agencies, test on the other half;
  v0  account features only (domain age, spend, e-mail form)
  v1  + entity-resolution clusters and shared-attribute graph features
  v1 without entity resolution (raw e-mails)
attack each with the automated red team on the TEST rings; then HARDEN v1 using the red team on
the TRAINING rings only:
  v2_naive  = v1 + prepaid-card and residential-IP signals + one round of adversarial training
  v2_r1..r3 = v1 + prepaid-card signal only; each round, attack the current model on the training
              rings, add the evasive variants as fraud examples, retrain
and attack every hardened model with a fresh, adaptive red team on the test rings.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import redteam as RT
from .detector import Detector, account_only, hardened_cols, split_groups
from .graph import features
from .sim import generate

OUT = "results"


def attacked_examples(df, det, rings, seed):
    d, _ = RT.attack(df, det, rings, seed=seed)
    return examples_from(d, det, rings)


def examples_from(d, det, rings):
    seed_bad = np.zeros(len(d))                       # attacked rings are treated as NEW rings (no enforced neighbours)
    X, _ = features(d, seed_bad, det.er)
    m = d.group.isin(rings).to_numpy()
    return X[m].reset_index(drop=True), np.ones(m.sum(), int)


def run_world(seed):
    t0 = time.time()
    df = generate(seed)
    tr = split_groups(df, seed=seed)
    te = ~tr
    test_rings = sorted(df.loc[te & (df.kind == "fraud"), "group"].unique())
    train_rings = sorted(df.loc[tr & (df.kind == "fraud"), "group"].unique())
    res = {"seed": seed, "accounts": len(df), "kinds": df.kind.value_counts().to_dict(),
           "test_rings": len(test_rings), "train_rings": len(train_rings)}
    dets = {"v0_account_only": account_only().fit(df, tr), "v1_graph": Detector().fit(df, tr),
            "v1_graph_no_entity_resolution": Detector(er=False).fit(df, tr)}
    v1 = dets["v1_graph"]
    extra = attacked_examples(df, v1, train_rings, seed + 1)
    dets["v2_naive_with_spoofable_signal"] = Detector(cols=hardened_cols(v1.cols, include_spoofable=True)).fit(df, tr, extra=extra)
    cur, ex = v1, None
    for k in (1, 2, 3):
        X, y = attacked_examples(df, cur, train_rings, seed + 10 * k)
        ex = (X, y) if ex is None else (pd.concat([ex[0], X], ignore_index=True), np.r_[ex[1], y])
        cur = Detector(cols=hardened_cols(v1.cols)).fit(df, tr, extra=ex)
        dets[f"v2_round{k}"] = cur
    res["detectors"] = {}
    for name, D in dets.items():
        r = D.evaluate(df, te)
        _, traj = RT.attack(df, D, test_rings, seed=seed + 2)
        r["evasion_by_budget"] = RT.evasion_curve(traj)
        r["cost_to_evade"] = RT.cost_to_evade(traj)
        r["actions_used"] = traj[traj["round"] > 0].action.value_counts().to_dict()
        res["detectors"][name] = r
        print(f"  {name}: ring recall {r['ring_recall']:.2f}, FPR {r['false_positive_rate']:.3f}, "
              f"agency FPR {r['agency_fpr']:.3f}, evaded {r['cost_to_evade']['evaded']}/{r['cost_to_evade']['rings']} "
              f"median ${r['cost_to_evade']['median_cost_if_evaded']}", flush=True)
    res["seconds"] = time.time() - t0
    return res


def main(seeds=(0, 1, 2)):
    os.makedirs(OUT, exist_ok=True)
    for s in seeds:
        p = f"{OUT}/world_{s}.json"
        if os.path.exists(p):
            continue
        print(f"world {s}", flush=True)
        r = run_world(s)
        with open(p, "w") as fh:
            json.dump(r, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or (0, 1, 2))
