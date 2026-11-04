"""
Distributed entity resolution: connected components over the account-attribute graph in PySpark.

Accounts and attribute values (card, device, normalized e-mail) form a bipartite graph. Labels
are propagated by alternating min-label joins (accounts take the smallest label of their
attribute values; values take the smallest label of their accounts) until nothing changes; the
number of iterations is bounded by the graph's diameter. Values shared by more than MAX_GROUP
accounts are dropped first (they would otherwise glue unrelated accounts into one giant cluster).

  python spark/connected_components.py --worlds 1      # ~24K accounts
  python spark/connected_components.py --worlds 10     # ~240K accounts
Checks the result against scipy's single-machine connected components.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from pyspark.sql import SparkSession                   # noqa: E402
from pyspark.sql import functions as F                 # noqa: E402

from agl.graph import MAX_GROUP, STRONG, components, derive   # noqa: E402
from agl.sim import generate                           # noqa: E402


def build(worlds):
    parts = []
    for w in range(worlds):
        d = derive(generate(w))
        for c in STRONG:                                # keep worlds disjoint
            d[c] = f"w{w}:" + d[c].astype(str)
        d["account_id"] = f"w{w}:" + d.account_id
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def spark_components(spark, d, max_iter=50):
    acc = spark.createDataFrame(pd.DataFrame({"a": np.arange(len(d), dtype=np.int64)}))
    edges = []
    for c in STRONG:
        e = pd.DataFrame({"a": np.arange(len(d), dtype=np.int64), "v": c + "=" + d[c].astype(str)})
        edges.append(e)
    E = spark.createDataFrame(pd.concat(edges, ignore_index=True))
    big = E.groupBy("v").count().filter(F.col("count") > MAX_GROUP).select("v")
    E = E.join(big, "v", "left_anti").cache()
    lab = acc.withColumn("l", F.col("a")).localCheckpoint()
    it = 0
    for it in range(1, max_iter + 1):
        vlab = E.join(lab, "a").groupBy("v").agg(F.min("l").alias("l"))
        new = E.join(vlab, "v").groupBy("a").agg(F.min("l").alias("l2"))
        # localCheckpoint cuts the query plan each iteration (otherwise it grows with every join)
        nxt = lab.join(new, "a", "left").select("a", F.least("l", F.coalesce("l2", "l")).alias("l")).localCheckpoint()
        changed = nxt.join(lab.withColumnRenamed("l", "old"), "a").filter(F.col("l") != F.col("old")).count()
        lab = nxt
        if changed == 0:
            break
    out = lab.toPandas().sort_values("a")
    return out.l.to_numpy(), it


def same_partition(a, b):
    """True if two labelings induce the same clustering."""
    pa = pd.Series(b).groupby(a).nunique().max() == 1
    pb = pd.Series(a).groupby(b).nunique().max() == 1
    return bool(pa and pb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = build(args.worlds)
    t = time.time()
    ref = components(d)
    t_local = time.time() - t
    spark = (SparkSession.builder.appName("entity-resolution-cc").config("spark.sql.shuffle.partitions", "16")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t = time.time()
    lab, iters = spark_components(spark, d)
    t_spark = time.time() - t
    res = {"accounts": len(d), "clusters": int(len(np.unique(lab))), "iterations": iters,
           "matches_scipy": same_partition(lab, ref), "seconds_spark_local2cores": round(t_spark, 1),
           "seconds_scipy": round(t_local, 2), "largest_cluster": int(pd.Series(lab).value_counts().max())}
    print(res)
    out = args.out or f"results/spark_cc_{args.worlds}w.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    spark.stop()


if __name__ == "__main__":
    main()
