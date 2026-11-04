"""results/world_*.json (+ spark_cc_*.json) -> RESULTS.md and a figure.   python -m agl.report"""
import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
W = [json.load(open(p)) for p in sorted(glob.glob("results/world_*.json"))]
NAMES = {"v0_account_only": "v0: account features only",
         "v1_graph": "v1: + entity resolution & graph features",
         "v1_graph_no_entity_resolution": "v1 without e-mail entity resolution",
         "v2_naive_with_spoofable_signal": "v2 naive: + prepaid-card & residential-IP signals",
         "v2_round1": "v2 hardening round 1 (prepaid-card signal + adversarial examples)",
         "v2_round2": "v2 hardening round 2",
         "v2_round3": "v2 hardening round 3"}


def rng_(v, fmt):
    v = np.asarray(v, float)
    return f"{fmt.format(v.mean())} ({fmt.format(v.min())}–{fmt.format(v.max())})"


def main():
    n = len(W)
    L = ["# Results\n",
         f"{n} independent simulated worlds, each with about {np.mean([w['accounts'] for w in W]):,.0f} advertiser accounts "
         f"(20,000 independent advertisers, 60 benign agencies, 60 fraud rings from 25 operators). Detectors train on half the "
         "rings and agencies and are tested on the other half. Mean (min–max) over worlds. All data simulated.\n",
         "Each test ring is then attacked by the automated red team, which can buy evasion actions for each of its accounts "
         "(prices in `agl/redteam.py`) and query the detector's risk score. A ring **evades** when fewer than half of its "
         "accounts are flagged. Threshold: 1% false positives on benign training accounts.\n",
         "## Detection before and after attack\n",
         "| Detector | Ring recall (no attack) | False-positive rate | Agency false-positive rate | Rings evading at ≤$10/account | at ≤$25 | at ≤$50 | at ≤$100 | Median cost to evade |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, lab in NAMES.items():
        d = [w["detectors"][k] for w in W]
        ev = lambda b: rng_([x["evasion_by_budget"][str(b)] for x in d], "{:.0%}")   # noqa: E731
        med = [x["cost_to_evade"]["median_cost_if_evaded"] for x in d]
        L.append(f"| {lab} | {rng_([x['ring_recall'] for x in d], '{:.0%}')} | {rng_([x['false_positive_rate'] for x in d], '{:.1%}')} | "
                 f"{rng_([x['agency_fpr'] for x in d], '{:.1%}')} | {ev(10)} | {ev(25)} | {ev(50)} | {ev(100)} | "
                 f"{rng_(med, '${:,.0f}')} |")
    acts = lambda k: pd.DataFrame([w["detectors"][k]["actions_used"] for w in W]).fillna(0).sum().sort_values(ascending=False)  # noqa: E731
    L += ["\n## What the red team bought\n", "| Detector | Evasion actions chosen (rings, summed over worlds) |", "|---|---|"]
    for k in ("v0_account_only", "v1_graph", "v2_naive_with_spoofable_signal", "v2_round1", "v2_round3"):
        a = acts(k)
        L.append(f"| {NAMES[k]} | " + ", ".join(f"{i} ({int(v)})" for i, v in a.items() if v > 0) + " |")
    L += ["\n## Findings\n",
          "* **Graph features catch every unmodified ring**, against 28-52% of rings for account features alone, and "
          "account-only detection is evaded for free or for a few dollars of new e-mail addresses.",
          "* **But v1 is cheap to beat**: fresh e-mails, paraphrased ads, prepaid cards and emulated devices break every "
          "link the graph relies on for roughly $20-30 per account.",
          "* **One round of hardening only moved the weak spot**: after training on the v1 attack variants, 92% of rings "
          "evaded the naive v2 and 71% evaded round 1 for $10 per account or less, mostly with a $5 residential proxy. "
          "The model apparently shifted its weight onto the IP block, which the first attack never changed.",
          "* **Iterated adversarial training raised the price**: after three rounds of attacking the training rings and "
          "retraining, 16-97% of test rings (depending on the world) could not evade at all with the full action menu, "
          "and those that did paid $90-146 per account instead of $23-28. Hardening is not free: false positives "
          "on benign agencies rose to 4-10% in two of the three worlds.",
          "* **E-mail entity resolution** raised the median cost to evade in two of three worlds ($18 vs $23, $13 vs $28) "
          "and cut false positives on agencies in one (15% to 0.3%); a determined ring simply buys unrelated e-mails.\n"]
    sp = [json.load(open(p)) for p in sorted(glob.glob("results/spark_cc_*.json"), key=os.path.getsize)]
    if sp:
        L += ["## Distributed entity resolution (PySpark)\n",
              "Connected components over the account-attribute graph by alternating min-label propagation; checked against scipy.\n",
              "| Accounts | Clusters | Iterations | Same clusters as scipy | Spark seconds (local, 2 cores) | scipy seconds |",
              "|---:|---:|---:|:---:|---:|---:|"]
        for s in sorted(sp, key=lambda x: x["accounts"]):
            L.append(f"| {s['accounts']:,} | {s['clusters']:,} | {s['iterations']} | {'yes' if s['matches_scipy'] else 'NO'} | "
                     f"{s['seconds_spark_local2cores']} | {s['seconds_scipy']} |")
        L.append("\nOn one machine scipy is far faster; the Spark job is the version that scales past one machine's memory.\n")
    with open("RESULTS.md", "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    # figure: share of rings evading vs budget
    BLUE, ORANGE, AQUA, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED, "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    budgets = [0, 1, 5, 10, 25, 50, 100, 200]
    for k, col, lab in (("v0_account_only", MUTED, "v0 account features"), ("v1_graph", BLUE, "v1 graph"),
                        ("v2_naive_with_spoofable_signal", ORANGE, "v2 after one naive hardening round"), ("v2_round3", AQUA, "v2 after 3 hardening rounds")):
        y = [np.mean([w["detectors"][k]["evasion_by_budget"][str(b)] for w in W]) * 100 for b in budgets]
        ax.plot(range(len(budgets)), y, marker="o", lw=2, ms=6, color=col, label=lab)
    ax.set_xticks(range(len(budgets)), [f"${b}" for b in budgets])
    ax.set_xlabel("Attacker budget per account")
    ax.set_ylabel("Test rings evading (%)")
    ax.set_title("How much it costs a fraud ring to evade each detector", color=INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig("results/figures/evasion_cost.png", dpi=150)


if __name__ == "__main__":
    main()
