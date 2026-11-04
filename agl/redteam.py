"""
Automated red team: a budget-aware black-box attacker that rewrites fraud rings to evade the
detector, plus a hardening loop.

The attacker can query the detector's risk SCORE for its own accounts (a worst-case assumption:
real attackers see only bans, which is slower but similar), but not its features or weights.
Each EVASION ACTION has a cost per account (what it costs a real operator: fresh cards, devices,
proxies, aged domains, ...). In each round, every ring tries every action it has not used yet
(one detector query per action, all rings at once) and keeps the one that lowers its accounts'
mean risk score (log-odds) the most per dollar. It stops when fewer than 10% of its accounts are flagged,
when nothing lowers the score, or after max_rounds.

PLANNERS: the default planner is this greedy search. `LLMPlanner` sends the same ring state and
action menu to an LLM and asks it to choose; it needs an API key and was not run for the results.
"""
import numpy as np
import pandas as pd

ACTIONS = {
    # name: (cost per account in USD, description)
    "email_variants":   (0.5, "re-spell the e-mail (+tags, dots, digits)"),
    "fresh_emails":     (3, "buy unrelated web-mail accounts"),
    "paraphrase_ads":   (0.2, "LLM-paraphrase the ad text"),
    "residential_ips":  (5, "residential proxy per account"),
    "new_devices":      (15, "emulated device per account"),
    "prepaid_cards":    (10, "a fresh prepaid virtual card per account"),
    "credit_cards":     (60, "a fresh credit card per account (synthetic identity)"),
    "stagger_creation": (8, "spread sign-ups over a month"),
    "aged_domains":     (25, "buy aged landing domains"),
    "throttle_spend":   (20, "keep week-1 spend at normal levels (lost revenue)"),
    "agency_camouflage": (60, "operate through a compromised agency's network"),
}


def apply_action(df, idx, action, rng, agencies):
    d = df
    n = len(idx)
    if action == "email_variants":
        stems = d.loc[idx, "email"].str.split("@").str[0].str.split("+").str[0].str.replace(".", "", regex=False)
        d.loc[idx, "email"] = [f"{'.'.join(s_) if rng.random() < .5 else s_}+{rng.integers(1e6)}@gmail.com" for s_ in stems]
    elif action == "fresh_emails":
        d.loc[idx, "email"] = [f"u{rng.integers(1e9)}@outlook.com" for _ in range(n)]
    elif action == "paraphrase_ads":
        d.loc[idx, "text"] = [" ".join(f"p{rng.integers(1e6)}" for _ in range(12)) for _ in range(n)]
    elif action == "residential_ips":
        d.loc[idx, "ip"] = [f"{rng.integers(60, 100)}.{rng.integers(256)}.{rng.integers(256)}.{rng.integers(256)}" for _ in range(n)]
    elif action == "new_devices":
        d.loc[idx, "device"] = [f"emu{rng.integers(1e12)}" for _ in range(n)]
    elif action == "prepaid_cards":
        d.loc[idx, "card"] = [f"pp{rng.integers(1e12)}" for _ in range(n)]
        d.loc[idx, "card_type"] = "prepaid"
    elif action == "credit_cards":
        d.loc[idx, "card"] = [f"cc{rng.integers(1e12)}" for _ in range(n)]
        d.loc[idx, "card_type"] = "credit"
    elif action == "stagger_creation":
        d.loc[idx, "created_day"] = d.loc[idx, "created_day"].min() + rng.uniform(0, 30, n)
    elif action == "aged_domains":
        d.loc[idx, "domain_age_days"] = rng.lognormal(6.0, 1.0, n)
    elif action == "throttle_spend":
        d.loc[idx, "spend_week1"] = rng.lognormal(4.3, 1.3, n)
    elif action == "agency_camouflage":
        a = agencies[rng.integers(len(agencies))]
        d.loc[idx, "ip"] = [f"{a}.{rng.integers(1, 20)}" for _ in range(n)]
        d.loc[idx, "email"] = [f"client{rng.integers(1e6)}@{a.replace('.', '-')}-partners.com" for _ in range(n)]
    return d


def attack(df, detector, rings, seed=0, max_rounds=11, stop_share=0.1, actions=ACTIONS):
    """
    Returns (attacked df, per-ring trajectory). Trajectory rows: ring, round, action, cost_per_account
    (cumulative), flagged_share.
    """
    rng = np.random.default_rng(seed)
    d = df.copy()
    agencies = d.loc[d.kind == "agency", "ip"].str.rsplit(".", n=1).str[0].unique()
    members = {r: d.index[d.group == r].to_numpy() for r in rings}

    def query(frame):
        s = detector.score(frame)
        f = s >= detector.threshold
        lo = np.log(np.clip(s, 1e-9, 1 - 1e-9) / (1 - np.clip(s, 1e-9, 1 - 1e-9)))     # log-odds: scores saturate near 1
        return {r: float(f[m].mean()) for r, m in members.items()}, {r: float(lo[m].mean()) for r, m in members.items()}

    cur, cur_s = query(d)
    used = {r: set() for r in rings}
    cost = {r: 0.0 for r in rings}
    traj = [{"ring": r, "round": 0, "action": None, "cost_per_account": 0.0, "flagged_share": cur[r]} for r in rings]
    active = [r for r in rings if cur[r] >= stop_share]
    for rnd in range(1, max_rounds + 1):
        if not active:
            break
        trial = {}
        for a in actions:
            cand = [r for r in active if a not in used[r]]
            if not cand:
                continue
            t = d.copy()
            for r in cand:
                t = apply_action(t, members[r], a, rng, agencies)
            _, ss = query(t)
            for r in cand:
                trial.setdefault(r, []).append((a, ss[r]))
        nxt = []
        for r in active:
            best = None
            for a, sc in trial.get(r, []):
                gain = cur_s[r] - sc
                if gain > 0.05:
                    val = gain / actions[a][0]
                    if best is None or val > best[0]:
                        best = (val, a)
            if best is None:
                continue
            _, a = best
            d = apply_action(d, members[r], a, rng, agencies)
            used[r].add(a)
            cost[r] += actions[a][0]
            nxt.append((r, a))
        cur, cur_s = query(d)
        for r, a in nxt:
            traj.append({"ring": r, "round": rnd, "action": a, "cost_per_account": cost[r], "flagged_share": cur[r]})
        nxt = [r for r, _ in nxt]
        active = [r for r in nxt if cur[r] >= stop_share]
    return d, pd.DataFrame(traj)


def evasion_curve(traj, budgets=(0, 1, 5, 10, 25, 50, 100, 200), evade_below=0.5):
    """Share of rings with fewer than half their accounts flagged, if each may spend up to `budget` per account."""
    out = {}
    for b in budgets:
        ok = traj[traj.cost_per_account <= b].groupby("ring").flagged_share.min()
        out[b] = float((ok < evade_below).mean())
    return out


def cost_to_evade(traj, evade_below=0.5):
    t = traj[traj.flagged_share < evade_below].groupby("ring").cost_per_account.min()
    rings = traj.ring.nunique()
    return {"rings": int(rings), "evaded": int(len(t)), "median_cost_if_evaded": float(t.median()) if len(t) else None}


class LLMPlanner:
    """Optional: ask an LLM to pick the next evasion action from the menu. Requires ANTHROPIC_API_KEY. Not used for results."""

    def __init__(self, model="claude-sonnet-4-5"):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def choose(self, ring_state, options):
        menu = "\n".join(f"- {a}: ${ACTIONS[a][0]}/account, {ACTIONS[a][1]}" for a in options)
        msg = (f"You are red-teaming an ads fraud detector in a SIMULATION. Ring state: {ring_state}.\n"
               f"Pick ONE action name from this menu that most cheaply reduces detection:\n{menu}\nAnswer with the name only.")
        r = self.client.messages.create(model=self.model, max_tokens=20, messages=[{"role": "user", "content": msg}])
        name = r.content[0].text.strip()
        return name if name in options else options[0]
