"""
Simulated advertiser accounts with fraud rings, benign agencies and KNOWN ground truth.

  legit advertisers   mostly unique cards, devices and IPs (a few carrier-NAT IP collisions),
                      older landing domains, varied ad text
  agencies            benign clusters: one agency runs 8-80 client accounts from the same office IP
                      block, a couple of company cards and shared employee devices -> hard negatives
  fraud rings         25 operators run 60 rings of 5-40 accounts. Accounts in a ring share a few
                      cards, devices and datacenter IP blocks, are created in a burst, use e-mail
                      variants of one pattern, new domains and near-duplicate scam ad text.
                      Operators reuse some infrastructure across their rings.

Each account is a row of attributes; the analyst sees no ring or operator ids.
"""
import numpy as np
import pandas as pd

LEGIT_N = 20_000
N_AGENCIES = 60
N_OPERATORS = 25
N_RINGS = 60
VOCAB_LEGIT = [f"w{i}" for i in range(3000)]
SCAM_TEMPLATES = [
    "limited time crypto doubling guaranteed returns click now verify wallet bonus".split(),
    "miracle weight loss pills celebrity secret doctors hate this trick free trial".split(),
    "official support refund your account suspended verify identity call now urgent".split(),
    "luxury watches ninety percent off clearance authentic warehouse sale today only".split(),
    "work from home earn thousands daily no experience needed sign up instantly".split(),
]
FIRST = ["john", "mary", "alex", "sam", "lee", "kim", "chris", "pat", "jordan", "taylor", "max", "ana"]


def _ip(rng, pool, n):
    return [f"{pool}.{rng.integers(0, 256)}.{rng.integers(0, 256)}" for _ in range(n)]


CATEGORY_WORDS = [[f"cat{c}_{k}" for k in range(12)] for c in range(30)]


def _text_legit(rng, cat):
    """Category copy (shared phrasing within a category) with personal variation."""
    words = list(rng.choice(CATEGORY_WORDS[cat], 8, replace=False))
    words += [VOCAB_LEGIT[i] for i in rng.choice(len(VOCAB_LEGIT), 4, replace=False)]
    return " ".join(words)


def _text_scam(rng, template):
    words = list(SCAM_TEMPLATES[template])
    for _ in range(2):                                        # small perturbations
        words[rng.integers(0, len(words))] = VOCAB_LEGIT[rng.integers(0, 3000)]
    return " ".join(words)


def generate(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    # --- independent legit advertisers
    for i in range(LEGIT_N):
        cat = rng.integers(0, 30)
        rows.append({
            "kind": "legit", "group": f"legit_{i}",
            "email": (f"{rng.choice(FIRST)}{rng.integers(1, 10 ** 6)}" + ("+ads" if rng.random() < 0.06 else "")
                      + f"@{rng.choice(['gmail.com', 'outlook.com', f'biz{rng.integers(1, 10**5)}.com'])}"),
            "card": f"c{rng.integers(1, 10 ** 9)}",
            "card_type": rng.choice(["credit", "debit", "prepaid"], p=[0.7, 0.26, 0.04]),
            "device": f"d{rng.integers(1, 10 ** 9)}",
            "ip": _ip(rng, f"{rng.integers(60, 100)}", 1)[0] if rng.random() > 0.08 else f"100.64.{rng.integers(0, 40)}.1",  # CGNAT collisions
            "created_day": float(rng.uniform(0, 365)),
            # a quarter are new small businesses with brand-new domains
            "domain_age_days": float(rng.lognormal(3.6, 1.2) if rng.random() < 0.25 else rng.lognormal(6.0, 1.0)),
            "spend_week1": float(rng.lognormal(4.3, 1.3)),
            "text": _text_legit(rng, cat),
        })
    # --- agencies (benign, clustered)
    for a in range(N_AGENCIES):
        n = int(rng.integers(8, 81))
        block = f"{rng.integers(100, 200)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}"
        cards = [f"ac{a}_{k}" for k in range(rng.integers(1, 3))]
        devices = [f"ad{a}_{k}" for k in range(rng.integers(2, 8))]
        start = rng.uniform(0, 330)
        for j in range(n):
            cat = rng.integers(0, 30)
            rows.append({
                "kind": "agency", "group": f"agency_{a}",
                "email": f"client{j}@agency{a}.com",
                "card": rng.choice(cards), "card_type": "corporate", "device": rng.choice(devices),
                "ip": f"{block}.{rng.integers(1, 20)}",
                "created_day": float(start + rng.exponential(20)),
                "domain_age_days": float(rng.lognormal(5.5, 1.2)),
                "spend_week1": float(rng.lognormal(5.0, 1.0)),
                "text": _text_legit(rng, cat),
            })
    # --- fraud operators and rings
    ops = []
    for o in range(N_OPERATORS):
        ops.append({"blocks": [f"{rng.integers(200, 224)}.{rng.integers(0, 256)}.{rng.integers(0, 256)}" for _ in range(3)],
                    "cards": [f"fc{o}_{k}" for k in range(6)], "devices": [f"fd{o}_{k}" for k in range(10)],
                    "stem": rng.choice(FIRST) + str(rng.integers(10, 99)), "template": int(rng.integers(0, len(SCAM_TEMPLATES)))})
    for r in range(N_RINGS):
        o = int(rng.integers(0, N_OPERATORS))
        op = ops[o]
        n = int(rng.integers(5, 41))
        cards = list(rng.choice(op["cards"], rng.integers(1, 4), replace=False))
        devices = list(rng.choice(op["devices"], rng.integers(2, 6), replace=False))
        block = rng.choice(op["blocks"])
        start = rng.uniform(0, 330)
        slow = rng.random() < 0.3                     # "low and slow" rings: spread out, aged domains
        for j in range(n):
            stem = op["stem"]
            dotted = ".".join(stem[k] for k in range(len(stem))) if rng.random() < 0.3 else stem
            rows.append({
                "kind": "fraud", "group": f"ring_{r}", "operator": o,
                "email": f"{dotted}+{r}x{j}@gmail.com" if rng.random() < 0.6 else f"{stem}{r}{j}@gmail.com",
                "card": rng.choice(cards), "card_type": "prepaid" if rng.random() < 0.5 else "credit",
                "device": rng.choice(devices),
                "ip": f"{block}.{rng.integers(1, 30)}",
                "created_day": float(start + (rng.uniform(0, 30) if slow else rng.exponential(1.0))),
                "domain_age_days": float(rng.lognormal(5.0, 1.2) if slow else rng.lognormal(3.8, 1.3)),
                "spend_week1": float(rng.lognormal(4.8, 1.2)),
                "text": _text_scam(rng, op["template"]) if rng.random() < 0.85 else _text_legit(rng, int(rng.integers(0, 30))),
            })
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "account_id", [f"A{i:06d}" for i in range(len(df))])
    df["operator"] = df["operator"].fillna(-1).astype(int)
    df["is_fraud"] = (df.kind == "fraud").astype(int)
    return df
