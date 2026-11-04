"""
Entity resolution and graph features over shared account attributes.

Accounts are linked through shared attribute VALUES (card, device, IP /24 block, normalized
e-mail, near-duplicate ad text). Values shared by huge numbers of accounts (carrier NAT, very
common text) carry little evidence, so links are weighted by 1 / log(group size) and values above
MAX_GROUP are ignored. "Operator clusters" are connected components over strong links (card,
device, normalized e-mail stem); they stand in for the entity-resolution step.
"""
import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MAX_GROUP = 300
STRONG = ["card", "device", "email_stem"]
WEAK = ["ip_block", "text_sig"]


def normalize_email(e, er=True):
    """Entity-resolution style normalization: lower-case, drop +tags, gmail dots, trailing digits."""
    e = e.lower()
    if not er:
        return e
    local, dom = e.split("@", 1)
    local = local.split("+", 1)[0]
    if dom == "gmail.com":
        local = local.replace(".", "")
    local = re.sub(r"\d+$", "", re.sub(r"(\d+)[a-z]?\d*$", r"\1", local))
    return f"{local}@{dom}"


def email_stem(e, er=True):
    n = normalize_email(e, er)
    local, dom = n.split("@", 1)
    if dom in ("gmail.com", "outlook.com"):
        return f"{local}@{dom}" if er else n
    return dom if er else n                                    # company domains: the domain is the entity


def text_signature(t, k=3, bands=4):
    """MinHash-style near-duplicate signature: min hash of the token set under k seeded hashes,
    in one band (texts sharing >= ~80% tokens usually collide; paraphrasing breaks it)."""
    import zlib
    toks = set(t.split())
    mins = [min(zlib.crc32(f"{s}:{w}".encode()) for w in toks) for s in range(k)]
    return "-".join(map(str, mins))


def derive(df, er=True):
    d = df.copy()
    d["ip_block"] = d.ip.str.rsplit(".", n=1).str[0]
    d["email_stem"] = [email_stem(e, er) for e in d.email]
    d["text_sig"] = [text_signature(t) for t in d.text]
    return d


def group_stats(d, col, seed_bad=None):
    g = d.groupby(col)
    size = g[col].transform("size")
    span = g.created_day.transform("max") - g.created_day.transform("min")
    out = {f"{col}_size": size.clip(upper=MAX_GROUP).to_numpy(float),
           f"{col}_created_span": np.where(size > 1, span, 365.0)}
    if seed_bad is not None:
        bad = pd.Series(seed_bad, index=d.index).groupby(d[col]).transform("sum") - seed_bad
        out[f"{col}_known_bad_others"] = np.minimum(bad.to_numpy(float), 50)
    return out


def components(d, cols=STRONG):
    n = len(d)
    rows, cols_idx = [], []
    for c in cols:
        codes, uniq = pd.factorize(d[c])
        sizes = np.bincount(codes)
        ok = sizes[codes] <= MAX_GROUP
        # link each account to the first account holding the same value (a star per value)
        first = pd.Series(np.arange(n)).groupby(codes).transform("min").to_numpy()
        m = ok & (first != np.arange(n))
        rows.append(np.arange(n)[m])
        cols_idx.append(first[m])
    r, c = np.concatenate(rows), np.concatenate(cols_idx)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    return lab


def features(df, seed_bad=None, er=True):
    """Account features. seed_bad: 0/1 array of accounts KNOWN bad from past enforcement (training labels)."""
    d = derive(df, er)
    F = {"domain_age_log": np.log1p(d.domain_age_days.to_numpy()), "spend_week1_log": np.log1p(d.spend_week1.to_numpy()),
         "gmail": d.email.str.endswith("@gmail.com").to_numpy(float), "plus_tag": d.email.str.contains(r"\+").to_numpy(float),
         "card_prepaid": (d.card_type == "prepaid").to_numpy(float),
         "ip_residential": d.ip.str.split(".").str[0].astype(int).between(60, 99).to_numpy(float)}
    for c in STRONG + WEAK:
        F.update(group_stats(d, c, seed_bad))
    lab = components(d)
    comp = pd.Series(lab)
    csize = comp.map(comp.value_counts()).to_numpy(float)
    cd = pd.Series(d.created_day.to_numpy()).groupby(lab)
    cspan = (cd.transform("max") - cd.transform("min")).to_numpy()
    cdom = pd.Series(np.log1p(d.domain_age_days.to_numpy())).groupby(lab).transform("median").to_numpy()
    F.update({"cluster_size": np.minimum(csize, 500), "cluster_created_span": np.where(csize > 1, cspan, 365.0),
              "cluster_median_domain_age": cdom,
              "cluster_text_dup_share": pd.Series(d.text_sig.duplicated(keep=False).to_numpy(float)).groupby(lab).transform("mean").to_numpy()})
    if seed_bad is not None:
        kb = pd.Series(seed_bad).groupby(lab).transform("sum").to_numpy() - seed_bad
        F["cluster_known_bad_others"] = np.minimum(kb, 50)
    X = pd.DataFrame(F)
    return X, lab


ACCOUNT_ONLY = ["domain_age_log", "spend_week1_log", "gmail", "plus_tag"]
HARDENING_ONLY = ["card_prepaid", "ip_residential"]          # candidate signals added after red-team rounds
SPOOFABLE = ["ip_residential"]                                # a residential proxy costs $5: easy for an attacker to fake
