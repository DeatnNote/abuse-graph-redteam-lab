import numpy as np

from agl import redteam as RT
from agl.graph import components, derive, email_stem, features


def test_email_entity_resolution():
    assert email_stem("J.o.h.n42+7x3@gmail.com") == email_stem("john42+99@gmail.com")
    assert email_stem("client3@agency7.com") == email_stem("client9@agency7.com") == "agency7.com"
    assert email_stem("J.o.h.n42+7x3@gmail.com", er=False) != email_stem("john42+99@gmail.com", er=False)


def test_generation_and_ground_truth(world):
    df, _, _ = world
    assert set(df.kind) == {"legit", "agency", "fraud"}
    assert (df.is_fraud == (df.kind == "fraud")).all() and df.account_id.is_unique


def test_rings_are_mostly_one_cluster(world):
    df, _, _ = world
    lab = components(derive(df))
    fr = df.kind.to_numpy() == "fraud"
    # most accounts of a ring share one operator cluster
    share = [np.bincount(lab[(df.group == g).to_numpy()]).max() / (df.group == g).sum() for g in df.group[fr].unique()]
    assert np.median(share) > 0.8


def test_known_bad_features_never_use_test_labels(world):
    df, tr, det = world
    assert det.seed_bad[~tr].sum() == 0
    X, _ = features(df, det.seed_bad)
    assert (X["card_known_bad_others"] >= 0).all()


def test_detector_threshold_controls_false_positives(world):
    df, tr, det = world
    r = det.evaluate(df, ~tr)
    assert r["false_positive_rate"] < 0.03 and r["ring_recall"] > 0.8


def test_attack_never_touches_benign_accounts_and_costs_add_up(world):
    df, tr, det = world
    rings = sorted(df.loc[~tr & (df.kind == "fraud"), "group"].unique())[:3]
    d2, traj = RT.attack(df, det, rings, max_rounds=3)
    benign = (df.kind != "fraud").to_numpy()
    assert d2[benign].equals(df[benign])
    for r, t in traj.groupby("ring"):
        acts = t.action.dropna()
        assert np.isclose(t.cost_per_account.max(), sum(RT.ACTIONS[a][0] for a in acts))
        assert acts.is_unique
    curve = RT.evasion_curve(traj)
    assert all(np.diff(list(curve.values())) >= 0)


def test_mcp_tools_mask_raw_attributes(world):
    from mcp_server.graph_tools import GraphStore
    df, tr, det = world
    s = GraphStore(df, det)
    acc = df.loc[df.kind == "fraud", "account_id"].iloc[0]
    out = s.cluster_summary(acc)
    raw = set(df.card) | set(df.device) | set(df.email)
    assert not any(v["value"] in raw for vals in out["shared_attributes"].values() for v in vals)
    assert s.account_risk(acc)["cluster_id"] == out["cluster_id"]
    assert len(s.top_clusters(5)) == 5
