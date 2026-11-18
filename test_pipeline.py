"""
test_pipeline.py - automated tests (run with:  python -m pytest -q)
===================================================================
They check the data preparation against the dissertation (Section 3.2.3), the building blocks of each
stage on small hand-made examples, and the design rules of the pipeline (Cases 1-3, growth cap,
Stage 3 reading only the explanation record).
"""
import copy

import numpy as np
import pytest

import config
from neurosym.data import RuleItem, load_ddxplus, load_kaggle, make_split
from neurosym.evaluate import open_set_metrics
from neurosym.ghsom import GHSOM
from neurosym.gng import CappedGNG
from neurosym.ontology import Ontology
from neurosym.pipeline import NeuroSymbolicPipeline
from neurosym.xai import ConformalScorer, jaccard_dist, qe_attribution, symptom_differential


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def kaggle():
    return load_kaggle(config.DATA_DIR)


@pytest.fixture(scope="module")
def trained(kaggle):
    cfg = config.DATASETS["kaggle"]
    split = make_split(kaggle, cfg["n_novel"], 0)
    pipe = NeuroSymbolicPipeline(seed=0, **cfg["pipeline"]).fit(
        kaggle.X[split.train], kaggle.y[split.train], kaggle.X[split.calib], kaggle.y[split.calib],
        kaggle.rules, kaggle.features, split.known)
    return pipe, split


# ----------------------------------------------------------------------------- data (Section 3.2.3)
def test_kaggle_cleaning_matches_dissertation(kaggle):
    assert kaggle.X.shape == (304, 131)                       # 304 distinct patients, 131 symptoms
    assert len(kaggle.diseases) == 41
    assert "fluid_overload" in kaggle.features and not any(f.endswith(".1") for f in kaggle.features)
    assert round(kaggle.X.sum(1).mean(), 2) == 7.65


def test_kaggle_no_duplicate_rows(kaggle):
    assert len({r.tobytes() for r in kaggle.X}) == len(kaggle.X)


def test_ddxplus_cleaning_matches_dissertation():
    d = load_ddxplus(config.DATA_DIR)
    assert d.X.shape[0] == 1943 and len(d.diseases) == 49
    assert d.cleaning["exact_duplicates_removed"] == 270
    assert d.cleaning["identical_rows_removed"] == 13
    assert d.cleaning["never_firing_rule_items"] == 6


def test_split_keeps_novel_diseases_out_of_training(kaggle):
    s = make_split(kaggle, 4, 0)
    assert not set(kaggle.y[s.train]) & set(s.novel)
    assert not set(kaggle.y[s.calib]) & set(s.novel)
    assert set(s.novel) <= set(kaggle.y[s.test])


# ----------------------------------------------------------------------------- Stage 1
def test_ghsom_is_deterministic(kaggle):
    a = GHSOM(seed=3).fit(kaggle.X[:120], kaggle.y[:120]).transform(kaggle.X[:120])[0]
    b = GHSOM(seed=3).fit(kaggle.X[:120], kaggle.y[:120]).transform(kaggle.X[:120])[0]
    assert (a == b).all()


def test_ghsom_grows_a_hierarchy(trained):
    pipe, _ = trained
    s = pipe.ghsom.summary()
    assert s["max_depth"] >= 2 and s["leaf_clusters"] > 1


def test_gng_starts_with_ghsom_node_count(trained):
    pipe, _ = trained
    assert pipe.gng.n_init == len(pipe.ghsom.leaf_key)


def test_gng_growth_cap_is_one():
    g = CappedGNG(cap=1)
    g.W = np.eye(3)
    g.label, g.kind = ["a", "b", "c"], ["known"] * 3
    g.n_assigned, g.E, g.purity, g.edges, g.n_init = np.zeros(3), np.zeros(3), np.ones(3), {}, 3
    q = g.propose(np.array([1.0, 1.0, 0.0]))
    assert q == 3 and g.cap == 0
    assert g.propose(np.array([0.0, 1.0, 1.0])) is None          # no second node while one is pending
    g.commit(q, "NEW-1")
    assert g.cap == 1 and g.kind[q] == "new"                     # node kept, cap back to 1
    q2 = g.propose(np.array([0.0, 1.0, 1.0]))
    g.reject(q2)
    assert len(g.W) == 4 and g.cap == 1                          # rejected node removed


# ----------------------------------------------------------------------------- Stage 2
def test_qe_attribution_is_exact():
    x = np.array([1.0, 0.0, 1.0, 0.0])
    w = np.array([0.9, 0.8, 0.1, 0.0])
    a = qe_attribution(x, w, ["a", "b", "c", "d"])
    assert set(a["present_unusual"]) == {"c"} and set(a["expected_absent"]) == {"b"}
    assert abs(sum(t["share"] for t in a["top"]) - 1.0) < 1e-9


def test_weighted_jaccard_equals_jaccard_for_binary():
    x = np.array([1, 1, 0, 0], bool)
    w = np.array([1.0, 0.0, 1.0, 0.0])
    assert abs(jaccard_dist(x, w)[0, 0] - (1 - 1 / 3)) < 1e-12


def test_conformal_pvalue():
    c = ConformalScorer(min_mondrian=1).fit(np.arange(1.0, 10.0), np.zeros(9))
    assert c.pvalue(100.0, 0) == pytest.approx(1 / 10)           # larger than every calibration score
    assert c.pvalue(0.0, 0) == pytest.approx(1.0)


def test_symptom_differential():
    d = symptom_differential(np.array([1, 0, 1.0]), np.array([0.9, 0.7, 0.2]), ["a", "b", "c"])
    assert d == {"shared": ["a"], "lacks": ["b"], "excess": ["c"], "hamming": 2}


def test_conformal_test_rejects_about_alpha_of_known_patients(trained, kaggle):
    pipe, split = trained
    known = [i for i in split.test if kaggle.y[i] in split.known]
    rejected = np.mean([pipe.explain(kaggle.X[i])["path"] != "fit" for i in known])
    assert rejected <= 0.15


# ----------------------------------------------------------------------------- Stage 3
def _tiny_ontology():
    names = ["fever", "cough", "rash", "itch"]
    rules = {"Flu": [RuleItem(frozenset({0}), "symptom", "fever"), RuleItem(frozenset({1}), "symptom", "cough")],
             "Eczema": [RuleItem(frozenset({2}), "symptom", "rash"), RuleItem(frozenset({3}), "symptom", "itch")]}
    o = Ontology(rules, names, ["Flu", "Eczema"])
    o.theta = {"Flu": 0.5, "Eczema": 0.5}
    return o


def test_ontology_score_is_harmonic_mean():
    r = _tiny_ontology().score({"fever", "rash"}, "Flu")
    assert r["coverage"] == 0.5 and r["explained"] == 0.5 and r["score"] == pytest.approx(0.5)


def test_only_the_top_ranked_disease_can_be_accepted():
    o = _tiny_ontology()
    best, _ = o.best_consistent({"fever", "cough"})
    assert best["disease"] == "Flu"
    o.theta["Flu"] = 1.1                                        # top-ranked fails its own threshold
    best, ranked = o.best_consistent({"fever", "cough"})
    assert best is None and ranked[0]["disease"] == "Flu"


def test_registered_profile_confirmation():
    o = _tiny_ontology()
    o.register("NEW-1", ["rash", "itch", "fever"], 9)
    assert o.confirm_registered({"rash", "itch", "fever"}, "NEW-1")["confirmed"]
    assert not o.confirm_registered({"cough"}, "NEW-1")["confirmed"]


def test_stage3_does_not_read_the_raw_vector(trained, kaggle):
    pipe, split = trained
    rec = pipe.explain(kaggle.X[split.test[0]])
    assert "findings" not in rec
    present = {pipe.names[j] for j in np.where(kaggle.X[split.test[0]] > 0.5)[0]}
    assert pipe.evidence(rec) == present                        # rebuilt from explanation fields only


# ----------------------------------------------------------------------------- Cases 1-3
def test_case1_case2_case3_sequence(trained, kaggle):
    """A pure known patient takes Case 1; a held-out disease's first patient is registered (Case 2)
    and a later patient of the same disease is recognised (Case 3) with the GNG unchanged."""
    pipe, split = trained
    p = copy.deepcopy(pipe)
    p.reset_stream()
    outs = [p.diagnose(kaggle.X[i]) for i in split.test if kaggle.y[i] == "Hyperthyroidism"]
    assert outs[0]["case"] == "Case 2" and outs[0]["outcome"] == "new"
    n_nodes = len(p.gng.W)
    assert any(o["case"] == "Case 3" for o in outs[1:])
    assert len(p.gng.W) == n_nodes                               # Case 3 adds no node
    first_known = next(i for i in split.test if kaggle.y[i] in split.known)
    out = p.diagnose(kaggle.X[first_known])
    assert out["outcome"] in {"known", "new", "old_new"}


def test_returning_new_disease_in_a_mixed_cluster_is_recognised(trained, kaggle):
    """A patient of a new disease who lands in a low-purity GHSOM cluster is checked against the new
    diseases registered earlier before another one is registered (no duplicate new diseases)."""
    pipe, _ = trained
    p = copy.deepcopy(pipe)
    p.reset_stream()
    x = np.zeros(len(p.names))
    for s in ["headache", "vomiting", "altered_sensorium", "weakness_of_one_body_side"]:
        x[p.names.index(s)] = 1
    first, second = p.diagnose(x, keep_record=True), p.diagnose(x, keep_record=True)
    assert first["case"] == "Case 2" and first["record"]["path"] == "fit"
    assert second["case"] == "Case 3" and second["disease"] == first["disease"]
    assert len(p.ontology.registry) == 1


def test_reset_stream_restores_state(trained, kaggle):
    pipe, split = trained
    p = copy.deepcopy(pipe)
    p.reset_stream()
    n0 = len(p.gng.W)
    for i in split.test[:40]:
        p.diagnose(kaggle.X[i])
    p.reset_stream()
    assert len(p.gng.W) == n0 and p.ontology.registry == {} and p.gng.cap == 1


# ----------------------------------------------------------------------------- metrics
def test_open_set_metrics():
    m = open_set_metrics(["A", "A", "N", "N"], ["A", "NEW-1", "NEW-1", "B"], novel=["N"], known=["A", "B"])
    assert m["known_accuracy"] == 0.5 and m["novel_detection_rate"] == 0.5
    assert m["open_set_accuracy"] == 0.5 and m["new_groups_created"] == 1
