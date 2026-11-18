"""
test_stage3_trace.py - the step-by-step Stage 3 account shown in app_nlp.py (run with:  python -m pytest -q)

The trace re-walks the pipeline's decision procedure and writes down each test with its numbers. These tests
check that it reproduces the pipeline's own decision for every test patient of a run, on both datasets,
including Case 2 and Case 3 patients later in the stream.
"""
import copy

import pytest

import config
from neurosym.data import load_kaggle, make_split
from neurosym.pipeline import NeuroSymbolicPipeline
from neurosym.stage3_trace import trace


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


# ----------------------------------------------------------------------------- Stage 3 trace
def test_trace_reproduces_every_decision(kaggle, trained):
    pipe0, split = trained
    pipe = copy.deepcopy(pipe0)
    pipe.reset_stream()
    seen = set()
    for i in split.test:
        before = copy.deepcopy(pipe.ontology)
        out = pipe.diagnose(kaggle.X[i], keep_record=True)
        t = trace(pipe, before, out["record"])
        assert t["outcome"] == out["outcome"], (i, t["conclusion"], out)
        if out["outcome"] != "new":
            assert t["disease"] == out["disease"]
        assert t["steps"][0]["key"] == "evidence" and t["steps"][-1]["key"] == "decision"
        seen.add(out["outcome"])
    assert seen == {"known", "new", "old_new"}               # all three cases were exercised



# ----------------------------------------------------------------------------- DDXPlus
@pytest.fixture(scope="module")
def ddx():
    from train import load
    return load("ddxplus")


@pytest.fixture(scope="module")
def ddx_trained(ddx):
    cfg = config.DATASETS["ddxplus"]
    split = make_split(ddx, cfg["n_novel"], 0)
    pipe = NeuroSymbolicPipeline(seed=0, **cfg["pipeline"]).fit(
        ddx.X[split.train], ddx.y[split.train], ddx.X[split.calib], ddx.y[split.calib],
        ddx.rules, ddx.features, split.known, demographic=[ddx.features[i] for i in ddx.demographic_idx])
    return pipe, split


def test_trace_reproduces_every_decision_ddxplus(ddx, ddx_trained):
    pipe0, split = ddx_trained
    pipe = copy.deepcopy(pipe0)
    pipe.reset_stream()
    for i in split.test:
        before = copy.deepcopy(pipe.ontology)
        out = pipe.diagnose(ddx.X[i], keep_record=True)
        t = trace(pipe, before, out["record"])
        assert t["outcome"] == out["outcome"]
        if out["outcome"] != "new":
            assert t["disease"] == out["disease"]
