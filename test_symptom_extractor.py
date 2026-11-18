"""Tests for the plain-English symptom extractor used by app_nlp.py  (python -m pytest)."""
import numpy as np
import pytest

from neurosym.symptom_extractor import SymptomExtractor
from neurosym.symptom_lexicon import LEXICON
from train import load


@pytest.fixture(scope="module")
def data():
    return load("kaggle")


@pytest.fixture(scope="module")
def ex(data):
    return SymptomExtractor(list(data.features))


def test_every_symptom_has_phrases(data):
    assert set(LEXICON) == set(data.features)


def test_every_phrase_finds_its_own_symptom(ex):
    for name, phrases in LEXICON.items():
        for ph in phrases:
            assert ex.extract(f"I have been having {ph}.").present == [name], ph


def test_negation(ex):
    r = ex.extract("I have no fever, cough or rash. I have a headache, but the vomiting has stopped.")
    assert r.present == ["headache"]
    assert set(r.absent) == {"high_fever", "cough", "skin_rash", "vomiting"}
    r = ex.extract("I'm not sleeping well and I feel sad all the time.")
    assert r.present == ["depression"]


def test_everyday_words_word_order_and_spelling(ex):
    r = ex.extract("Doc, I have running stomach, my throat is scratchy and I've had diarhea since Monday.")
    assert set(r.present) == {"diarrhoea", "throat_irritation"}
    assert ex.extract("I have a mild fever").present == ["mild_fever"]          # longest phrase wins
    assert ex.extract("hip pain, swollen joints and knee pain").present == ["hip_joint_pain", "swelling_joints", "knee_pain"]


def test_complaints_written_from_every_patient_are_read_back_exactly(ex, data):
    for i in range(len(data.y)):
        sy = [data.features[j] for j in np.where(data.X[i] > 0.5)[0]]
        assert set(ex.extract(ex.complaint_from(sy, seed=i)).present) == set(sy), i
