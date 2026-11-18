"""symptom_extractor.py - turn a patient's complaint in plain English into symptom columns.

A small, transparent, rule-based NLP step that runs before Stage 1. It needs no internet and no
downloaded model, and every symptom it finds can be traced to the words that produced it.

    from neurosym.symptom_extractor import SymptomExtractor
    ex = SymptomExtractor(pipe.names)
    res = ex.extract("I've had a high fever and headache since Monday, but no cough.")
    res.present   -> ['high_fever', 'headache']
    res.absent    -> ['cough']                       (mentioned, but negated)

How it works
  1. Normalise: lower case, expand contractions ("don't" -> "do not"), strip punctuation,
     split into clauses at full stops, semicolons, "but", "however" ...
  2. Reduce each word to a crude stem ("aching" -> "ach", "joints" -> "joint") so that word forms match.
  3. Look for every phrase of the lexicon (symptom_lexicon.py, plus each column name itself). The words of
     a phrase must appear in order; up to two extra words may sit between them ("pain in my LOWER back"),
     and misspelt words longer than four letters still match if they are close (difflib ratio >= 0.84).
  4. Where matches overlap, the longest phrase wins ("mild fever" beats "fever").
  5. Negation (NegEx-style): a match is negated if a trigger such as "no", "not", "never", "without",
     "denies" appears earlier in the same clause, at most six words before it, or if it is followed by
     "is gone", "has stopped", "resolved".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .symptom_lexicon import LEXICON

CONTRACTIONS = {
    "can't": "can not", "cannot": "can not", "won't": "will not", "n't": " not", "i'm": "i am", "i've": "i have",
    "i'd": "i had", "it's": "it is", "there's": "there is", "that's": "that is", "i'll": "i will",
}
CLAUSE_BREAK = re.compile(r"[.;!?\n]+|\bbut\b|\bhowever\b|\balthough\b|\bthough\b|\bexcept\b|\byet\b")
PRE_NEG = [("no",), ("not",), ("never",), ("without",), ("deny",), ("denies",), ("denied",), ("nor",),
           ("neither",), ("free", "of"), ("absence", "of"), ("negative", "for"), ("no", "longer"), ("none",)]
# "I am not sleeping well AND I feel sad": a new subject after "and" ends the negation's reach
SCOPE_END = {"i", "my", "it", "he", "she", "they", "there", "the", "his", "her", "their"}
POST_NEG = [("gone",), ("stopped",), ("resolved",), ("went", "away"), ("cleared",)]
NEG_WINDOW = 6
BLOCK = {",", "and", "or", "plus"}
GAP = 2
FUZZY = 0.84
# words that carry no meaning for matching (removed from phrases and text alike)
STOP = {"a", "an", "the", "my", "i", "me", "am", "is", "are", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "of", "to", "it", "its", "this", "that", "so", "very", "really", "quite", "some", "kind",
        "bit", "lot", "lots", "feel", "feeling", "felt", "get", "getting", "got", "keep", "keeps", "also", "too",
        "just", "like", "with", "since", "for", "from", "days", "day", "week", "weeks", "now", "all",
        "always", "time", "times", "sometimes", "often", "little", "bit", "there", "on", "at", "by", "in", "into",
        "around", "over", "up", "out"}
# ...but these stop words matter inside some phrases, so they are kept when a phrase needs them
KEEP_IN_PHRASE = {"in", "on", "around", "over", "up", "out", "all", "always", "keep", "feel", "time"}


def _norm(text: str) -> str:
    t = text.lower().replace("’", "'").replace("`", "'")
    for k in sorted(CONTRACTIONS, key=len, reverse=True):
        t = t.replace(k, " " + CONTRACTIONS[k] + " ") if k == "n't" else re.sub(rf"\b{re.escape(k)}\b", CONTRACTIONS[k], t)
    return t


def _stem(w: str) -> str:
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    for suf, rep, minlen in (("ies", "y", 5), ("ing", "", 6), ("ed", "", 5), ("es", "", 5), ("s", "", 4)):
        if w.endswith(suf) and len(w) >= minlen and not w.endswith("ss"):
            w = w[: len(w) - len(suf)] + rep
            break
    if len(w) > 3 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]                      # "ache" and "aching" both become "ach"
    return w


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9']+", text)]


@dataclass
class Finding:
    symptom: str          # column name
    phrase: str           # lexicon phrase that matched
    heard: str            # the patient's own words that matched
    negated: bool
    trigger: str = ""     # negation word, if negated
    pos: tuple = (0, 0)   # (clause, word) where it was said


@dataclass
class Extraction:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def present(self) -> list[str]:
        neg = {f.symptom for f in self.findings if f.negated}
        out = []
        for f in self.findings:
            if not f.negated and f.symptom not in out and f.symptom not in neg:
                out.append(f.symptom)
        return out

    @property
    def absent(self) -> list[str]:
        pres = set(self.present)
        return list(dict.fromkeys(f.symptom for f in self.findings if f.negated and f.symptom not in pres))


class SymptomExtractor:
    def __init__(self, names: list[str], lexicon: dict | None = None):
        lex = lexicon if lexicon is not None else LEXICON
        self.names = list(names)
        self.phrases: list[tuple[tuple[str, ...], str, str]] = []    # (stemmed tokens, symptom, phrase)
        seen = set()
        for name in self.names:
            cands = list(lex.get(name, [])) + [name.replace("_", " ").replace("(", " ").replace(")", " ")]
            for ph in cands:
                toks = tuple(t for t in _tokens(_norm(ph)) if t not in STOP or t in KEEP_IN_PHRASE)
                if toks and (toks, name) not in seen:
                    seen.add((toks, name))
                    self.phrases.append((toks, name, ph))
        # longest phrases first, so that ties in span resolution favour the more specific phrase
        self.phrases.sort(key=lambda p: -len(p[0]))

    # ------------------------------------------------------------------ matching
    @staticmethod
    def _same(a: str, b: str) -> bool:
        if a == b:
            return True
        if len(a) > 4 and len(b) > 4 and a[0] == b[0]:
            return SequenceMatcher(None, a, b).ratio() >= FUZZY
        return False

    def _match(self, toks: list[str], pat: tuple[str, ...], _rev: bool = False):
        """All (start, end) spans where pat occurs in order, with at most GAP words between its words."""
        spans = []
        for s in range(len(toks)):
            if not self._same(toks[s], pat[0]):
                continue
            pos, ok, gaps = s, True, 0
            for p in pat[1:]:
                nxt = None
                for k in range(pos + 1, min(pos + 2 + GAP - gaps, len(toks))):
                    if self._same(toks[k], p):
                        nxt = k
                        break
                    if toks[k] in BLOCK:          # never reach across a list ("hip pain, swollen joints")
                        break
                if nxt is None:
                    ok = False
                    break
                gaps += nxt - pos - 1             # at most GAP extra words in the whole phrase
                pos = nxt
            if ok:
                spans.append((s, pos + 1))
        if len(pat) == 2 and not _rev:  # two-word phrases also match the other way round:
            for s, e in self._match(toks, (pat[1], pat[0]), True):
                spans.append((s, e))    # "scratchy throat" = "throat is scratchy"
        return spans

    def extract(self, text: str) -> Extraction:
        res = Extraction(text=text)
        for ci, clause in enumerate(CLAUSE_BREAK.split(_norm(text))):
            raw = re.findall(r"[a-z0-9']+|,", clause)
            stems = [_stem(w) for w in raw]
            keep = [i for i, t in enumerate(stems) if t not in STOP or t in KEEP_IN_PHRASE]
            toks = [stems[i] for i in keep]
            if not toks:
                continue
            cands = []
            for pat, name, ph in self.phrases:
                for s, e in self._match(toks, pat):
                    cands.append((len(pat), -(e - s), s, e, name, ph))
            cands.sort(key=lambda c: (-c[0], -c[1], c[2]))
            used = set()
            for n, _, s, e, name, ph in cands:
                span = set(range(s, e))
                if span & used:
                    continue
                used |= span
                rs, re_ = keep[s], keep[e - 1] + 1          # back to positions in the raw clause
                neg, trig = self._negation(raw, rs, re_)
                res.findings.append(Finding(name, ph, " ".join(raw[rs:re_]), neg, trig, (ci, rs)))
        res.findings.sort(key=lambda f: f.pos)       # the order in which the patient said them
        return res

    @staticmethod
    def _negation(raw: list[str], s: int, e: int):
        before = raw[max(0, s - NEG_WINDOW):s]
        for i in range(len(before)):
            for trig in PRE_NEG:
                if tuple(before[i:i + len(trig)]) == trig:
                    between = before[i + len(trig):]
                    if any(a == "and" and b in SCOPE_END for a, b in zip(between, between[1:])):
                        continue
                    return True, " ".join(trig)
        after = raw[e:e + 4]
        for i in range(len(after)):
            for trig in POST_NEG:
                if tuple(after[i:i + len(trig)]) == trig:
                    return True, " ".join(trig)
        return False, ""

    # ------------------------------------------------------------------ the reverse: an example complaint
    def complaint_from(self, symptoms: list[str], seed: int = 0) -> str:
        """Write a plain-English complaint for a list of symptom columns (used to make demo patients)."""
        import random
        rnd = random.Random(seed)
        lay = [LEXICON.get(s, [s.replace("_", " ")])[0] for s in symptoms]
        rnd.shuffle(lay)
        openers = ["For the past few days I have been having", "Doctor, I have been having",
                   "Since last week I have had", "Good morning doctor. I have been suffering from"]
        def join(xs):
            return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
        if len(lay) <= 3:
            return f"{rnd.choice(openers)} {join(lay)}."
        k = (len(lay) + 1) // 2
        tails = ["I have also noticed", "On top of that there is", "There is also", "I also have"]
        return f"{rnd.choice(openers)} {join(lay[:k])}. {rnd.choice(tails)} {join(lay[k:])}."
