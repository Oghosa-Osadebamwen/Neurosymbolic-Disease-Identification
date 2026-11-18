"""
app_nlp.py - presentation app with a plain-English patient consultation in front of the pipeline
================================================================================================

    python -m streamlit run app_nlp.py

Same pipeline and models as app.py (which is unchanged). The difference: the patient describes the
complaint in their own words, a symptom extractor (neurosym/symptom_extractor.py) picks out the
symptoms, the clinician can correct the list, and only then are the symptoms passed to Stage 1.

Tabs
  1. Consultation   ONE screen for seeing patients and finding new diseases: call the next test patient from the
                    waiting room (ordinary or outbreak day), type a walk-in patient's own words, or choose the
                    symptoms directly. The symptoms go through Stage 1 -> 2 -> 3; nobody tells the system the
                    disease; unfamiliar diseases should be registered (Case 2) and recognised later (Case 3).
  2. Results        the experiment results (from results/*.csv)
  3. How it works   the pipeline and the three cases

Uses the trained models in models/ (run train.py first, or press "Train this model" in the sidebar).
"""
from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

os.chdir(Path(__file__).resolve().parent)          # so relative paths work wherever it is launched from

import config                                       # noqa: E402
from train import load, train_one                   # noqa: E402
from neurosym.symptom_extractor import SymptomExtractor   # noqa: E402

# ----------------------------------------------------------------------------- oxblood palette
OX = {"950": "#2A0709", "900": "#3B0A0E", "800": "#4A0E12", "700": "#6B1B20", "600": "#7A1F24",
      "500": "#9A3A3F", "400": "#B9676B", "300": "#D29A9C", "200": "#E8C9CA", "100": "#F3E6E5",
      "50": "#FBF7F6"}
METHOD_COLORS = {"Symbolic-alone": OX["200"], "Neural-alone (GHSOM+GNG)": OX["400"],
                 "Neural-alone, strict GNG": OX["600"], "Neuro-symbolic (full)": OX["900"]}
OUTCOME = {"known": ("Known disease", OX["700"], "✔"),
           "old_new": ("Old disease (added earlier)", OX["500"], "↺"),
           "new": ("New disease", OX["900"], "✚")}

st.set_page_config(page_title="GHSOM + GNG Diagnosis: consultation", page_icon="🩺", layout="wide")
st.markdown(f"""
<style>
  .block-container {{padding-top: 2rem; max-width: 1350px;}}
  h1, h2, h3 {{color: {OX['800']};}}
  .stage {{background: white; border: 1px solid {OX['200']}; border-top: 6px solid {OX['600']};
           border-radius: 10px; padding: 14px 16px; height: 100%;}}
  .stage h4 {{margin: 0 0 6px 0; color: {OX['700']}; font-size: 0.95rem; letter-spacing: .04em;
              text-transform: uppercase;}}
  .stage .big {{font-size: 1.15rem; font-weight: 650; color: {OX['900']}; margin-bottom: 6px;}}
  .stage .muted {{color: #6d5557; font-size: .88rem;}}
  .pill {{display:inline-block; padding: 2px 9px; margin: 2px 3px 2px 0; border-radius: 999px;
          font-size: .82rem; background: {OX['100']}; color: {OX['800']}; border: 1px solid {OX['200']};}}
  .pill.dark {{background: {OX['700']}; color: white; border-color: {OX['700']};}}
  .pill.ghost {{background: white; color: {OX['600']}; border: 1px dashed {OX['400']};}}
  .verdict {{border-radius: 12px; padding: 18px 22px; color: white; margin: 14px 0 6px 0;}}
  .verdict .lab {{font-size: .85rem; opacity: .85; text-transform: uppercase; letter-spacing: .08em;}}
  .verdict .dis {{font-size: 1.7rem; font-weight: 700;}}
  .verdict .why {{font-size: .9rem; opacity: .92; margin-top: 4px;}}
  .arrow {{text-align:center; font-size: 1.6rem; color: {OX['400']}; padding-top: 70px;}}
  .hero {{background: linear-gradient(120deg, {OX['900']}, {OX['600']}); color: white; border-radius: 14px;
          padding: 20px 26px; margin-bottom: 14px;}}
  .hero h1 {{color: white; margin: 0; font-size: 1.75rem;}}
  .hero p {{margin: 4px 0 0 0; opacity: .9;}}
  div[data-testid="stMetricValue"] {{color: {OX['800']};}}
  .said {{background: white; border-left: 5px solid {OX['600']}; border-radius: 8px; padding: 12px 16px;
          font-size: 1.02rem; color: #2b1d1e; margin: 6px 0 10px 0;}}
  .said mark {{background: {OX['200']}; color: {OX['900']}; padding: 0 3px; border-radius: 4px;}}
  .said mark.neg {{background: white; color: {OX['500']}; text-decoration: line-through; border: 1px dashed {OX['400']};}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- loading
def model_path(ds, seed):
    return Path(config.MODEL_DIR) / f"{ds}_seed{seed}.pkl"


@st.cache_resource(show_spinner="Loading trained model…")
def load_model(ds, seed, mtime):
    blob = pickle.load(open(model_path(ds, seed), "rb"))
    return blob["pipeline"], blob["split"], blob["info"]


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset(ds):
    return load(ds)


@st.cache_data
def feature_labels(ds):
    if ds == "ddxplus":
        d = pd.read_csv(Path(config.DATA_DIR) / "rules" / "ddxplus_feature_descriptions.csv")
        return dict(zip(d.feature, d.description))
    return {}


def pretty(ds, name):
    lab = feature_labels(ds).get(name)
    return f"{lab}  ({name})" if lab else name.replace("_", " ")


def short(ds, name):
    lab = feature_labels(ds).get(name)
    if lab:
        return lab if len(lab) < 70 else lab[:67] + "…"
    return name.replace("_", " ")


def session_pipe(key, base):
    """Each browser session gets its own copy, so the GNG and new-disease registry carry over
    between patients in that session (needed for Case 3) without touching the cached model."""
    if key not in st.session_state:
        p = copy.deepcopy(base)
        p.reset_stream()
        st.session_state[key] = p
        st.session_state[key + "_hist"] = []
    return st.session_state[key]


# ----------------------------------------------------------------------------- rendering
def pills(items, cls="", limit=12, ds="kaggle"):
    if not items:
        return "<span class='muted'>none</span>"
    more = f"<span class='pill ghost'>+{len(items) - limit} more</span>" if len(items) > limit else ""
    return "".join(f"<span class='pill {cls}'>{short(ds, i)}</span>" for i in items[:limit]) + more


def render(out, pipe, ds, true_label=None):
    rec = out["record"]
    c1, a1, c2, a2, c3 = st.columns([10, 1, 10, 1, 10])
    # ---------- Stage 1
    if rec["path"] == "fit":
        s1 = (f"<div class='big'>GHSOM cluster {rec['cluster_code']}</div>"
              f"<div>Trained on: <b>{rec['cluster_disease']}</b> · purity {rec['cluster_purity']:.2f}</div>"
              f"<div style='margin-top:8px'>Error within cluster range? <b>Yes</b></div>"
              f"<div class='muted'>conformal p = {rec['p_value']:.3f} &gt; α = {pipe.alpha_fit}</div>"
              f"<div class='muted' style='margin-top:8px'>Patient stays on the GHSOM path.</div>")
    else:
        if rec["proposed_node"] is not None:
            gng = (f"GNG: <b>no node accepts X</b> → adds node #{rec['proposed_node']} "
                   f"(cap 1 → 0)")
        else:
            gng = "GNG: accepted by " + ", ".join(
                f"<b>#{c['node']}</b> {c['disease']} (p={c['p']:.2f})" for c in rec["conformal_set"][:3])
        s1 = (f"<div class='big'>Outlier → GNG</div>"
              f"<div>Nearest GHSOM cluster {rec['ghsom_cluster']}: error <b>out of range</b></div>"
              f"<div class='muted'>conformal p = {rec['ghsom_p_value']:.3f} ≤ α = {pipe.alpha_fit}</div>"
              f"<div style='margin-top:8px'>{gng}</div>")
    c1.markdown(f"<div class='stage'><h4>Stage 1 · Neural</h4>{s1}</div>", unsafe_allow_html=True)
    a1.markdown("<div class='arrow'>➜</div>", unsafe_allow_html=True)

    # ---------- Stage 2
    if rec["path"] == "fit":
        a = rec["qe_attribution"]
        s2 = (f"<div class='big'>Fit explanation</div>"
              f"<div class='muted'>Quantization-error attribution: how X differs from the cluster</div>"
              f"<div><b>Present but unusual:</b><br>{pills(a['present_unusual'], ds=ds)}</div>"
              f"<div style='margin-top:4px'><b>Expected but absent:</b><br>{pills(a['expected_absent'], 'ghost', ds=ds)}</div>"
              f"<div class='muted' style='margin-top:8px'>Shallow CART rule (for readers only, "
              f"fidelity {rec['cart']['fidelity']:.2f}, agrees with GHSOM: "
              f"{'yes' if rec['cart_agrees'] else 'no'})</div>")
    else:
        rows = ""
        for nb in rec["nearest"][:3]:   # same three nodes Stage 1 lists
            d = nb["differential"]
            tag = "registered new disease" if nb["kind"] == "new" else nb["disease"]
            rows += (f"<div style='margin-top:6px'><b>vs node #{nb['node']}</b> ({tag}) · "
                     f"Hamming {d['hamming']} · p={nb['p']:.2f}<br>"
                     f"<span class='muted'>shares {len(d['shared'])} · lacks</span> {pills(d['lacks'], 'ghost', 5, ds)}"
                     f"<span class='muted'> · extra</span> {pills(d['excess'], '', 5, ds)}</div>")
        cs = rec["conformal_set"]
        s2 = (f"<div class='big'>Proximity explanation</div>"
              f"<div>Conformal set: <b>{'empty → likely new' if not cs else str(len(cs)) + ' node(s)'}</b></div>"
              f"<div class='muted'>Symptom differential vs the nearest GNG nodes</div>{rows}")
    c2.markdown(f"<div class='stage'><h4>Stage 2 · XAI</h4>{s2}</div>", unsafe_allow_html=True)
    a2.markdown("<div class='arrow'>➜</div>", unsafe_allow_html=True)

    # ---------- Stage 3
    rank = ("<div class='muted' style='font-style:italic'>Only the top-ranked disease may be accepted, "
            "and only if it clears its own threshold.</div>") if out.get("ontology_top") else ""
    for r in (out.get("ontology_top") or [])[:3]:
        ok = "✔" if r["score"] >= r["theta"] else "✘"
        rank += (f"<div class='muted'>{ok} {r['disease']}: score {r['score']:.2f} "
                 f"(needs {r['theta']:.2f})</div>")
    chk = out.get("ontology_check") or out.get("case3_check")
    if chk:
        rank += (f"<div class='muted'>{'✔' if chk['confirmed'] else '✘'} registered profile "
                 f"{chk['disease']}: score {chk['score']:.2f} (needs {chk['theta']:.2f})</div>")
    s3 = (f"<div class='big'>{out['case']}</div>"
          f"<div>{out.get('reason', '')}</div>"
          f"<div style='margin-top:8px'><b>Ontology reasoning</b></div>{rank or '<div class=muted>direct lookup of the cluster code</div>'}")
    c3.markdown(f"<div class='stage'><h4>Stage 3 · Symbolic</h4>{s3}</div>", unsafe_allow_html=True)

    lab, col, icon = OUTCOME[out["outcome"]]
    truth = ""
    if true_label is not None:
        truth = f"<div class='why'>True disease in the dataset: <b>{true_label}</b></div>"
    st.markdown(f"<div class='verdict' style='background:{col}'><div class='lab'>{icon} {lab}</div>"
                f"<div class='dis'>{out['disease']}</div>{truth}</div>", unsafe_allow_html=True)
    with st.expander("Full Stage 2 explanation record (what Stage 3 reads)"):
        st.json(rec, expanded=False)


def gng_panel(pipe):
    g = pipe.gng
    c1, c2, c3 = st.columns(3)
    c1.metric("GNG nodes", len(g.W), delta=len(g.W) - g.n_init or None)
    c2.metric("Growth cap", g.cap)
    c3.metric("New diseases", len(pipe.ontology.registry))
    if pipe.ontology.registry:
        st.dataframe(pd.DataFrame([{"name": k, "GNG node": v["gng_node"], "patients": v["cases"],
                                    "profile size": len(v["profile"])}
                                   for k, v in pipe.ontology.registry.items()]),
                     hide_index=True, width="stretch")


@st.cache_resource
def get_extractor(names_tuple):
    return SymptomExtractor(list(names_tuple))


def highlight(text, res):
    """The complaint with the words that produced each symptom marked (denied ones struck through)."""
    import html as _h
    import re as _re
    low = text.lower().replace("’", "'")
    marks, taken = [], set()
    for f in res.findings:
        words = [w for w in f.heard.split() if w != ","]
        if not words:
            continue
        pat = r"\b" + r"[\W_]+".join(_re.escape(w) for w in words) + r"\b"
        # the extractor expanded contractions ("can't" -> "can not"); match the patient's spelling too
        pat = _re.sub(r"(do|does|did|is|are|was|were|have|has|had|could|would|should)\[\\W_\]\+not",
                      r"\1(?:n't|[\\W_]+not)", pat)
        pat = pat.replace(r"can[\W_]+not", r"can(?:'t|not|[\W_]+not)").replace(r"will[\W_]+not", r"(?:won't|will[\W_]+not)")
        for m in _re.finditer(pat, low):
            if not taken & set(range(m.start(), m.end())):
                taken |= set(range(m.start(), m.end()))
                marks.append((m.start(), m.end(), f))
                break
    marks.sort(key=lambda m: m[0])
    out, pos = [], 0
    for s, e, f in marks:
        cls = " class='neg'" if f.negated else ""
        out += [_h.escape(text[pos:s]), f"<mark{cls} title='{_h.escape(f.symptom)}'>", _h.escape(text[s:e]), "</mark>"]
        pos = e
    out.append(_h.escape(text[pos:]))
    return "".join(out)


def said_box(text, res):
    st.markdown(f"<div class='said'>🗣 “{highlight(text, res)}”</div>", unsafe_allow_html=True)


def heard_table(res, ds):
    if not res.findings:
        st.warning("No symptoms recognised. Try describing what the patient feels in more words.")
        return
    st.dataframe(pd.DataFrame([{"patient said": f.heard.replace(" ,", ","), "symptom": pretty(ds, f.symptom),
                                "status": f"denied ({f.trigger})" if f.negated else "present"}
                               for f in res.findings]), hide_index=True, width="stretch")


def vector(names, symptoms):
    idx = {n: j for j, n in enumerate(names)}
    x = np.zeros(len(names))
    for s in symptoms:
        x[idx[s]] = 1
    return x


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"<h2 style='color:{OX['800']};margin-bottom:0'>Model</h2>", unsafe_allow_html=True)
    ds = st.selectbox("Dataset", list(config.DATASETS),
                      format_func=lambda d: {"kaggle": "Kaggle (41 diseases)", "ddxplus": "DDXPlus (49 diseases)"}[d])
    seed = st.selectbox("Trained model (seed)", config.SEEDS,
                        help="Each seed holds out a different set of diseases as 'new'.")
    mp = model_path(ds, seed)
    if not mp.exists():
        st.warning("This model hasn't been trained yet.")
        if st.button("Train this model", type="primary"):
            with st.spinner("Training… (Kaggle: seconds, DDXPlus: about 30 s)"):
                pipe_, split_, info_ = train_one(load_dataset(ds), ds, seed)
                Path(config.MODEL_DIR).mkdir(exist_ok=True)
                pickle.dump({"pipeline": pipe_, "split": split_, "info": info_}, open(mp, "wb"))
            st.rerun()
        st.stop()
    base, split, info = load_model(ds, seed, mp.stat().st_mtime)
    st.caption(f"GHSOM: {info['ghsom_maps']} maps, depth {info['ghsom_max_depth']}, "
               f"{info['ghsom_leaf_clusters']} clusters · GNG starts with {info['gng_init_nodes']} nodes")
    with st.expander(f"Known diseases ({len(split.known)})"):
        st.write(", ".join(split.known))
    st.markdown("**Held out as 'new'** (never trained, not in the ontology):")
    st.markdown(pills(split.novel, "dark", ds=ds), unsafe_allow_html=True)
    st.divider()
    if st.button("Reset session (clear new diseases)"):
        for k in list(st.session_state):
            if k.startswith("pipe_") or k.startswith("demo_"):
                del st.session_state[k]
        st.rerun()

data = load_dataset(ds)
KEY = f"pipe_{ds}_{seed}"
pipe = session_pipe(KEY, base)

st.markdown("<div class='hero'><h1>GHSOM + GNG Neuro-Symbolic Disease Identification</h1>"
            "<p>Patient's own words → symptoms picked out → Stage 1 neural clustering → Stage 2 explanation → "
            "Stage 3 symbolic ontology. Known disease, old (earlier-added) disease, or new disease.</p></div>",
            unsafe_allow_html=True)

NLP_OK = ds == "kaggle"
ex = get_extractor(tuple(pipe.names)) if NLP_OK else None

tab0, tab3, tab4 = st.tabs(["🏥 Consultation", "📊 Results", "🧭 How it works"])

# ============================================================================= CONSULTATION (one screen)
with tab0:
    st.markdown("Patients arrive at the clinic one at a time and describe their complaint in their own words. "
                "**Nobody tells the system the disease**, and some patients have a disease it was never trained on "
                "(the held-out diseases in the sidebar). When the system meets one, it should register it as a new "
                "disease (**Case 2**) and recognise later patients of the same disease (**Case 3**), without the "
                "GNG growing again. The true disease is shown only after the system has answered.")
    mode = st.radio("Today's patients", ["Ordinary day: every test patient, in random order",
                                         "Outbreak day: patients with unfamiliar illnesses arrive more often"],
                    key="demo_clinic_mode", horizontal=True)
    outbreak = mode.startswith("Outbreak")
    dkey = f"demo_clinic_{ds}_{seed}_{'outbreak' if outbreak else 'ordinary'}"
    dpipe = session_pipe(dkey, base)
    if dkey + "_queue" not in st.session_state:          # the waiting room, fixed for this clinic day
        rng = np.random.default_rng(1000 + seed)
        test = [int(i) for i in split.test]
        unfamiliar = [i for i in test if data.y[i] in split.novel]
        familiar = [i for i in test if data.y[i] not in split.novel]
        rng.shuffle(unfamiliar)
        rng.shuffle(familiar)
        if outbreak:                                     # alternate: unfamiliar, familiar, unfamiliar, ...
            queue = [p for pair in zip(unfamiliar, familiar) for p in pair]
            queue += unfamiliar[len(familiar):] + familiar[len(unfamiliar):]
        else:
            queue = unfamiliar + familiar
            rng.shuffle(queue)
        st.session_state[dkey + "_queue"] = queue
    queue = st.session_state[dkey + "_queue"]
    hist = st.session_state[dkey + "_hist"]

    def see_patient(said, truth, x=None, picked=None):
        """One patient through the whole chain: words (or picked symptoms) -> Stage 1 -> 2 -> 3."""
        heard = ex.extract(said) if (NLP_OK and said) else None
        if heard is not None:
            x = vector(dpipe.names, heard.present)
        elif picked is not None:
            x = vector(dpipe.names, picked)
        out = dpipe.diagnose(x, keep_record=True)
        if truth is None:
            ok = "–"
        elif truth in split.novel:
            ok = "✔ flagged as new" if out["outcome"] != "known" else "✘ missed"
        else:
            ok = "✔" if (out["outcome"] == "known" and out["disease"] == truth) else "✘"
        hist.append({"#": len(hist) + 1,
                     "patient said": (said[:70] + "…") if said and len(said) > 70 else (said or "(symptoms picked)"),
                     "system's answer": out["disease"], "route": out["case"].split(":")[0],
                     "true disease (revealed after)": truth if truth is not None else "walk-in", "right?": ok,
                     "_out": out, "_said": said, "_heard": heard, "_truth": truth, "_picked": picked})

    # ------------------------------------------------------------------ the waiting room
    c1, c2, c3 = st.columns([1.2, 1.2, 3])
    left_in_queue = len(queue) - sum(h["_truth"] is not None for h in hist)
    nxt = c1.button("Call next patient ▶", type="primary", disabled=left_in_queue <= 0)
    five = c2.button("Call 5 more ⏩", disabled=left_in_queue <= 0)
    steps = 1 if nxt else 5 if five else 0
    for _ in range(min(steps, left_in_queue)):
        i = queue[len(queue) - left_in_queue]
        left_in_queue -= 1
        if NLP_OK:
            sy = [dpipe.names[j] for j in np.where(data.X[i] > 0.5)[0]]
            see_patient(ex.complaint_from(sy, seed=i), data.y[i])
        else:
            see_patient(None, data.y[i], data.X[i])
    c3.caption(f"{len(hist)} patient{'' if len(hist) == 1 else 's'} seen · {left_in_queue} still waiting. "
               "Use **Reset session** in the sidebar to start the day again.")

    # ------------------------------------------------------------------ a walk-in patient
    if NLP_OK:
        with st.expander("A walk-in patient: type what they say", expanded=not hist):
            st.text_area("What brings you to the hospital today?", key=dkey + "_walkin", height=90,
                         placeholder="e.g. Doctor, my body has been very hot since Monday, I'm always tired and my "
                                     "head is pounding. I'm not coughing.")
            if st.button("See this patient", disabled=not (st.session_state.get(dkey + "_walkin") or "").strip()):
                see_patient(st.session_state[dkey + "_walkin"], None)
    with st.expander("Or choose the symptoms yourself" if NLP_OK else "A walk-in patient: choose the symptoms",
                     expanded=not NLP_OK and not hist):
        picked = st.multiselect("Symptoms present", dpipe.names, key=dkey + "_pick",
                                format_func=lambda n: pretty(ds, n), placeholder="Type to search symptoms…")
        if st.button("See this patient", key=dkey + "_pick_go", disabled=not picked):
            see_patient(None, None, picked=picked)

    # ------------------------------------------------------------------ the patient just seen
    if hist:
        h = hist[-1]
        st.markdown(f"**Patient {h['#']} says:**" if h["_said"] else f"**Patient {h['#']}** "
                    f"({'symptoms chosen by the clinician' if h['_picked'] is not None else 'recorded symptoms'})")
        if h["_heard"] is not None:
            said_box(h["_said"], h["_heard"])
            st.markdown("**Symptoms picked out and passed to Stage 1:** " + pills(h["_heard"].present, ds=ds),
                        unsafe_allow_html=True)
        elif h["_picked"] is not None:
            st.markdown("**Symptoms passed to Stage 1:** " + pills(h["_picked"], ds=ds), unsafe_allow_html=True)
        render(h["_out"], dpipe, ds, h["_truth"])
        if h["_truth"] in split.novel:
            st.caption(f"{h['_truth']} is one of the held-out diseases: the system was never trained on it and it "
                       "is not in the ontology.")

        left, right = st.columns([3, 2])
        left.markdown("**Today's patients**")
        left.dataframe(pd.DataFrame([{k: v for k, v in x.items() if not k.startswith("_")} for x in hist]),
                       hide_index=True, width="stretch")
        with right:
            gng_panel(dpipe)
            fam = [x for x in hist if x["_truth"] is not None and x["_truth"] not in split.novel]
            unf = [x for x in hist if x["_truth"] in split.novel]
            if fam:
                st.markdown(f"**Familiar diseases named correctly:** {sum(x['right?'] == '✔' for x in fam)} of {len(fam)}")
            if unf:
                st.markdown(f"**Unfamiliar diseases flagged as new:** "
                            f"{sum(x['right?'].startswith('✔') for x in unf)} of {len(unf)}")
            found = {}
            for x in hist:
                if x["_out"]["outcome"] != "known":
                    found.setdefault(x["_out"]["disease"], []).append(x["_truth"] or "walk-in")
            if found:
                st.markdown("**New diseases the system found, and what those patients really had**")
                st.dataframe(pd.DataFrame([{"new disease": k, "patients": len(v),
                                            "their true disease(s)": ", ".join(f"{d} ×{v.count(d)}"
                                                                               for d in dict.fromkeys(v))}
                                           for k, v in found.items()]), hide_index=True, width="stretch")
    with st.expander("How the symptoms are picked out of the words"):
        st.markdown(
            "A small rule-based language step, with no internet and no external model, so every symptom can be "
            "traced to the patient's words:\n"
            "* each of the 131 symptoms has a list of everyday phrases (*running stomach*, *catarrh*, *my body is "
            "hot*, *can't sit still* …), and the word order and small spelling mistakes are tolerated;\n"
            "* where phrases overlap the longest wins (*mild fever* beats *fever*);\n"
            "* a symptom is **denied** when *no, not, never, without, denies…* comes shortly before it in the "
            "same clause, or *gone, stopped…* after it (\"no cough\", \"the headache has gone\").")

# ============================================================================= TAB 3
with tab3:
    rp = Path(config.RESULTS_DIR) / "per_run.csv"
    if not rp.exists():
        st.info("No results yet. Run `python test.py` (or section 4 of the notebook).")
    else:
        df = pd.read_csv(rp)
        names = {"kaggle": "Kaggle (41 diseases)", "ddxplus": "DDXPlus (49 diseases)"}
        met = {"known_accuracy": "Known-disease accuracy", "novel_detection_rate": "New-disease detection",
               "known_false_new_rate": "Known called new (lower is better)",
               "open_set_accuracy": "Open-set accuracy"}
        rds = st.radio("Dataset", [d for d in names if d in set(df.dataset)], horizontal=True,
                       format_func=names.get, key="res_ds")
        sub = df[df.dataset == rds]
        long = (sub.melt(id_vars=["method", "seed"], value_vars=list(met), var_name="metric")
                   .groupby(["method", "metric"])["value"].agg(["mean", "std"]).reset_index())
        long["metric"] = long["metric"].map(met)
        order_m = list(METHOD_COLORS)
        long = long[long["method"].isin(order_m)]          # chart: the four main systems; table: all of them
        xenc = alt.X("method:N", sort=order_m, axis=None)
        yscale = alt.Scale(domain=[0, 1.12])
        bars = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=xenc, y=alt.Y("mean:Q", title="Rate (mean ± sd over 10 seeds)", scale=yscale),
            color=alt.Color("method:N", sort=order_m, title=None,
                            scale=alt.Scale(domain=order_m, range=list(METHOD_COLORS.values())),
                            legend=alt.Legend(orient="bottom", columns=4, labelLimit=0, symbolType="square")),
            tooltip=["method", "metric", alt.Tooltip("mean:Q", format=".3f"), alt.Tooltip("std:Q", format=".3f")])
        err = alt.Chart(long).transform_calculate(
            lo="max(datum.mean - datum.std, 0)", hi="datum.mean + datum.std").mark_rule(
            color=OX["950"], opacity=0.55).encode(x=xenc, y=alt.Y("lo:Q", scale=yscale, title=None), y2="hi:Q")
        txt = alt.Chart(long).transform_calculate(top="datum.mean + datum.std").mark_text(
            dy=-7, fontSize=11, fontWeight=600, color=OX["950"]).encode(
            x=xenc, y=alt.Y("top:Q", scale=yscale, title=None), text=alt.Text("mean:Q", format=".2f"))
        chart = (alt.layer(bars, err, txt).properties(width=170, height=300)
                 .facet(column=alt.Column("metric:N", sort=list(met.values()), title=None,
                                          header=alt.Header(labelFontSize=12, labelColor=OX["800"])))
                 .configure_view(stroke=None).configure_axis(gridColor=OX["100"]))
        st.altair_chart(chart, width="content")

        tbl = (sub.groupby("method")[list(met) + ["new_groups_created", "false_new_groups", "novel_group_ARI",
                                                  "novel_diseases_recovered"]]
                  .mean().reindex(order_m + sorted(set(sub.method) - set(order_m))).round(3))
        tbl.columns = list(met.values()) + ["New groups created", "…mostly known patients",
                                            "New-group agreement (ARI)", "Held-out diseases recovered"]
        st.dataframe(tbl, width="stretch")

        st.markdown("#### What the numbers say")
        st.markdown(
            "- The full pipeline clearly beats **symbolic-alone** and the **neural pipeline with the same lenient GNG**, "
            "which never declares a new disease.\n"
            "- Against a neural pipeline whose GNG is **as strict as GHSOM** it is ahead on Kaggle (0.82 vs 0.80) "
            "and slightly behind on DDXPlus (0.92 vs 0.93): the ontology names some new-disease patients as known ones.\n"
            "- The ontology's clearest gains: best known-disease accuracy on Kaggle, fewer false 'new disease' groups "
            "on DDXPlus, and new-disease groups that match the true diseases better.\n"
            "- Open problem: many small new-disease groups on DDXPlus, because the design registers a new disease "
            "from a single patient.")

        for fname, col, xl in [("sweep_alpha.csv", "alpha_fit", "GHSOM conformal rejection level α"),
                               ("sweep_q.csv", "ontology_q", "Ontology threshold quantile q")]:
            fp = Path(config.RESULTS_DIR) / fname
            if fp.exists():
                sw = pd.read_csv(fp)
                sw = sw[sw.dataset == rds].groupby(col)[list(met)].mean().reset_index().melt(
                    id_vars=col, var_name="metric")
                sw["metric"] = sw["metric"].map(met)
                line = alt.Chart(sw).mark_line(point=True, strokeWidth=2.5).encode(
                    x=alt.X(f"{col}:Q", title=xl), y=alt.Y("value:Q", title="Rate", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", title=None, sort=list(met.values()),
                                    scale=alt.Scale(domain=list(met.values()),
                                                    range=[OX["900"], OX["600"], OX["300"], OX["400"]]),
                                    legend=alt.Legend(orient="bottom")),
                    strokeDash=alt.StrokeDash("metric:N", legend=None, sort=list(met.values())),
                    tooltip=[col, "metric", alt.Tooltip("value:Q", format=".3f")]
                ).properties(height=280, title=f"Sensitivity to {xl} (full pipeline)")
                st.altair_chart(line.configure_axis(gridColor=OX["100"]), width="stretch")

# ============================================================================= TAB 4
with tab4:
    def box(title, sub, dark=False):
        bg, fg, bd = (OX["700"], "white", OX["700"]) if dark else ("white", OX["950"], OX["600"])
        return (f"<div style='background:{bg};color:{fg};border:2px solid {bd};border-radius:12px;"
                f"padding:12px 14px;text-align:center'><div style='font-weight:700'>{title}</div>"
                f"<div style='font-size:.86rem;opacity:.85'>{sub}</div></div>")

    def band(label, sub, inner):
        return (f"<div style='border:1px solid {OX['300']};border-radius:14px;background:{OX['50']};"
                f"padding:12px 16px 16px 16px;margin:0'>"
                f"<div style='font-weight:700;color:{OX['800']}'>{label}</div>"
                f"<div style='font-size:.82rem;color:#6d5557;margin-bottom:10px'>{sub}</div>{inner}</div>")

    def down(lbl_left="", lbl_right=""):
        return (f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:24px;color:{OX['500']};"
                f"text-align:center;font-size:1.3rem;line-height:1.1;margin:4px 0'>"
                f"<div>↓<div style='font-size:.78rem;color:{OX['700']}'>{lbl_left}</div></div>"
                f"<div>↓<div style='font-size:.78rem;color:{OX['700']}'>{lbl_right}</div></div></div>")

    two = "display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:center"
    stage1 = (f"<div style='display:grid;grid-template-columns:0.8fr 0.25fr 1fr 0.25fr 1fr;align-items:center'>"
              f"{box('Patient X', 'symptom vector')}"
              f"<div style='text-align:center;color:{OX['500']};font-size:1.4rem'>➜</div>"
              f"{box('GHSOM', 'pretrained on labelled symptoms, fixed')}"
              f"<div style='text-align:center;color:{OX['500']};font-size:1.4rem'>➜"
              f"<div style='font-size:.75rem;color:{OX['700']}'>outlier</div></div>"
              f"{box('GNG', 'same node count as GHSOM, cap 1')}</div>")
    stage2 = (f"<div style='{two}'>{box('Fit explanation', 'quantization-error attribution · shallow CART')}"
              f"{box('Proximity explanation', 'conformal outlier test · symptom differential')}</div>")
    stage3 = box("Medical ontology", "lookup · reasoning over explanations · earlier additions — reads Stage 2 records only", dark=True)
    outs = (f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-top:6px'>"
            f"<div style='text-align:center;color:{OX['500']};font-size:1.3rem'>↓</div>"
            f"<div style='text-align:center;color:{OX['500']};font-size:1.3rem'>↓</div>"
            f"<div style='text-align:center;color:{OX['500']};font-size:1.3rem'>↓</div>"
            f"{box('Known disease', 'Case 1 (or ontology reasoning)')}"
            f"{box('Old disease', 'Case 3: added earlier')}"
            f"{box('New disease', 'Case 2: GNG keeps node, cap back to 1', dark=True)}</div>")
    st.markdown(
        band("Stage 1 · Neural", "classifies and routes", stage1)
        + "<div style='display:grid;grid-template-columns:0.8fr 0.25fr 1fr 0.25fr 1fr'><div></div><div></div>"
          f"<div style='text-align:center;color:{OX['500']};font-size:1.3rem'>↓<div style='font-size:.78rem;color:{OX['700']}'>known: cluster code</div></div>"
          f"<div></div><div style='text-align:center;color:{OX['500']};font-size:1.3rem'>↓<div style='font-size:.78rem;color:{OX['700']}'>nearest node or new node</div></div></div>"
        + band("Stage 2 · XAI", "explains, does not decide", stage2)
        + down()
        + band("Stage 3 · Symbolic", "decides from the explanations", stage3 + outs),
        unsafe_allow_html=True)
    st.caption("After a confirmed new disease, the GNG keeps the new node and its growth cap returns to 1, "
               "so the next patient of that disease can be recognised as an old disease (Case 3).")
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='stage'><h4>Case 1 · Known disease</h4>"
                "GHSOM accepts X (error within the cluster's range). If the cluster is pure, the ontology looks up "
                "the cluster code; a low-purity cluster goes to ontology reasoning instead.</div>",
                unsafe_allow_html=True)
    c2.markdown(f"<div class='stage'><h4>Case 2 · New disease</h4>"
                "GHSOM rejects X; no GNG node accepts it, so the GNG adds one node (cap 1). If the ontology finds "
                "no matching disease, a new disease is registered with the node's profile; the node is kept and "
                "the cap returns to 1.</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stage'><h4>Case 3 · Old disease</h4>"
                "GHSOM rejects X, but a node added in an earlier Case 2 accepts it and the symptom differential is "
                "small. The ontology confirms X against the registered profile and reports that earlier disease. "
                "The GNG is unchanged.</div>", unsafe_allow_html=True)

