# GHSOM + GNG Neuro-Symbolic Pipeline for Disease Identification

Python implementation of the design in *"A GHSOM + GNG Neuro-Symbolic Pipeline for Disease
Identification"*: a Kautz Type 3 (Neuro-Symbolic) pipeline. The stages run
in strict sequence: **neural → XAI → symbolic**. Each stage reads only the previous stage's output.

## Run

The `data/` folder (Kaggle `Training.csv`, DDXPlus, and both rule files) is included, so nothing has to be downloaded.

**Option A: Jupyter notebook (recommended).** Unzip the package, open `ghsom_gng_ns/run_and_test.ipynb` in
Jupyter Notebook, JupyterLab, VS Code or Anaconda Navigator, and run the cells from top to bottom. The notebook installs
the libraries, runs the quick Kaggle test, lets you test single patients, and then runs the full experiment.
It already contains the outputs from a complete run, so you can compare your numbers against them.

**Option B: terminal** (Python 3.9 or newer). Open a terminal in the `ghsom_gng_ns` folder:

```bash
pip install -r requirements.txt

python train.py --datasets kaggle            # quick: trains 10 Kaggle models (a few seconds)
python test.py --datasets kaggle --no-sweep  # tests them, prints the results table
python diagnose.py --demo                    # one patient at a time through the three stages

python train.py                              # full: both datasets, 10 seeds each (well under a minute)
python test.py                               # full test + parameter sweeps (about a minute)
python supervised_ceiling.py                 # classifiers trained with labels on every disease
python per_disease.py                        # results for each disease
python make_figures.py                       # figures into results/figures/
python -m pytest                             # 26 automated tests
python nlp_eval/evaluate_nlp.py              # check the plain-English intake on 30 complaints
python compare_fix.py                        # same complaints through the earlier and current pipeline
```

Test your own patient (comma-separated symptom names; `--list-symptoms` prints valid names,
`--list-diseases` shows which diseases this model knows and which were held out as "new"):

```bash
python diagnose.py --patient "itching,skin_rash,nodal_skin_eruptions,dischromic_patches"
python diagnose.py --patient "a,b,c" --patient "a,b,d"     # several patients in a row
python diagnose.py --dataset ddxplus --demo
```

Every run is seeded, so the same commands give the same numbers.

### Presentation app (Streamlit)

```bash
pip install -r requirements.txt     # includes streamlit
streamlit run app.py                # opens http://localhost:8501 in your browser
```
It runs entirely on your laptop (no internet needed once installed) and uses the trained models in `models/`;
if a model is missing, the sidebar offers a **Train this model** button. Four tabs:
* **Diagnose a patient**: pick symptoms or load a test patient, and follow Stage 1 → Stage 2 → Stage 3.
* **New-disease demo**: stream patients of a held-out disease; watch Case 2 register it and Case 3 recognise it.
* **Results**: the experiment results and parameter sweeps, from `results/`.
* **How it works**: the pipeline and the three cases.

Use **Reset session** in the sidebar to clear the new diseases registered during a demo.

### Consultation version: the patient describes the complaint in plain English

```bash
python -m streamlit run app_nlp.py
```
Same models and pipeline as `app.py` (which is unchanged). Everything is on **one Consultation screen**: press
**Call next patient** (or **Call 5 more**) to see test patients from the waiting room, whose complaints are written in
plain English from their records (an *ordinary day*: every test patient in random order; an *outbreak day*: patients
of the held-out, never-trained diseases arrive more often); type a **walk-in** patient's own words ("Doctor, my body
has been very hot since Monday, I'm always tired and I'm not coughing"); or **choose the symptoms yourself**. The
system picks out the symptoms, shows which words produced each one and which were denied, and shows Stage 1 →
Stage 2 → Stage 3 and the verdict. Nobody tells the system the disease: a test patient's true disease is revealed
only after it answers. The GNG and the new-disease registry carry over between patients, so a disease registered for
one patient (Case 2) is recognised for a later one (Case 3), and the screen keeps a tally of the day's patients,
familiar diseases named correctly, unfamiliar ones flagged as new, and what each new disease really was.

The symptom extraction (`neurosym/symptom_extractor.py`, phrases in `neurosym/symptom_lexicon.py`) is rule-based and
runs offline: everyday phrases for each of the 131 Kaggle symptoms (including *catarrh*, *running stomach*,
*stooling*), tolerance of word order and small spelling mistakes, longest phrase wins, and NegEx-style negation
("no fever", "the headache has gone"). It covers the Kaggle symptom list; DDXPlus records questionnaire answers
(pain location left/right, pain scales), so for DDXPlus use the symptom picker. Tests: `test_symptom_extractor.py`. Check on 30 complaints written independently of the phrase list (`nlp_eval/`): precision 0.93, recall 0.83, all 6 denials found.

## Files

| File | What it is |
|---|---|
| `config.py` | All hyper-parameters per dataset (chosen on train/calibration data only) |
| `neurosym/data.py` | Loads and encodes Kaggle (deduplicated) and DDXPlus into binary symptom vectors, reads the rule files, and builds the open-set split |
| `neurosym/ghsom.py` | **Stage 1** GHSOM: horizontal + vertical growth (τ1, τ2), plus label-aware expansion (pretraining with labels) |
| `neurosym/gng.py` | **Stage 1** GNG: initialised from GHSOM leaves (same node count), growth cap 1, commit/reject/reset |
| `neurosym/xai.py` | **Stage 2** QE attribution, shallow CART surrogate, Mondrian conformal test, symptom differential |
| `neurosym/ontology.py` | **Stage 3** rule-based ontology scoring, per-disease thresholds, new-disease registry |
| `neurosym/pipeline.py` | The full pipeline: Stage 2 explanation record → Stage 3 resolution (Cases 1/2/3), plus baselines |
| `neurosym/evaluate.py` | Open-set metrics |
| `train.py` / `test.py` | Training and testing scripts |
| `diagnose.py` | Try single patients and print each stage's output |
| `app.py`, `.streamlit/config.toml` | Presentation app (oxblood theme) |
| `app_nlp.py`, `neurosym/symptom_extractor.py`, `neurosym/symptom_lexicon.py` | Consultation version of the app (one screen): plain-English complaint → symptoms → pipeline |
| `supervised_ceiling.py` | Logistic regression and random forest trained with labels on all diseases (the ceiling) |
| `per_disease.py` | Recall and new-disease flag rate for every disease |
| `make_figures.py` | Draws the figures from the saved results |
| `test_pipeline.py`, `test_symptom_extractor.py` | 26 automated tests (`python -m pytest`) |
| `run_and_test.ipynb` | Jupyter notebook: install, train, test, single patients, full experiment |

## Datasets and rule bases

Sourcing and cleaning are exactly as described in Section 3.2.3 of the dissertation.

* **Kaggle disease–symptom** (`data/Training.csv`): 4,920 rows, 41 diseases. The duplicate column
  `fluid_overload.1` is merged into `fluid_overload` and exact duplicate rows are removed, leaving
  **304 distinct patients × 131 symptoms** (7.65 symptoms per patient on average). Rule base:
  `data/rules/independent_rules.csv` (354 items from 43 public-health pages; sources in `rule_sources.csv`).
* **DDXPlus** (`data/ddxplus/release_test_patients.csv.gz`): 134,529 patients, 49 diseases. Exact duplicates are
  removed (270), a class-balanced sample of up to **40 patients per disease** is drawn (seed 0), evidence codes are
  converted to 0/1 columns, columns that never occur are dropped and same-disease identical rows are removed (13),
  giving **1,943 patients × 426 columns**. Rule base: `data/rules/ddxplus_rules.csv` (606 items from 48 sources).
  The supervised ceiling uses a separate pool of 18,757 other patients (up to 400 per disease).

## Experimental protocol (open-set)

For each of **10 seeds** (0–9), `n_novel` diseases (4 Kaggle, 5 DDXPlus) are held out. They are **not used for
training and are removed from the rule base**, so they are genuinely unknown. The known diseases are split
50/20/30 into train / calibration / test. The test stream mixes the known-disease test patients with all patients
of the held-out diseases, in random order, one patient at a time.

Systems compared on the same stream:

* **Symbolic-alone**: the rule base applied to raw symptoms. Unmatched patients all go into one "unknown" bucket.
* **Neural-alone (lenient GNG)**: the same networks and routing with no rule base.
* **Neural-alone (strict GNG)**: the GNG is as strict as the GHSOM, so the networks alone can declare new diseases.
* **Neuro-symbolic (full)**: Stage 1 → Stage 2 → Stage 3.
* **Supervised**: logistic regression and a 300-tree random forest on the same training data, plus a random forest
  that flags a patient as new when its top class probability is below the 5th percentile of the calibration patients.

## Alignment with the design document

| Design requirement | Where / how | Status |
|---|---|---|
| Stages in strict sequence; Stage 3 reads Stage 2 records only | `pipeline.explain()` builds the record; `resolve()` reads only it. The record holds no raw symptom list; Stage 3 rebuilds its evidence from the cluster profile + QE-attribution deviations (fit) or the symptom differential (proximity) | Implemented |
| GHSOM pretrained on labelled symptoms, fixed | `ghsom.py` (horizontal + vertical growth; impure units are expanded); never retrained during testing | Implemented (how labels are used is my interpretation) |
| GNG with the same node count as GHSOM, cap 1 | `gng.py` `init_from_ghsom`, `propose/commit/reject` | Implemented; the brief GNG pretraining is an addition |
| Fit explanation: QE attribution + shallow CART (readability only) | `xai.qe_attribution`, `xai.CARTSurrogate`; CART never gates a decision | Implemented (CART depth 8) |
| Proximity explanation: conformal test per GNG cluster + symptom differential | `xai.ConformalScorer` (Mondrian), `xai.symptom_differential` | Implemented; clusters with < 30 calibration patients use pooled calibration |
| Hamming or Jaccard distance for binary symptoms | Conformal tests: weighted Jaccard; differential: Hamming | Implemented |
| Case 1: error in range + high purity → lookup; low purity → rule-base reasoning | `resolve()` fit path | Implemented |
| Case 2: no node accepts → add one; rule base finds nothing → register, keep node, cap back to 1 | `resolve()` + `diagnose()` | Implemented |
| Case 3: earlier node accepts, small differential → rule base confirms registered profile → old disease; GNG unchanged; not confirmed → Case 2 path | `Ontology.confirm_registered()`; no node is moved | Implemented |
| GHSOM retrained (and CART refitted) after a set number of new diseases | — | **Not implemented**: the number is still to be chosen |
| Medical ontology (e.g. HPO / ICD-11) | Sourced rule files (NHS, CDC, Mayo, MedlinePlus, …) | Rule base, not a formal ontology |

Note on the document itself: the Case 2 flowchart sends "ontology matches a disease → yes" to "Old disease (Case 3)",
while the Stage 3 table sends a match with a *known* disease to a known diagnosis. The code follows the table.

## Results (10 seeds, mean ± sd; from `results/summary.csv`)

**Kaggle (4 of 41 diseases held out per run)**

| System | Known acc. | Known macro-F1 | Known called new | New-disease detection | Open-set acc. | New groups | ARI |
|---|---|---|---|---|---|---|---|
| Symbolic-alone | 0.596 ± 0.041 | 0.585 | 0.112 | 0.581 ± 0.249 | 0.591 ± 0.058 | 1.0 | 0.000 |
| Neural (lenient GNG) | 0.950 ± 0.036 | 0.925 | 0.000 | 0.000 ± 0.000 | 0.709 ± 0.036 | 0.0 | – |
| Neural (strict GNG) | 0.941 ± 0.036 | 0.920 | 0.022 | 0.421 ± 0.203 | 0.807 ± 0.058 | 3.5 | 0.925 |
| **Neuro-symbolic** | **0.969 ± 0.020** | 0.969 | 0.023 | 0.415 ± 0.177 | **0.829 ± 0.054** | 4.1 | 1.000 |
| Logistic regression | 1.000 ± 0.000 | 1.000 | 0.000 | 0.000 | 0.746 ± 0.021 | 0.0 | – |
| Random forest | 1.000 ± 0.000 | 1.000 | 0.000 | 0.000 | 0.746 ± 0.021 | 0.0 | – |
| RF + threshold | 0.935 ± 0.040 | 0.943 | 0.065 | 0.997 ± 0.011 | 0.951 ± 0.029 | 1.0 | 0.000 |

**DDXPlus (5 of 49 diseases held out per run)**

| System | Known acc. | Known macro-F1 | Known called new | New-disease detection | Open-set acc. | New groups | ARI |
|---|---|---|---|---|---|---|---|
| Symbolic-alone | 0.668 ± 0.024 | 0.680 | 0.120 | 0.314 ± 0.163 | 0.571 ± 0.043 | 1.0 | 0.000 |
| Neural (lenient GNG) | 0.952 ± 0.015 | 0.956 | 0.015 | 0.640 ± 0.258 | 0.867 ± 0.062 | 8.3 | 0.805 |
| Neural (strict GNG) | 0.941 ± 0.015 | 0.948 | 0.026 | 0.684 ± 0.204 | 0.871 ± 0.047 | 14.3 | 0.844 |
| **Neuro-symbolic** | **0.953 ± 0.015** | 0.954 | 0.015 | 0.482 ± 0.249 | 0.825 ± 0.066 | 10.4 | 0.820 |
| Logistic regression | 0.989 ± 0.006 | 0.989 | 0.000 | 0.000 | 0.719 ± 0.005 | 0.0 | – |
| Random forest | 0.987 ± 0.007 | 0.987 | 0.000 | 0.000 | 0.717 ± 0.006 | 0.0 | – |
| RF + threshold | 0.946 ± 0.014 | 0.965 | 0.049 | 0.869 ± 0.114 | 0.925 ± 0.029 | 1.0 | 0.000 |

Figures are in `results/figures/`. Full explanation records for Cases 1, 2 and 3 are in `results/example_records.json`.

### Reading the results honestly
* The full pipeline beats **symbolic-alone** on both datasets (open-set 0.829 vs 0.591 Kaggle, 0.825 vs 0.571 DDXPlus)
  and is the best of the four neural/symbolic systems on known-disease accuracy.
* Against the **neural stages alone** it is ahead on Kaggle (0.829 vs 0.807 strict) but **behind on DDXPlus**
  (0.825 vs 0.871): there the rule base names about 17% of unseen-disease patients as a similar known disease.
* The main leak is in the GHSOM: it accepts 43% (Kaggle) and 30% (DDXPlus) of unseen-disease patients into a known cluster.
* A **random forest with a confidence threshold** has the highest open-set accuracy (0.951, 0.925), but it cannot
  register, group or recognise a new disease, and cannot explain its answers.
* The new-disease groups the pipeline registers are pure (purity ≈ 1.0) and match the real held-out diseases
  (ARI 1.00 Kaggle, 0.82 DDXPlus), but there are more groups than real new diseases.
* A stricter rule-base threshold (q = 0.20) raises DDXPlus new-disease detection from 0.48 to 0.60 (`sweep_q.csv`).
* CART fidelity is low (about 0.31 Kaggle, 0.25 DDXPlus); per the design it gates nothing.
