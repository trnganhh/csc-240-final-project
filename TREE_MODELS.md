# Model 3: Decision Tree & Random Forest (Anh)

Script: [tree_models.py](tree_models.py) · Outputs: [results/tree_models/](results/tree_models/)

## How to run

```
python tree_models.py           # full run, ~45 min on 6 cores
python tree_models.py --quick   # small samples, ~2 min, for checking the code
```

Needs `clean/natality_2024.parquet` and `clean/natality_2025.parquet` (made by `clean_data.py`).

## Design

| Split | Data | Used for |
|---|---|---|
| train | 800,000 random 2024 births | fitting the models |
| valid | 300,000 other 2024 births | choosing hyper-parameters and the F1 threshold |
| test | **all** 3.5M 2025 births | temporal validation, used once at the end |

**Features (main models):** 22 maternal characteristics (28 model inputs after one-hot encoding race/ethnicity). These are the same pre-pregnancy and early-pregnancy variables that Model 2 should use:
- Demographics: mother_age, education, married, race/Hispanic origin (one-hot, 7 columns)
- Body size: bmi, height_in
- Smoking: smoked_before, smoked_during, cigs_before
- Pregnancy history: prior_births, prior_deaths, prior_terminations, first_baby, months_since_last_birth, prior_preterm, prior_cesarean
- Pre-pregnancy conditions: diabetes_prepreg, hypertension_prepreg, infertility_treatment
- Prenatal care: prenatal_first_tri, no_prenatal_care
- Engineered: **risk_count**, the number of 6 established risk factors present (age < 20 or ≥ 35, BMI ≥ 30, smoking before pregnancy, pre-pregnancy hypertension, pre-pregnancy diabetes, prior preterm birth). Preterm rate rises steadily with the count: 0 → 6.6%, 1 → 9.0%, 2 → 14.3%, 3 → 24.1%, 4 → 35.2% (2024 sample).

**Borderline variables (kept, but note as a limitation):** months_since_last_birth is measured up to *this* birth date, no_prenatal_care is more likely when a pregnancy ends early, and smoked_during includes 3rd-trimester smoking. Dropping or replacing all three lowers random forest AUC by only about 0.01, compared with the +0.09 gain from PREVIS.

**Excluded on purpose:**
- Gestational diabetes, gestational hypertension, and eclampsia are diagnosed during the pregnancy and are often the reason for an early delivery.
- prenatal_start_month leaks the outcome, because a 32-week birth cannot start care in month 9.
- prenatal_visits and weight_gain grow with the length of the pregnancy, so they are used only in the H6 experiment.
- Birth weight and baby sex are outcomes of the birth.

**Missing values:** left as NaN. scikit-learn ≥ 1.4 trees split on missingness directly, so no imputation is needed.

**Tuning** picks the setting with the best ROC-AUC on the 2024 validation set:
- Decision tree: max_depth ∈ {4, 6, 8, 10, 14} × min_samples_leaf ∈ {100, 500}
- Random forest (100 trees): max_depth ∈ {10, 16, None} × min_samples_leaf ∈ {50, 200}

The final forest uses 300 trees, max_features = sqrt, and max_samples = 0.5. The chosen settings are in `tuning.json`.

**Threshold:** precision, recall, and F1 use the threshold that maximises F1 on the 2024 validation set. That same threshold is then applied to 2025.

## Results (2025 test set)

| Model | ROC-AUC | PR-AUC | F1 | AUC 2024 valid → 2025 test |
|---|---|---|---|---|
| Age-only LR (reference baseline) | 0.523 | 0.099 | 0.162 | 0.520 → 0.523 |
| Decision tree | 0.659 | 0.184 | 0.250 | 0.660 → 0.659 |
| Random forest | **0.672** | **0.195** | **0.256** | 0.672 → 0.672 |
| Decision tree + PREVIS | 0.751 | 0.260 | 0.323 | 0.745 → 0.751 |
| Random forest + PREVIS | 0.761 | 0.277 | 0.333 | 0.756 → 0.761 |

Preterm rate in 2025 ≈ 9%, so a random model has PR-AUC ≈ 0.09 and ROC-AUC = 0.5.

**H5:** Using all the maternal features raises ROC-AUC by about 0.15 over age alone, well above the +0.05 target. The random forest beats a single tree by only about 0.01. Neither model loses performance from 2024 to 2025, so both generalise across years. Compare these numbers with Navya's full logistic regression to see whether the extra complexity is worth it. The age-only LR here is a linear-in-age reference; age has a U-shaped relationship with preterm birth, so a linear term can't capture it.

**H6:** Adding total prenatal visits raises ROC-AUC by about 0.09 and PR-AUC by about 40% for both algorithms. This is an inflated "improvement", because longer pregnancies have more visits.

**Most important features** (random forest permutation importance, measured as the drop in 2025 AUC when the feature is shuffled):
1. prior_preterm
2. risk_count
3. months_since_last_birth
4. no_prenatal_care
5. bmi
6. hypertension_prepreg
7. height_in
8. diabetes_prepreg

These results support H1–3. Results for age, smoking, and BMI should be confirmed with Navya's odds ratios.

## Output files

| File | Contents |
|---|---|
| `summary.csv` | One row per model: ROC-AUC, PR-AUC, and F1 on 2024 valid and 2025 test, plus the drop between years |
| `metrics.csv` | Long format with precision, recall, F1, and threshold as well |
| `leakage_h6.csv` | H6 table: AUC with and without PREVIS |
| `tuning.json` | Validation AUC for every hyper-parameter setting tried |
| `feature_importance_rf.csv` / `.png` | Permutation and impurity importance for the random forest |
| `roc_pr_curves.png` | ROC and precision-recall curves on 2025 for all five models |
| `decision_tree_depth3.png` | A readable depth-3 tree for the slides |
| `run_log.txt` | Full console log of the run |
