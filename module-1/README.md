# Module 1: Advanced Machine Learning Techniques

In this project I trained ensemble models (Random Forest and XGBoost) to classify breast tumours as malignant or benign, compared them against simple baselines, and then opened up the best tree ensemble with SHAP and LIME to see what it was actually relying on. The short version: the ensembles did their job of fixing a single tree's instability, but a plain scaled logistic regression was just as good on this dataset, and the explanations point to clinically sensible features.

## The two techniques and why I picked them

**1. Ensemble learning.** I used both flavours: bagging (Random Forest, many decorrelated trees averaged together) and boosting (XGBoost, trees fitted one after another on the previous trees' mistakes), plus a random search to tune XGBoost. I chose ensembles because they are the go-to approach for tabular data, and I wanted to see for myself whether they really beat simpler models or whether that is just the reputation.

**2. Explainable AI.** I used SHAP for global and local explanations and LIME for a second, independent local explanation of the same patient. I chose XAI because this is a medical prediction task, and I don't think a model that flags cancer should be trusted on accuracy alone. I also wanted to check whether two different explanation methods tell the same story.

## Data and setup

- **Dataset:** Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn (`load_breast_cancer`). 569 patients, 30 numeric features describing cell nuclei. I flipped the labels so that 1 means malignant, which is the class I care most about catching (37.3% of rows).
- **Split:** 60/20/20 stratified train / validation / test (341 / 114 / 114 rows). I compared and picked models on validation only, and I touched the test set once at the end.
- **Cross-validation:** stratified 5-fold CV on the training set as a second check, since one 114-row validation split is small.
- **Seed:** 42 everywhere (`SEED = 42`).

## How to run it

I ran it locally with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute Module1_Ensemble_and_XAI.ipynb --output executed.ipynb
# or open it interactively: pip install jupyterlab && jupyter lab
```

On macOS, XGBoost needs the OpenMP runtime. If importing it fails with a `libomp.dylib` error, run `brew install libomp`.

## Results

Here are the numbers straight from the CSVs in `evidence/`. Everything uses a 0.5 decision threshold, and "recall" is recall on the malignant class.

| Model | CV AUC (mean ± sd) | Val accuracy | Val AUC | Test accuracy | Test recall | Test AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9941 ± 0.0078 | 0.974 | 0.996 | 0.974 | 0.952 | 0.995 |
| Decision Tree (baseline) | 0.9364 ± 0.0180 | 0.904 | 0.904 | 0.904 | 0.833 | 0.889 |
| Random Forest (bagging) | 0.9860 ± 0.0149 | 0.974 | 0.992 | 0.956 | 0.905 | 0.997 |
| XGBoost (boosting) | 0.9916 ± 0.0102 | 0.965 | 0.993 | 0.965 | 0.905 | 0.993 |
| XGBoost (tuned) | n/a | n/a | n/a | 0.974 | 0.929 | 0.995 |

The tuned XGBoost isn't in the CV and validation CSVs because the notebook reports it separately in section 3: best search CV AUC 0.9917, validation accuracy 0.974 and AUC 0.994.

**What I found.** The ensembles clearly fixed the single decision tree, lifting CV AUC from 0.936 to 0.986 (Random Forest) and 0.992 (XGBoost) with a tighter spread. But the scaled logistic regression matched or beat both of them, with the best CV AUC (0.994) and the best validation AUC (0.996), so on this nearly linearly separable dataset the extra flexibility of trees bought me nothing. Tuning XGBoost didn't meaningfully help either: CV AUC stayed at 0.992, and validation moved by exactly one patient (accuracy 0.965 to 0.974), which is noise on 114 rows. The same goes for the test set: with 114 rows, one patient is worth about 0.9 points of accuracy, so the gaps of a point or two between the top models are not a real ranking.

## SHAP and LIME

The model behind the explanations is the tuned XGBoost, which the notebook picks automatically as the tree ensemble with the best validation AUC (logistic regression won overall, but SHAP's tree explainer needs a tree model).

- **Global (SHAP):** the most important features by mean |SHAP value| are worst texture, mean concave points, worst perimeter, area error, worst area and worst concavity. These describe the size and irregular shape of the nuclei, which lines up with how malignancy is judged clinically.
- **Local (one hard case):** I explained test row 112, the patient closest to the decision boundary (P(malignant) = 0.440, actually malignant, so the model gets this one wrong). SHAP and LIME shared **8 of their top 10 features**, and both put worst texture first, though they ordered the rest differently. That is about what I expected, since LIME fits a local linear model on random perturbations while SHAP is exact for trees.
- **Sanity check:** the Spearman rank correlation between XGBoost's built-in feature importance and SHAP importance was 0.86, so they broadly agree but aren't interchangeable.

## Evidence

Everything in `evidence/` comes from the executed notebook, not from anything I typed in by hand.

- `notebook_executed.html`: the full executed notebook as HTML, viewable without Jupyter
- `results_validation.csv`: validation metrics for the four main models
- `results_test.csv`: test metrics for all five models, including tuned XGBoost
- `cv_results.csv`: 5-fold CV ROC-AUC mean and standard deviation on the training set
- `fig_roc_validation.png`: validation ROC curves, plus Random Forest error as trees are added
- `fig_confusion_test.png`: test confusion matrix for the tuned XGBoost
- `fig_shap_beeswarm.png`: SHAP beeswarm, one dot per test patient
- `fig_shap_bar.png`: global SHAP importance (mean |SHAP value|)
- `fig_shap_waterfall_uncertain.png`: SHAP waterfall for test row 112, the most uncertain case
- `fig_lime_uncertain.png`: LIME explanation for the same row 112
- `extract_evidence.py`: the small script that copies the tables and figures above out of the executed notebook

## Limitations

- The dataset is small (569 rows) and comes from a single source, so I wouldn't expect these numbers to transfer to other hospitals or imaging equipment.
- The decision threshold is fixed at 0.5. A real screening tool should set it based on the cost of a missed cancer versus a false alarm.
- I didn't check probability calibration or do any fairness or subgroup analysis.

## Software versions and seed

Python 3.12.15, random seed 42. Exact versions are pinned in `requirements.txt`:

scikit-learn 1.9.1, xgboost 3.4.1, shap 0.52.0, lime 0.2.0.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, nbformat 5.11.1, nbconvert 7.17.1, ipykernel 7.4.0.

The tuning step is sensitive to the XGBoost version. Older XGBoost builds pick different hyperparameters in the random search, so I'd stick to the pinned versions if you want to reproduce these exact numbers.
