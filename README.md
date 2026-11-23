# Feature Selection Study

Compares six feature-selection methods on data where **the right answer is known**, then shows the most common feature-selection mistake: choosing features before cross-validation, which makes pure noise look predictive.

## Data

50 features: **6 informative**, **4 redundant** (linear combinations of the informative ones) and **40 pure noise**. 3,000 rows. The column order is fixed, so every method's picks can be checked.

## 1. Which methods find the useful features? (top 10 picks)

| Method | Type | Useful in top 10 | Informative | Redundant | Time |
|---|---|---|---|---|---|
| mutual information | filter | 7 | 4 | 3 | 0.4 s |
| ANOVA F-test | filter | 8 | 4 | 4 | 0.004 s |
| RFE + logistic | wrapper | 9 | **6** | 3 | 0.07 s |
| L1 logistic (C=0.05) | embedded | 6 | 5 | 1 | 0.02 s |
| random-forest impurity | embedded | **10** | **6** | 4 | 2.1 s |
| permutation importance | model-agnostic | **10** | **6** | 4 | 13.6 s |

- **Filters judge each feature alone.** They're fast but miss features that only help in combination with others, which is how `make_classification` builds its informative features.
- **Tree-based and permutation methods found all 10 useful features**, at 500–3,000× the cost of an F-test.
- **L1 regularization keeps only a few features on purpose.** At C=0.05 it zeroes out redundant copies, which is useful when you want a small model.

**Accuracy barely changes:** every top-10 subset scores CV AUC 0.869–0.871, and all 50 features score 0.866. With 3,000 rows, a regularized linear model just ignores the noise. Feature selection pays off mainly when **features are expensive** to compute, store or explain, or when there are far more features than rows.

## 2. The mistake: selecting features outside cross-validation

Here the data is 200 rows × 5,000 features of **pure noise**, with no relationship to the label at all:

```
select on all rows, then CV  -> AUC 0.894   (looks predictive: it is not)
select inside each CV fold   -> AUC 0.557   (the truth: about chance)
```

Picking the "best" 20 features using every row lets the label leak into the choice. Each validation fold then contains rows that helped pick the features. The fix is to make selection **a step in the pipeline** (`SelectKBest` inside `make_pipeline`), so it's re-fit on each training fold.

## How to run

```bash
pip install -r requirements.txt
python study.py     # ~20 seconds
pytest
```

The commit history includes a real bug from early on: the module was first named `selectors.py`, which shadowed Python's standard-library `selectors` module and broke imports deep inside scipy.

## Next steps
- Add stability selection: how often each feature gets picked across bootstrap samples
- Add Boruta and SHAP-based selection
