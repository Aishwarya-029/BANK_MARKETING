# Bank Marketing: Predicting Term Deposit Subscription

Predict whether a client will subscribe to a term deposit after a bank's direct marketing phone call, using the UCI Bank Marketing dataset.

## Problem

A Portuguese bank ran phone-based marketing campaigns to sell term deposits. Calling every client is expensive, so the goal is to predict which clients are likely to say "yes" and focus calls on them.

- **Task:** binary classification
- **Target:** `y` (`yes` / `no`: did the client subscribe?)

## Dataset

| Property | Value |
|---|---|
| File | `bank-additional-full.csv` |
| Source | [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| Citation | Moro, S., Cortez, P., Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems |
| Size | 41,188 rows, 20 input features + 1 target |
| Format | Semicolon-separated (`sep=";"`) |

Key facts about the data:

- **Imbalanced target:** only 4,640 of 41,188 clients (11.3%) said `yes`.
- **Missing values are coded as `"unknown"`** in `default`, `education`, `housing`, `loan`, `job` and `marital`.
- **`duration` leaks the target:** call length is only known after the call, so it is excluded from the model.
- **`pdays = 999`** means "not contacted before" and affects 96% of rows.

## Approach

All steps are in `Bank_Marketing.ipynb`.

1. **EDA:** target balance, count plots for every categorical feature, box plots and histograms for numeric features, correlation heatmaps.
2. **Cleaning:** replaced `unknown` with missing values and dropped those rows (38,245 of 41,188 rows kept). Dropped `default`, which showed almost no useful signal.
3. **Feature selection:** dropped `duration` (target leakage), `emp.var.rate`, `nr.employed` and `euribor3m` (highly correlated with each other), and `pdays`, `previous` and `campaign`.
4. **Encoding and scaling:** mapped yes/no columns to 1/0, one-hot encoded the other categorical columns, and scaled numeric features with `MinMaxScaler` fitted on the training set only. The model uses 38 features.
5. **Model:** logistic regression (`C=0.1`) on an 80/20 train/test split (30,596 train rows, 7,649 test rows).

## Results

Evaluated on the 7,649-row test set.

| Metric | Value |
|---|---|
| Accuracy | 0.895 |
| ROC-AUC (test) | 0.749 |
| ROC-AUC (train) | 0.760 |
| Precision, `yes` class | 0.62 |
| Recall, `yes` class | 0.16 |
| F1, `yes` class | 0.26 |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted no | Predicted yes |
|---|---|---|
| **Actual no** | 6706 | 86 |
| **Actual yes** | 717 | 140 |

**Reading the results:** always predicting "no" already gives about 88.8% accuracy on this test set, so the 89.5% accuracy is only a small gain. The model is precise when it predicts "yes" (62%) but finds only 16% of the real subscribers. The train and test AUC are close, so the model is not overfitting. It is underfitting the minority class.

## Limitations and next steps

- Use `class_weight="balanced"` or tune the decision threshold to raise recall for the `yes` class.
- Try tree-based models (random forest, gradient boosting) and compare them using ROC-AUC and PR-AUC.
- Use the already-defined `GridSearchCV` helper to tune hyperparameters.
- The split is random, but the data is time-ordered, so a time-based split would be a more realistic test.
- Rows with `unknown` values were dropped. Treating `unknown` as its own category would keep about 3,000 more rows.

## Getting started

```bash
git clone https://github.com/Aishwarya-029/BANK_MARKETING.git
cd BANK_MARKETING
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Bank_Marketing.ipynb
```

The notebook reads the data from `/content/bank-additional-full.csv` (the Google Colab path). If you run it locally, change that path to where your copy of the CSV is.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
