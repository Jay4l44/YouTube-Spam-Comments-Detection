# 🎓 YouTube Spam Comment Detection
### Machine Learning Final Project

> Final project for the **Machine Learning** course, building a complete ML pipeline
> to classify YouTube comments as spam or legitimate (ham) across 5 music video datasets.

**Best Model: Random Forest — 95.9% Accuracy | 97.9% Precision | 95.9% F1**

---

## 📁 Dataset
**YouTube Spam Collection** — 5 CSV files of YouTube comments from music videos:

| Artist | Total Comments | Spam | Ham |
|--------|---------------|------|-----|
| Eminem | 448 | 245 (54.7%) | 203 |
| LMFAO | 438 | 236 (53.9%) | 202 |
| Shakira | 370 | 174 (47.0%) | 196 |
| Psy | 350 | 175 (50.0%) | 175 |
| Katy Perry | 350 | 175 (50.0%) | 175 |
| **Total** | **1,956** | **1,005 (51.4%)** | **951 (48.6%)** |

Columns: `COMMENT_ID`, `AUTHOR`, `DATE`, `CONTENT`, `CLASS` (0=Ham, 1=Spam)

## 📂 Project Structure
├── notebooks/

│   └── YouTube_Spam_Detection.ipynb

├── data/

│   ├── Youtube01Psy.csv

│   ├── Youtube02KatyPerry.csv

│   ├── Youtube03LMFAO.csv

│   ├── Youtube04Eminem.csv

│   └── Youtube05Shakira.csv

├── report/

│   └── YouTube_Spam_Detection_Report.pdf

└── README.md

---

## 🔬 Stage 1 — Data Understanding & Cleaning

- Combined all 5 datasets into a unified DataFrame of **1,956 comments**
- Fixed `DATE` column dtype from `object` → `datetime64`
- Found **245 missing DATE values**, all concentrated in the Eminem dataset (54.7% of its rows)
- Dropped `DATE` (54.7% missing + irrelevant to classification), `COMMENT_ID` and `AUTHOR` (identifiers with no predictive value)
- Final working columns: `CONTENT`, `CLASS`, `ARTIST`
- Dataset is **naturally balanced** (51.4% spam / 48.6% ham) — no resampling (SMOTE) needed

---

## 📊 Stage 2 — Visualization

### Univariate
- Target variable is well-balanced: 1,005 spam vs 951 ham (~54 comment difference, negligible)
- Comment counts per artist range from 350 to 448 — no single artist dominates
- Comment lengths are **heavily right-skewed**: mean=95 chars, median=48 chars, max=1,200 chars — outliers are likely spam with excessive promotional text

### Bivariate
- **Spam comments are ~3x longer than ham** on average (137 vs 49 characters)
- Spam IQR (36–171) is much wider than ham IQR (18–62), confirming spam varies more in length
- Eminem (54.7%) and LMFAO (53.9%) had the highest spam rates; Shakira was the only artist with more ham than spam (47%)
- The length difference between spam and ham ranges from **1.5x (Psy) to 6.3x (Shakira)** across all artists

### Multivariate
- A split violin plot confirmed all three findings simultaneously: ham is short and tightly clustered (0–100 chars), spam is longer with much wider spread, and this pattern is **universal across all 5 artists**
- Shakira and Eminem showed the most extreme spam tails reaching 1,200+ characters

**Key insight: Comment length is a powerful, artist-independent feature for spam detection.**

## ⚙️ Stage 3 — Preprocessing

| Step | Details |
|------|---------|
| TF-IDF Vectorization | `max_features=5000`, `stop_words='english'` → produced 4,229 unique word features |
| MinMaxScaler | Normalized `COMMENT_LENGTH` from (2–1,200) → (0.0–1.0) to match TF-IDF scale |
| Feature Combination | TF-IDF (1956×4229) + Length (1956×1) = combined matrix **(1956×4230)** |
| Train/Test Split | 80/20 split with `stratify=y` → 1,564 train / 392 test samples |
| Cross Validation | StratifiedKFold (5 splits) → **mean accuracy 94.7%**, std=0.011 (low variance = no overfitting) |

## 🏆 Stage 4 — Model Evaluation

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** 🏆 | **95.9%** | **97.9%** | 94.0% | **95.9%** |
| XGBoost | 95.7% | 95.5% | **96.0%** | 95.8% |
| Logistic Regression | 94.9% | 97.4% | 92.5% | 94.9% |
| Linear SVC | 94.4% | 95.4% | 93.5% | 94.5% |
| Naive Bayes | 90.3% | 88.6% | 93.0% | 90.8% |

### Random Forest Confusion Matrix
|  | Predicted Ham | Predicted Spam |
|--|--------------|----------------|
| **Actual Ham** | 187 ✅ | 4 ❌ |
| **Actual Spam** | 12 ❌ | 189 ✅ |

Only **16 total errors** out of 392 — 4 false positives (ham flagged as spam) and 12 false negatives (spam that slipped through). False positive rate of only **2%**.

### GridSearch CV (Bonus)
Tested 18 hyperparameter combinations × 5 folds = **90 total training runs**

- Best parameters: `n_estimators=200`, `max_depth=None`, `min_samples_split=2`
- Tuned model: **95.7%** vs original **95.9%** — only 0.2% difference
- Conclusion: the default Random Forest was already near-optimal

## 🛠️ Tech Stack
Python · pandas · numpy · scikit-learn · XGBoost · matplotlib · seaborn · scipy

## ▶️ How to Run
1. Clone the repo
```bash
   git clone https://github.com/Jay4l44/YouTube-Spam-Comments-Detection.git
```
2. Install dependencies
```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```
3. Open the notebook
```bash
   jupyter notebook notebooks/YouTube_Spam_Detection.ipynb
```

*This project was submitted as a final course requirement.*
