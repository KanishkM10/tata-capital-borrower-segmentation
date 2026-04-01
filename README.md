# 🏦 Tata Capital — Corporate Borrower Segmentation using K-Means Clustering

> A complete machine learning pipeline built in **Python (Jupyter Notebook)** to segment Tata Capital's corporate lending portfolio of **19,000 borrowers** into actionable risk and opportunity clusters using **K-Means clustering**. Submitted as **ML Assignment 3** for the PGDM programme at MILE Education (Batch 2025–27).

---

## 🗂️ Project Overview

Tata Capital manages a large, heterogeneous corporate borrower portfolio spanning multiple industries, company sizes, geographies, and loan types. This notebook applies an unsupervised machine learning approach to group borrowers into financially distinct segments — enabling targeted risk management, personalised product recommendations, and data-driven portfolio strategy.

The analysis spans the full data science workflow: quality checks, EDA, visualisation, preprocessing, feature engineering, model selection, clustering, profiling, and executive-level business recommendations.

---

## 📋 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Tata Capital corporate borrower dataset (`25_Tata_Capital.csv`) |
| **Rows** | 19,000 borrowers |
| **Columns** | 16 (10 numerical, 6 categorical) |
| **Key Features** | Annual Revenue, Loan Outstanding, Debt-to-Equity Ratio, Interest Coverage Ratio, DPD Instances, Credit Rating Score, Collateral Coverage %, Years in Business, Repayment Track, Business Sector, Company Size, Geographic Region, Loan Type |

---

## 🧱 Notebook Structure — 10 Tasks

### ✅ Task 1 — Data Quality Check
- Shape: 19,000 rows × 16 columns; no duplicates found
- 4 columns with missing values: `DPD_Instances_Last_12M` (4.48%), `Debt_to_Equity_Ratio` (4.19%), `Interest_Coverage_Ratio` (3.98%), and one additional column
- Outlier detection using IQR method — `Debt_to_Equity_Ratio` and `Current_Ratio` flagged

---

### ✅ Task 2 — Data Exploration
- 5-point summary statistics for all 10 numerical columns
- Value counts and percentage share for all 6 categorical columns
- Key findings: Manufacturing (20%) and Services (18%) dominate the sector mix; Small (31%) and Medium (28%) companies make up the majority of the portfolio
- Extended summary table with skewness, range, IQR, and variance
- Notable: `Annual_Revenue_INR_Cr` and `Loan_Outstanding_INR_Cr` are both heavily right-skewed, reflecting SME dominance with a long tail of large corporates

---

### ✅ Task 3 — Exploratory Data Analysis (EDA)
- **Histograms with KDE:** `Annual_Revenue_INR_Cr`, `DPD_Instances_Last_12M`, `Credit_Rating_Score`
- **Boxplot:** Credit Rating Score by Company Size — Large/Medium firms show higher medians with tighter IQR; Micro/Small firms show higher risk and wider variability
- **Correlation Heatmap:** `Credit_Rating_Score` positively correlated with ICR and Collateral Coverage; negatively correlated with DPD — consistent with credit risk theory
- **Grouped Bar Chart:** Mean DPD by Repayment Track — DPD is a monotonic predictor of repayment classification (Excellent ≈ 0 → NPA = high)

---

### ✅ Task 4 — Visualisation (6 Charts)

| Chart | Type | Insight |
|-------|------|---------|
| 1 | Bar Chart | Business Sector distribution — Manufacturing + Services = largest share |
| 2 | Scatter Plot | Annual Revenue vs. Loan Outstanding by Company Size — Micro clusters bottom-left, Large widely dispersed |
| 3 | Boxplot | Interest Coverage Ratio by Repayment Track — Excellent = high ICR; NPA = critically low and volatile ICR |
| 4 | Grouped Bar | Loan Type by Geographic Region — Term Loans dominate; West/North have higher Working Capital concentration |
| 5 | Heatmap | Mean numerical features by Business Sector — IT & Healthcare lead on Credit Score and ICR; Real Estate & Infrastructure show elevated D/E and DPD |
| 6 | Violin Plot | Collateral Coverage % by Company Size — Micro/Small show under-collateralised tail below 100%; Large shows tighter, more uniform coverage |

---

### ✅ Task 5 — Data Pre-Processing
- **Median imputation** for all numerical missing values
- **Mode imputation** for all categorical missing values
- **Label Encoding** for ordinal feature `Repayment_Track` (NPA=0 → Excellent=4)
- **One-Hot Encoding** via `pd.get_dummies` for all remaining nominal categorical features
- **Standard Scaling** (`StandardScaler`) applied to all numerical features before clustering

---

### ✅ Task 6 — Feature Engineering

Two new features engineered and added to the dataset:

**Feature 1: `Revenue_per_Loan`**
```
Annual_Revenue_INR_Cr / Loan_Outstanding_INR_Cr
```
Revenue-to-debt efficiency metric. High (>10) = lean, low-leverage borrower. Low (<2) = over-borrowed relative to earnings.

**Feature 2: `Financial_Stress_Index`**
```
(Debt_to_Equity_Ratio × DPD_Instances_Last_12M) / (Credit_Rating_Score + 1)
```
Composite stress indicator combining leverage, delinquency, and credit quality. Near-zero = financially healthy; high = distressed.

Both features show strong right-skewed distributions — the majority of clients cluster near healthy values, with a meaningful stressed tail.

---

### ✅ Task 7 — K-Means Clustering Model
- **Features used:** All 12 numerical features (including 2 engineered features)
- **Model:** `KMeans(n_clusters=5, random_state=42)` fit on `StandardScaler`-transformed data
- **Cluster labels** assigned back to original unscaled dataframe as `Cluster` column

---

### ✅ Task 8 — Identifying Optimal K
- **Elbow Method (WSS/Inertia):** Clear bend between K=4 and K=5; inertia flattens from K=5 onward
- **Silhouette Score:** Highest score at K=5, confirming well-separated, cohesive clusters
- **Final K Selected: 5** — justified by convergence of both methods

---

### ✅ Task 9 — Cluster Profiles

| Cluster | Label | Size | Avg Revenue (₹Cr) | Avg Loan (₹Cr) | Avg DPD | Credit Score | ICR |
|---------|-------|------|-------------------|----------------|---------|--------------|-----|
| 0 | Seasoned Leveraged Survivors | 3,179 (16.7%) | ₹1,652 | ₹44 | 4.92 | Medium | Medium |
| 1 | High-Debt Emerging Risk | 2,572 (13.5%) | ₹580 | ₹303 | 2.04 | Medium | 5.54 |
| 2 | Stressed Young Borrowers | 7,573 (39.9%) | ₹500 | Medium | 8.87 | 8.57 | 9.40 |
| 3 | High-Revenue Prime Clients | 4,937 (26.0%) | ₹3,391 | ₹239 | 1.54 | High | 12.79 |
| 4 | Legacy Near-Zero Exposure | 739 (3.9%) | ₹2,039 | ₹0.10 | Low | High | High |

- Cluster profile heatmap (normalised) produced for visual separation
- Mode of categorical features per cluster computed and interpreted
- Grouped bar chart: Credit Score, ICR, and DPD compared across all 5 clusters

---

### ✅ Task 10 — Business Recommendations

**Cluster 0 — Seasoned Leveraged Survivors (16.7%)**
Senior RM-led reviews for DPD-deteriorating accounts; offer Lease Rental Discounting and Loan Against Property to deleverage; debt restructuring programme for D/E > 8 with milestone-linked improvement incentives.

**Cluster 1 — High-Debt Emerging Risk (13.5%)**
Immediate Enhanced Credit Monitoring; no fresh unsecured credit; steer toward self-liquidating instruments (Invoice Discounting, PO Financing); stress test each account at 20% revenue decline scenario.

**Cluster 2 — Stressed Young Borrowers (39.9%)** *(highest priority)*
Tiered digital intervention model (DPD > 10 → Specialised Stressed Assets Team); EMI moratorium for viable accounts; Credit Rehabilitation Programme converting recovering borrowers into Prime clients over 3–5 years.

**Cluster 3 — High-Revenue Prime Clients (26.0%)**
Platinum Client Programme with dedicated Senior RMs and C-suite access; priority for large-ticket Project Finance and Syndicated Loans; loyalty-linked pricing reductions of 25–50 bps for sustained repayment track.

**Cluster 4 — Legacy Near-Zero Exposure (3.9%)** *(highest-return opportunity)*
Dormant Client Reactivation Campaign within 90 days; Welcome Back incentive (fee waiver + 30 bps concession); potential to add ₹11,000–22,000 Cr to lending book by converting just 30% of segment at ₹50–100 Cr per facility.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and aggregation |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | `KMeans`, `StandardScaler`, `silhouette_score` |
| `IPython.display` | Rich notebook output formatting |

**Python version:** 3.11.13 | **Environment:** Jupyter Notebook (ipykernel)

---

## 📁 File Structure

```
tata-capital-borrower-segmentation/
│
├── 25_Tata_Capital_ML_Assignment3.ipynb   # Full ML pipeline notebook (133 cells)
├── 25_Tata_Capital.csv                    # Source dataset (19,000 corporate borrowers)
└── README.md                             # Project documentation
```

> ⚠️ **Note:** If `25_Tata_Capital.csv` is not present in the same directory as the notebook, the `pd.read_csv("25_Tata_Capital.csv")` call in Cell 1 will raise a `FileNotFoundError`. Place both files in the same folder before running.

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebook
```bash
# Clone the repo
git clone https://github.com/your-username/tata-capital-borrower-segmentation.git
cd tata-capital-borrower-segmentation

# Launch Jupyter
jupyter notebook 25_Tata_Capital_ML_Assignment3.ipynb
```

Then run all cells sequentially (**Kernel → Restart & Run All**) for a clean end-to-end execution.

---

## 📌 Key Results Summary

- **5 distinct borrower clusters** identified via K-Means (validated by both Elbow and Silhouette methods)
- **Cluster 2 (Stressed Young Borrowers)** — largest segment at 39.9%, highest DPD, most critical for portfolio risk management
- **Cluster 3 (Prime Clients)** — anchor portfolio at 26%, highest revenue and best credit quality, priority for retention
- **Cluster 4 (Legacy Near-Zero)** — smallest at 3.9% but highest reactivation ROI, avg revenue ₹2,039 Cr with near-zero current exposure
- **2 engineered features** (`Revenue_per_Loan`, `Financial_Stress_Index`) added meaningful discriminative signal to the clustering

---

## 🎓 Academic Context

Submitted as **ML Assignment 3** for the PGDM programme (Finance + Research & Business Analytics) at **MILE Education**, Batch 2025–27. The assignment demonstrates end-to-end proficiency in:
- Unsupervised machine learning (K-Means clustering)
- Feature engineering for financial domain problems
- Optimal K selection using quantitative methods
- Cluster profiling and business interpretation
- Executive-level strategic recommendations grounded in data

---

## 👤 Author

**Kanishk**
PGDM — Finance & Business Analytics | MILE Education (Batch 2025–27)
Former Associate Data Engineer @ Celebal Technologies

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/)

---

## 📜 License

This project is shared for **educational and portfolio purposes**. The dataset is proprietary to the MILE Education coursework and should not be redistributed.
