# 📓 Finance_Analysis.ipynb — Full Walkthrough

This notebook contains the complete end-to-end analysis pipeline behind the
[Personal Finance Audit app](https://finance-audit.streamlit.app/).
Every chart, ML model, and recommendation in the live app originates here.

---

## Step 1 — Data Generation & Setup

### What happens
- Installs and imports all libraries
- Generates a realistic 12-month synthetic Indian bank statement
  (618 transactions across 8 spending categories, Indian merchants, ₹ amounts)
- Injects 10 deliberate anomalies for ML validation
- Applies keyword-based auto-categorisation to every transaction
- Exports `sample_bank_statement.csv`

### Key design decisions
- `np.random.seed(42)` — makes the synthetic data identical every run
- `np.random.normal(avg, std)` — bell-curve amounts per category (realistic)
- Anomalies injected deliberately so Step 3 ML has something real to detect

### Output
```
✅ 618 transactions generated
Date range: 2024-01-01 → 2024-12-31
Total spend: ₹5,96,537
sample_bank_statement.csv exported
```

---

## Step 2 — Exploratory Data Analysis (5 Charts)

### Chart 1 — Spending by Category (Donut)
- **Chart type:** Plotly `px.pie` with `hole=0.45`
- **Insight:** Bills & Rent dominates at 21.6%, Others at 24.6%
- **Why it matters:** Identifies the highest-impact category for savings

### Chart 2 — Monthly Spending Trend (Line)
- **Chart type:** Plotly `px.line` with markers + `add_hline()` average reference
- **Insight:** Spending peaks visible vs monthly average
- **Why it matters:** Identifies high-spend months for investigation

### Chart 3 — Category Breakdown by Month (Stacked Bar)
- **Chart type:** Plotly `px.bar` with `barmode='stack'`
- **Insight:** Which categories drive high-spend months
- **Interactive:** Click legend items to isolate any category

### Chart 4 — Top 10 Merchants (Horizontal Bar)
- **Chart type:** Plotly `px.bar` with `orientation='h'`
- **Insight:** Unknown Transfer and Society Maintenance are top spends
- **Note:** Anomalous merchants visible in top 5

### Chart 5 — Day-of-Week Spending Heatmap
- **Chart type:** Plotly `px.imshow` on a pivot matrix
- **Insight:** Spending patterns by day of week × week of month
- **Why it matters:** Reveals behavioural patterns — weekend impulse buying etc.

---

## Step 3 — Anomaly Detection (IsolationForest)

### Algorithm
**IsolationForest** — unsupervised ML that isolates anomalies by building
random decision trees. Anomalous transactions are easier to isolate
(fewer splits needed) than normal ones.

### Features used
| Feature | Why |
|---|---|
| `Amount` | Unusually high amounts = strongest signal |
| `Month_Num` | Unusual timing for a category |
| `Category_Code` | Label-encoded category |
| `Day_of_Week` | Transactions at unusual times |
| `Amount_Rank` | Percentile rank within category — catches relative outliers |

### Parameters
```python
IsolationForest(
    n_estimators  = 100,   # 100 isolation trees
    contamination = 0.05,  # expect 5% anomalies
    random_state  = 42,    # reproducibility
)
```

### Validation results
Tested against 10 deliberately injected anomalies:

| Injected Anomaly | Amount | Detected |
|---|---|---|
| Unknown Transfer | ₹45,000 | ✅ |
| Luxury Purchase | ₹28,000 | ✅ |
| Foreign Transaction | ₹15,000 | ✅ |
| ATM Withdrawal | ₹20,000 | ✅ |
| Cash Withdrawal | ₹25,000 | ✅ |
| Duplicate Netflix | ₹649 × 2 | ✅ |
| Zomato (inflated) | ₹8,500 | ✅ |
| UPI Payment × 2 | ₹10,000 × 2 | ✅ |

**Recall: 8/10 injected anomalies detected (80%)**

### Output files
- `transactions_scored.csv` — all transactions with anomaly score
- `anomalies_detected.csv` — flagged transactions only

---

## Step 4 — K-Means Spending Personas

### Algorithm
**K-Means Clustering** groups months with similar spending patterns
into personas. Each month is represented as a vector of 8 numbers
(one per spending category).

### Why months — not transactions?
Clustering months reveals behavioural patterns over time.
"January was a Saver month, December was a High Spender month"
is far more actionable than clustering individual transactions.

### Steps
```
Monthly spending matrix (12 rows × 8 category columns)
    ↓ StandardScaler — zero mean, unit variance
    ↓ Elbow method — confirms K=3 is optimal
    ↓ KMeans(n_clusters=3, random_state=42)
    ↓ Rank clusters by total spend
    ↓ Label: Saver / Balanced / High Spender
```

### Elbow method
Inertia was calculated for K=1 to K=6.
The "elbow" — where adding clusters stops reducing inertia — appears at K=3,
confirming 3 personas is the right choice for 12 months of data.

### Persona profiles

| Persona | Avg Monthly Spend | Characteristics |
|---|---|---|
| 🟢 Saver | Lowest | Essentials only — bills, food. Low entertainment + shopping |
| 🔵 Balanced | Middle | Consistent across all categories. No single category dominates |
| 🔴 High Spender | Highest | Elevated discretionary spend — travel, shopping, dining |

### Savings recommendations engine
`generate_recommendations()` produces 5 data-driven tips:
- Cuts tied to top spending category (with exact ₹ figure)
- Monthly budget alert based on High Spender months
- Food/Dining reduction estimate (cooking 3x/week)
- Subscription audit estimate
- Annual savings potential if Saver pace is maintained year-round

### Output files
- `monthly_personas.csv` — each month labelled with its persona
- `recommendations.csv` — 5 personalised tips with ₹ figures

---

## 🛠️ How to Run

```bash
# Install dependencies
pip install pandas numpy plotly scikit-learn

# Open notebook
jupyter notebook Finance_Analysis.ipynb

# Run all cells in order (Cell 1 → Cell 33)
# Kernel → Restart & Run All
```

> **Note:** Run cells top to bottom. Each step depends on variables
> created in the previous step.

---

## 📦 Dependencies

```
pandas      — data manipulation
numpy       — numerical operations
plotly      — interactive charts
scikit-learn — IsolationForest, KMeans, StandardScaler, LabelEncoder
```

---

## 🔗 Related

- **Live App:** [Personal Finance Audit](https://finance-audit.streamlit.app/)
- **App code:** `app.py`
- **Data loader:** `multi_loader.py`
- **Portfolio:** [sasikumar-19.github.io/Portfolio](https://sasikumar-19.github.io/Portfolio)

---

*By [Sasi Kumar](https://linkedin.com/in/sasikumar19) — B.Tech CSE, Vignan's Institute of Information Technology*
