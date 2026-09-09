# 💰 Personal Finance Anomaly & Lifestyle Audit

> An end-to-end ML-powered Streamlit app that analyses your bank and UPI statements — detecting unusual transactions, clustering spending personas, and delivering personalised savings recommendations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finance-audit.streamlit.app/)

---

## 🚀 Live Demo

**👉 [Open the live app](https://finance-audit.streamlit.app/)**

No signup needed. Click **"Load Demo Data"** to see the full analysis instantly.

---

## 📌 What It Does

| Feature | Description |
|---|---|
| 🚨 **Anomaly Detection** | IsolationForest ML model flags unusual transactions — duplicate charges, unusually large spends, suspicious patterns |
| 🧬 **Spending Personas** | K-Means clustering groups your months into Saver / Balanced / High Spender personas |
| 📊 **5 Interactive Charts** | Donut, monthly trend, stacked category bar, top merchants, day-of-week heatmap |
| 💡 **Savings Tips** | Personalised recommendations with real ₹ figures based on your actual data |
| 📂 **Multi-source Upload** | Supports SBI, HDFC, ICICI, Axis, Kotak, PhonePe, Paytm — auto-detects format |

---

## 🖥️ App Screenshots

> *<img width="1903" height="911" alt="image" src="https://github.com/user-attachments/assets/296e9fa6-65c7-441e-9063-6b2f013a905d" />*

---

## 🧠 ML Models Used

### IsolationForest — Anomaly Detection
- **Type:** Unsupervised ML
- **Features:** Transaction amount, month, category, day of week, amount percentile rank
- **Contamination:** 5% (flags the most unusual 5% of transactions)
- **Output:** Normal / 🚨 Anomaly label + severity score per transaction

### K-Means Clustering — Spending Personas
- **Type:** Unsupervised ML
- **Input:** Monthly spending vectors across 8 categories
- **K=3:** Saver 🟢 / Balanced 🔵 / High Spender 🔴
- **Validation:** Elbow method confirms optimal K

---

## 📓 Analysis Notebook

`Finance_Analysis.ipynb` contains the full step-by-step analysis:

| Step | Content |
|---|---|
| Step 1 | Synthetic data generation + PDF/CSV loader foundation |
| Step 2 | 5 Plotly EDA charts — donut, trend, stacked bar, merchants, heatmap |
| Step 3 | IsolationForest anomaly detection + validation against injected anomalies |
| Step 4 | K-Means clustering + elbow method + persona radar chart + savings engine |

Run locally:
```bash
jupyter notebook Finance_Analysis.ipynb
```
---

## ⚙️ How It Works

```
CSV / Excel uploaded
      ↓
Read ALL rows — no header assumption
      ↓
find_transaction_header() — scans and scores each row
      ↓
Skip bank/customer metadata rows automatically
      ↓
detect_source() — SBI / HDFC / PhonePe / Paytm / Generic
      ↓
Parse transactions → clean DataFrame
      ↓
IsolationForest → anomaly scores
K-Means → spending personas
Plotly → interactive charts
```

---

## 📥 Supported File Formats

| Source | Format | Notes |
|---|---|---|
| SBI | Excel (.xlsx) | Password protected — enter password in app |
| HDFC | Excel / CSV | Auto-detected |
| ICICI | CSV | Auto-detected |
| Axis | CSV | Auto-detected |
| Kotak | CSV | Auto-detected |
| PhonePe | CSV | Downloaded from app → Transaction History |
| Paytm | CSV | Downloaded from Passbook → Statement |
| Google Pay | CSV | Downloaded from web.google.com/pay |
| Any bank | CSV/Excel | Generic parser as fallback |

---

## 🔒 Privacy

- Files and passwords are used **only within your browser session**
- **Nothing is stored** in any database after you close the tab
- This app has no backend storage, login, or account system
- Source code is fully open — verify it yourself

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas | Data manipulation |
| scikit-learn | IsolationForest + K-Means |
| Plotly | Interactive charts |
| Streamlit | Web app framework |
| msoffcrypto-tool | Password-protected Excel unlock |
| openpyxl | Excel file reading |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/Sasikumar-19/personal-finance-audit.git
cd personal-finance-audit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📈 Sample Analysis Output

From the synthetic demo dataset (618 transactions, Jan–Dec 2024):

| Metric | Value |
|---|---|
| Total spend | ₹5,96,537 |
| Monthly average | ₹49,711 |
| Anomalies detected | 31 |
| Top category | Bills & Rent (21.6%) |
| Best saver month | February |
| Savings potential | ₹X/year (matching saver months) |

---

## 🔮 Future Enhancements

- [ ] PDF statement support (text-based PDFs)
- [ ] Budget goal setting and progress tracking
- [ ] Multi-month comparison across years
- [ ] Export analysis as PDF report
- [ ] Email alerts for anomalous transactions

---

## 👤 About

**Sasi Kumar**
B.Tech CSE (Cyber Security) — Vignan's Institute of Information Technology, Duvvada
Batch 2023–2027 | CGPA 8.77

[![GitHub](https://img.shields.io/badge/GitHub-Sasikumar--19-181717?logo=github)](https://github.com/Sasikumar-19)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sasikumar19-0A66C2?logo=linkedin)](https://linkedin.com/in/sasikumar19)
[![Portfolio](https://img.shields.io/badge/Portfolio-sasikumar--19.github.io-1D9E75)](https://sasikumar-19.github.io/Portfolio)


---
