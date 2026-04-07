# Amazon Fine Food Reviews — NLP Insights Pipeline

An end-to-end NLP project analyzing 568,000+ Amazon food reviews to extract 
business insights using sentiment analysis, topic modeling, and trend analysis.

---

## Live Dashboard
🔗 [Coming Soon — Streamlit Deployment]

---

##  Project Overview

This project builds a complete NLP insights pipeline on the 
[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 
dataset, covering:

- **Sentiment Analysis** — TF-IDF + Logistic Regression (91% accuracy) vs VADER
- **Topic Modeling** — LDA discovering 8 business-relevant topics from 364K reviews
- **Helpfulness Prediction** — Predicting review helpfulness using text + metadata
- **Trend Analysis** — How sentiment and topics evolved from 1999 to 2012
- **Interactive Dashboard** — Streamlit app for business stakeholder reporting

---

##  Key Business Insights

1.  **Packaging Crisis** — 40.88% of packaging-related reviews are negative, 
   representing ~17,500 dissatisfied customers — the highest complaint rate of any category

2.  **Declining Satisfaction** — Positive sentiment declined steadily from 93% (2003) 
   to 82% (2012) as the platform scaled to mainstream audiences

3.  **Snacks Lead Satisfaction** — Snacks & Sweet Foods maintains 91.85% positive 
   sentiment — Amazon's most consistently satisfying food category

4.  **Price Sensitivity Growing** — Shopping & Pricing became the largest topic 
   by 2010, outpacing all food categories — customers increasingly focused on value

5.  **Coffee vs Tea** — Coffee has a 14.33% negative rate vs Tea's 10.89% — 
   coffee buyers have significantly higher expectations

6.  **Negative Reviews Are Longer** — Dissatisfied customers write more detailed 
   reviews — providing richer feedback for product improvement

---

##  Tech Stack

| Area | Tools |
|---|---|
| Data Processing | Pandas, NumPy |
| NLP | NLTK, Scikit-learn, TF-IDF |
| Sentiment | Logistic Regression, VADER |
| Topic Modeling | LDA (Latent Dirichlet Allocation) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Environment | Python 3.11, VS Code |

---

##  Project Structure
amazon-nlp-insights/
│
├── datasets/
│   └── Reviews.csv                  ← Download from Kaggle
│
├── Notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Sentiments.ipynb
│   ├── 03_Topic_Modelling.ipynb
│   ├── 04_Helpfulness.ipynb
│   └── 05_Trends.ipynb
│
├── dashboard/
│   └── app.py
│
├── .gitignore
└── README.md

---

##  How to Run

**1 — Clone the repository**
```bash
git clone https://github.com/praharsh2527/NLP-insights-on-Unstructured-Data.git
cd NLP-insights-on-Unstructured-Data
```

**2 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud streamlit plotly
```

**3 — Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 
and place `Reviews.csv` in the `datasets/` folder.

Dataset files are not included due to size. Download Reviews.csv from Kaggle and run notebooks in order to generate all processed CSV files.

**4 — Run notebooks in order**
01_EDA.ipynb → 02_Sentiments.ipynb → 03_Topic_Modelling.ipynb →
04_Helpfulness.ipynb → 05_Trends.ipynb

**5 — Launch the dashboard**
```bash
cd dashboard
python -m streamlit run app.py
```

---

## Model Results

| Model | Task | Accuracy |
|---|---|---|
| TF-IDF + Logistic Regression | Sentiment Classification | 91% |
| VADER (Raw Text) | Sentiment Classification | 88% |
| SGD Classifier | Helpfulness Prediction | 74% |
| LDA | Topic Discovery | 8 Topics |

---

## 👤 Author
**Praharsh Aditya Agrawal**  
[LinkedIn](https://www.linkedin.com/in/praharsh-agrawal-403166286/) | 
[GitHub](https://github.com/praharsh2527)
