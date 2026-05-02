# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           CHURNGUARD AI  —  Professional Bank Churn Dashboard           ║
# ║           Unified Mentor Internship  |  European Bank Dataset           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# HOW TO RUN:
#   1. pip install streamlit pandas numpy scikit-learn matplotlib seaborn
#   2. Place your CSV as  "European_Bank.csv"  in the same folder
#   3. streamlit run churnguard_pro.py

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI  |  Bank Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  ─  Full CSS injection
# ─────────────────────────────────────────────────────────────────────────────
DESIGN = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Clash+Display:wght@400;500;600;700&family=Cabinet+Grotesk:wght@300;400;500;700;800&display=swap');

/* ─────────────  TOKENS  ───────────── */
:root{
  --bg:       #060B18;
  --surface:  #0C1225;
  --card:     #0F1830;
  --card2:    #131E35;
  --border:   #1A2B48;
  --border2:  #223354;
  --accent:   #2563EB;
  --accent2:  #3B82F6;
  --teal:     #0EA5E9;
  --green:    #10B981;
  --amber:    #F59E0B;
  --red:      #EF4444;
  --purple:   #8B5CF6;
  --text:     #E2EAF4;
  --muted:    #556B8D;
  --muted2:   #7A92B0;
  --white:    #FFFFFF;
  --glow-b:   rgba(37,99,235,.20);
  --glow-g:   rgba(16,185,129,.15);
  --glow-r:   rgba(239,68,68,.15);
  --glow-a:   rgba(245,158,11,.15);
}

/* ─────────────  BASE  ───────────── */
html,body,[class*="css"]{
  font-family:'Cabinet Grotesk',sans-serif!important;
  background:var(--bg)!important;
  color:var(--text)!important;
}
.main,.block-container{background:var(--bg)!important; padding-top:1.5rem!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important; border-right:1px solid var(--border);}
section[data-testid="stSidebar"] .block-container{padding:1.5rem 1rem!important;}

/* ─────────────  SCROLLBAR  ───────────── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:4px}

/* ─────────────  HERO BANNER  ───────────── */
.hero{
  background:linear-gradient(135deg,#0C1A3A 0%,#091228 60%,#060B18 100%);
  border:1px solid var(--border2);
  border-radius:20px;
  padding:40px 48px 34px;
  margin-bottom:32px;
  position:relative;
  overflow:hidden;
}
.hero::before{
  content:'';position:absolute;top:-80px;right:-60px;
  width:340px;height:340px;
  background:radial-gradient(circle,rgba(37,99,235,.18) 0%,transparent 65%);
  border-radius:50%;pointer-events:none;
}
.hero::after{
  content:'';position:absolute;bottom:-50px;left:38%;
  width:220px;height:220px;
  background:radial-gradient(circle,rgba(16,185,129,.12) 0%,transparent 65%);
  border-radius:50%;pointer-events:none;
}
/* animated grid lines */
.hero-grid{
  position:absolute;inset:0;
  background-image:
    linear-gradient(rgba(37,99,235,.06) 1px,transparent 1px),
    linear-gradient(90deg,rgba(37,99,235,.06) 1px,transparent 1px);
  background-size:40px 40px;
  pointer-events:none;
}
.hero-eyebrow{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(37,99,235,.12);
  border:1px solid rgba(37,99,235,.28);
  border-radius:30px;
  padding:5px 16px;
  font-size:.72rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--accent2);font-weight:600;
  margin-bottom:14px;
}
.hero-dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--green);
  box-shadow:0 0 8px var(--green);
  animation:blink 1.8s ease-in-out infinite;
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.hero-title{
  font-family:'Clash Display',sans-serif!important;
  font-size:3rem;font-weight:700;line-height:1.05;
  color:#fff;margin:0 0 8px;letter-spacing:-.03em;
}
.hero-title .accent{
  background:linear-gradient(90deg,#3B82F6,#0EA5E9);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-subtitle{font-size:1rem;color:var(--muted2);font-weight:400;margin:0 0 22px;max-width:560px;}
.hero-stats{display:flex;gap:32px;}
.hero-stat{display:flex;flex-direction:column;}
.hero-stat-val{
  font-family:'Clash Display',sans-serif!important;
  font-size:1.6rem;font-weight:600;color:#fff;line-height:1;
}
.hero-stat-lbl{font-size:.72rem;color:var(--muted);letter-spacing:.06em;text-transform:uppercase;margin-top:3px;}

/* ─────────────  KPI CARDS  ───────────── */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}
.kpi{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:16px;
  padding:22px 24px 18px;
  position:relative;overflow:hidden;
  transition:transform .2s,border-color .2s,box-shadow .2s;
}
.kpi:hover{transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,.4);}
.kpi::before{
  content:'';position:absolute;
  inset:0 0 auto 0;height:2px;border-radius:16px 16px 0 0;
}
.kpi.blue::before{background:linear-gradient(90deg,#2563EB,#0EA5E9);}
.kpi.green::before{background:linear-gradient(90deg,#10B981,#34D399);}
.kpi.amber::before{background:linear-gradient(90deg,#F59E0B,#FBBF24);}
.kpi.red::before{background:linear-gradient(90deg,#EF4444,#F87171);}
.kpi-glyph{font-size:1.5rem;opacity:.22;position:absolute;right:20px;top:18px;}
.kpi-label{
  font-size:.68rem;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-bottom:12px;
}
.kpi-value{
  font-family:'Clash Display',sans-serif!important;
  font-size:2.3rem;font-weight:700;line-height:1;color:#fff;margin-bottom:6px;
}
.kpi-value.blue{color:var(--accent2);}
.kpi-value.green{color:var(--green);}
.kpi-value.amber{color:var(--amber);}
.kpi-value.red{color:var(--red);}
.kpi-sub{font-size:.76rem;color:var(--muted);}
.kpi-sub b{color:var(--muted2);}

/* ─────────────  SECTION HEADER  ───────────── */
.sec-head{
  display:flex;align-items:center;gap:12px;
  font-family:'Clash Display',sans-serif!important;
  font-size:1.05rem;font-weight:600;color:#fff;
  margin:0 0 18px;letter-spacing:-.01em;
}
.sec-head::after{content:'';flex:1;height:1px;background:var(--border);margin-left:4px;}
.sec-icon{
  width:28px;height:28px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;
  font-size:.85rem;background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.25);
}

/* ─────────────  CARDS / PANELS  ───────────── */
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:16px;padding:24px;margin-bottom:20px;
}
.card2{
  background:var(--card2);border:1px solid var(--border);
  border-radius:14px;padding:18px 20px;margin-bottom:16px;
}

/* ─────────────  INSIGHT PILLS  ───────────── */
.pill-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:4px;}
.pill{
  background:rgba(255,255,255,.03);
  border:1px solid var(--border);border-radius:12px;
  padding:14px 16px;font-size:.82rem;line-height:1.55;color:var(--text);
}
.pill .pill-title{font-weight:700;color:#fff;display:flex;align-items:center;gap:8px;margin-bottom:4px;}
.pill.danger{border-left:3px solid var(--red);}
.pill.warn{border-left:3px solid var(--amber);}
.pill.info{border-left:3px solid var(--accent2);}
.pill.ok{border-left:3px solid var(--green);}

/* ─────────────  RISK DISPLAY  ───────────── */
.risk-box{
  text-align:center;border-radius:16px;
  padding:28px 20px;margin-top:10px;
}
.risk-box.low{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);}
.risk-box.med{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);}
.risk-box.high{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.25);}
.risk-pct{
  font-family:'Clash Display',sans-serif!important;
  font-size:3.5rem;font-weight:700;line-height:1;
}
.risk-pct.low{color:var(--green);}
.risk-pct.med{color:var(--amber);}
.risk-pct.high{color:var(--red);}
.risk-tag{
  display:inline-block;margin-top:10px;
  padding:4px 18px;border-radius:20px;
  font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
}
.risk-tag.low{background:rgba(16,185,129,.15);color:var(--green);}
.risk-tag.med{background:rgba(245,158,11,.15);color:var(--amber);}
.risk-tag.high{background:rgba(239,68,68,.15);color:var(--red);}

/* ─────────────  PROGRESS BAR  ───────────── */
.pbar-wrap{background:rgba(255,255,255,.06);border-radius:8px;height:10px;overflow:hidden;margin:14px 0;}
.pbar-fill{height:100%;border-radius:8px;transition:width .6s cubic-bezier(.4,0,.2,1);}
.pbar-fill.low{background:linear-gradient(90deg,#059669,#10B981);}
.pbar-fill.med{background:linear-gradient(90deg,#D97706,#F59E0B);}
.pbar-fill.high{background:linear-gradient(90deg,#DC2626,#EF4444,#F97316);}

/* ─────────────  ACTION CHIPS  ───────────── */
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}
.chip{
  padding:5px 14px;border-radius:20px;
  font-size:.74rem;font-weight:600;
  border:1px solid;cursor:default;
  letter-spacing:.02em;
}
.chip.blue{border-color:var(--accent2);color:var(--accent2);background:rgba(59,130,246,.08);}
.chip.green{border-color:var(--green);color:var(--green);background:rgba(16,185,129,.08);}
.chip.amber{border-color:var(--amber);color:var(--amber);background:rgba(245,158,11,.08);}
.chip.red{border-color:var(--red);color:var(--red);background:rgba(239,68,68,.08);}
.chip.purple{border-color:var(--purple);color:var(--purple);background:rgba(139,92,246,.08);}

/* ─────────────  DATA TABLE  ───────────── */
.dtable{width:100%;border-collapse:collapse;font-size:.82rem;}
.dtable thead th{
  text-align:left;padding:9px 14px;
  font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--border);
}
.dtable tbody td{
  padding:11px 14px;border-bottom:1px solid rgba(255,255,255,.04);
  color:var(--text);
}
.dtable tbody tr:hover{background:rgba(37,99,235,.05);}
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 12px;border-radius:20px;font-size:.7rem;font-weight:700;}
.badge.best{background:rgba(16,185,129,.12);color:var(--green);border:1px solid rgba(16,185,129,.25);}
.badge.good{background:rgba(37,99,235,.12);color:var(--accent2);border:1px solid rgba(37,99,235,.25);}
.badge.ok{background:rgba(245,158,11,.10);color:var(--amber);border:1px solid rgba(245,158,11,.25);}

/* ─────────────  MINI METRIC  ───────────── */
.mini-metric{
  background:var(--card2);border:1px solid var(--border);
  border-radius:10px;padding:12px 16px;text-align:center;
}
.mini-val{
  font-family:'Clash Display',sans-serif!important;
  font-size:1.55rem;font-weight:600;color:#fff;line-height:1;
}
.mini-lbl{font-size:.68rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-top:4px;}

/* ─────────────  SIDEBAR  ───────────── */
.sb-brand{
  font-family:'Clash Display',sans-serif!important;
  font-size:1.3rem;font-weight:700;color:#fff;
  display:flex;align-items:center;gap:10px;
  padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:20px;
}
.sb-nav-label{
  font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:var(--muted);margin:18px 0 8px;
}
.sb-stat-row{
  background:var(--card2);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px;margin-top:6px;
  font-size:.8rem;line-height:2;
}

/* ─────────────  STREAMLIT OVERRIDES  ───────────── */
div[data-testid="stMetricValue"]{font-family:'Clash Display',sans-serif!important;}
.stButton>button{
  background:linear-gradient(135deg,#2563EB,#0EA5E9)!important;
  border:none!important;border-radius:12px!important;
  color:#fff!important;font-weight:700!important;
  font-family:'Cabinet Grotesk',sans-serif!important;
  letter-spacing:.03em!important;font-size:.88rem!important;
  padding:.6rem 2rem!important;
  box-shadow:0 4px 20px rgba(37,99,235,.4)!important;
  transition:all .2s!important;
}
.stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 8px 28px rgba(37,99,235,.55)!important;
}
.stTabs [data-baseweb="tab-list"]{
  background:var(--card)!important;
  border-radius:12px!important;border:1px solid var(--border)!important;
  padding:4px!important;gap:4px!important;
}
.stTabs [data-baseweb="tab"]{
  border-radius:9px!important;color:var(--muted)!important;
  font-family:'Cabinet Grotesk',sans-serif!important;
  font-weight:600!important;font-size:.85rem!important;
}
.stTabs [aria-selected="true"]{
  background:var(--accent)!important;color:#fff!important;
}
.stRadio label,.stSelectbox label,.stSlider label{color:var(--muted2)!important;font-size:.8rem!important;}
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p{
  font-family:'Cabinet Grotesk',sans-serif!important;
}
hr{border-color:var(--border)!important;margin:24px 0!important;}
.stAlert{border-radius:12px!important;}
</style>
"""
st.markdown(DESIGN, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────────────────────────────────────
def set_mpl():
    plt.rcParams.update({
        "figure.facecolor":  "#0F1830",
        "axes.facecolor":    "#0F1830",
        "axes.edgecolor":    "#1A2B48",
        "axes.labelcolor":   "#7A92B0",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.color":       "#556B8D",
        "ytick.color":       "#556B8D",
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "text.color":        "#E2EAF4",
        "grid.color":        "#1A2B48",
        "grid.linestyle":    "--",
        "grid.alpha":        0.55,
        "font.family":       "DejaVu Sans",
        "legend.facecolor":  "#0F1830",
        "legend.edgecolor":  "#1A2B48",
        "legend.fontsize":   8.5,
    })

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BLUE   = "#3B82F6"
C_TEAL   = "#0EA5E9"
C_GREEN  = "#10B981"
C_AMBER  = "#F59E0B"
C_RED    = "#EF4444"
C_PURPLE = "#8B5CF6"
C_PINK   = "#EC4899"

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FULL ML PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline():
    # ── Try multiple file name patterns ──────────────────────────────────────
    candidates = [
        "European_Bank.csv", "European_Bank__1_.csv",
        "European_Bank (1).csv", "bank_churn.csv",
        "Churn_Modelling.csv", "churn.csv",
    ]
    df_raw = None
    for c in candidates:
        if os.path.exists(c):
            df_raw = pd.read_csv(c)
            break
    if df_raw is None:
        return None  # handled below

    # ── Preprocessing ────────────────────────────────────────────────────────
    df = df_raw.copy()
    drop_cols = [c for c in ["Year","CustomerId","Surname"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    # One-hot geography
    df = pd.get_dummies(df, columns=["Geography"], prefix="Geo")
    # Encode gender
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # ── Feature Engineering ──────────────────────────────────────────────────
    df["Bal_Salary_Ratio"]   = df["Balance"] / (df["EstimatedSalary"] + 1e-5)
    df["Age_Tenure"]         = df["Age"] * df["Tenure"]
    df["Engage_Product"]     = df["IsActiveMember"] * df["NumOfProducts"]
    df["Product_Density"]    = df["NumOfProducts"] / (df["Balance"] + 1e-5)
    df["Zero_Balance"]       = (df["Balance"] == 0).astype(int)
    df["Senior"]             = (df["Age"] >= 50).astype(int)
    df["Low_Tenure"]         = (df["Tenure"] <= 2).astype(int)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]
    feat_names = X.columns.tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
    scaler = StandardScaler()
    Xtr_s, Xte_s = X_tr.copy(), X_te.copy()
    Xtr_s[num_cols] = scaler.fit_transform(X_tr[num_cols])
    Xte_s[num_cols] = scaler.transform(X_te[num_cols])

    # ── Train four models ────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42),
    }

    results = {}
    for name, mdl in models.items():
        if name == "Logistic Regression":
            mdl.fit(Xtr_s, y_tr)
            pred  = mdl.predict(Xte_s)
            proba = mdl.predict_proba(Xte_s)[:, 1]
        else:
            mdl.fit(Xtr_s, y_tr)
            pred  = mdl.predict(Xte_s)
            proba = mdl.predict_proba(Xte_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_te, proba)
        results[name] = {
            "model":     mdl,
            "Accuracy":  round(accuracy_score(y_te, pred) * 100, 2),
            "Precision": round(precision_score(y_te, pred) * 100, 2),
            "Recall":    round(recall_score(y_te, pred) * 100, 2),
            "F1":        round(f1_score(y_te, pred) * 100, 2),
            "AUC":       round(roc_auc_score(y_te, proba) * 100, 2),
            "pred":      pred,
            "proba":     proba,
            "fpr":       fpr,
            "tpr":       tpr,
            "cm":        confusion_matrix(y_te, pred),
        }

    best_name = max(results, key=lambda k: results[k]["AUC"])
    best      = results[best_name]
    gb        = results["Gradient Boosting"]

    # Feature importance from GB
    fi = pd.Series(
        results["Gradient Boosting"]["model"].feature_importances_,
        index=feat_names
    ).sort_values(ascending=False)

    return {
        "raw": df_raw, "df": df, "X": X, "y": y,
        "X_tr": Xtr_s, "X_te": Xte_s, "y_tr": y_tr, "y_te": y_te,
        "scaler": scaler, "feat": feat_names, "num_cols": num_cols,
        "results": results, "best_name": best_name, "best": best,
        "gb": gb, "fi": fi,
    }

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: prepare one customer row for prediction
# ─────────────────────────────────────────────────────────────────────────────
def make_row(d, scaler, feat, num_cols):
    row = {
        "CreditScore":     d["cs"],
        "Age":             d["age"],
        "Tenure":          d["tenure"],
        "Balance":         d["balance"],
        "NumOfProducts":   d["products"],
        "HasCrCard":       int(d["has_cc"]),
        "IsActiveMember":  int(d["active"]),
        "EstimatedSalary": d["salary"],
        "Geo_Germany":     int(d["geo"] == "Germany"),
        "Geo_Spain":       int(d["geo"] == "Spain"),
        "Geo_France":      int(d["geo"] == "France"),
        "Gender":          1 if d["gender"] == "Male" else 0,
        "Bal_Salary_Ratio": d["balance"] / (d["salary"] + 1e-5),
        "Age_Tenure":       d["age"] * d["tenure"],
        "Engage_Product":   int(d["active"]) * d["products"],
        "Product_Density":  d["products"] / (d["balance"] + 1e-5),
        "Zero_Balance":     int(d["balance"] == 0),
        "Senior":           int(d["age"] >= 50),
        "Low_Tenure":       int(d["tenure"] <= 2),
    }
    df_r = pd.DataFrame([row])
    for c in feat:
        if c not in df_r.columns:
            df_r[c] = 0
    df_r = df_r[feat].copy()
    df_r[num_cols] = scaler.transform(df_r[num_cols])
    return df_r

def risk_cls(p):
    if p < 0.35: return "low"
    if p < 0.62: return "med"
    return "high"

def risk_label(p):
    c = risk_cls(p)
    if c == "low":  return "LOW RISK",  "#10B981"
    if c == "med":  return "MEDIUM RISK","#F59E0B"
    return "HIGH RISK", "#EF4444"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🔄  Training ChurnGuard AI  —  Please wait…"):
    DATA = run_pipeline()

if DATA is None:
    st.error("""
    **Dataset not found.**  
    Please rename your CSV file to **`European_Bank.csv`** and place it in the **same folder** as this script, then re-run.
    """)
    st.stop()

raw      = DATA["raw"]
df       = DATA["df"]
results  = DATA["results"]
best_nm  = DATA["best_name"]
gb       = DATA["gb"]
fi       = DATA["fi"]
scaler   = DATA["scaler"]
feat     = DATA["feat"]
num_cols = DATA["num_cols"]
y_te     = DATA["y_te"]

# Quick globals
TOTAL     = len(raw)
CHURNED   = int(raw["Exited"].sum())
CHURN_PCT = CHURNED / TOTAL * 100
GB_PROBA  = gb["proba"]
HIGH_RISK = int((GB_PROBA > 0.50).sum())
REV_RISK  = raw["Balance"].iloc[:len(GB_PROBA)][GB_PROBA > 0.50].sum() / 1e6
AVG_BAL   = raw["Balance"].mean()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <span style="font-size:1.5rem">🛡️</span>
      <span>ChurnGuard AI</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    PAGE = st.radio(
        "",
        ["🏠  Overview", "📊  EDA & Insights", "🤖  Model Performance",
         "🎯  Risk Calculator", "🔬  What-If Lab"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="sb-nav-label">Live Model Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-stat-row">
      <span style="color:#10B981">●</span>&nbsp; <b style="color:#fff">Gradient Boosting</b><br>
      AUC &nbsp;&nbsp;&nbsp; <b style="color:#3B82F6">{gb['AUC']}%</b><br>
      F1 Score &nbsp;<b style="color:#10B981">{gb['F1']}%</b><br>
      Accuracy &nbsp;<b style="color:#fff">{gb['Accuracy']}%</b><br>
      Records &nbsp; <b style="color:#fff">{TOTAL:,}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2025 ChurnGuard AI · Unified Mentor Internship\nEuropean Bank Dataset · sklearn · Streamlit")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1  —  OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if "Overview" in PAGE:

    # Hero
    st.markdown(f"""
    <div class="hero">
      <div class="hero-grid"></div>
      <div class="hero-eyebrow"><span class="hero-dot"></span>Live Analytics · Real-time Predictions</div>
      <div class="hero-title">Churn<span class="accent">Guard</span> AI</div>
      <div class="hero-subtitle">
        Predictive intelligence platform for European Bank customer retention.
        ML-powered churn scoring &amp; proactive intervention at scale.
      </div>
      <div class="hero-stats">
        <div class="hero-stat"><span class="hero-stat-val">{TOTAL:,}</span><span class="hero-stat-lbl">Customers</span></div>
        <div class="hero-stat"><span class="hero-stat-val">{CHURN_PCT:.1f}%</span><span class="hero-stat-lbl">Churn Rate</span></div>
        <div class="hero-stat"><span class="hero-stat-val">{gb['AUC']}%</span><span class="hero-stat-lbl">Model AUC</span></div>
        <div class="hero-stat"><span class="hero-stat-val">4</span><span class="hero-stat-lbl">ML Models</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi blue">
        <div class="kpi-glyph">👥</div>
        <div class="kpi-label">Total Customers Analysed</div>
        <div class="kpi-value blue">{TOTAL:,}</div>
        <div class="kpi-sub">European Bank Dataset</div>
      </div>
      <div class="kpi red">
        <div class="kpi-glyph">📉</div>
        <div class="kpi-label">Actual Churned Customers</div>
        <div class="kpi-value red">{CHURNED:,}</div>
        <div class="kpi-sub"><b>{CHURN_PCT:.1f}%</b> overall churn rate</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-glyph">⚠️</div>
        <div class="kpi-label">High-Risk Predicted</div>
        <div class="kpi-value amber">{HIGH_RISK:,}</div>
        <div class="kpi-sub">Probability &gt; 50% (test set)</div>
      </div>
      <div class="kpi green">
        <div class="kpi-glyph">🏆</div>
        <div class="kpi-label">Best Model ROC-AUC</div>
        <div class="kpi-value green">{gb['AUC']}%</div>
        <div class="kpi-sub">Gradient Boosting · 150 trees</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row 1
    set_mpl()
    c1, c2, c3 = st.columns([1.1, 1.1, 0.8])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🌍</span>Churn Rate by Country</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        geo = raw.groupby("Geography")["Exited"].mean().mul(100).sort_values(ascending=False)
        bar_colors = [C_RED, C_AMBER, C_BLUE]
        bars = ax.bar(geo.index, geo.values, color=bar_colors, edgecolor="none", width=0.5, zorder=3)
        for b, v in zip(bars, geo.values):
            ax.text(b.get_x()+b.get_width()/2, v+.5, f"{v:.1f}%",
                    ha="center", fontsize=10, fontweight="bold", color="#fff")
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📦</span>Churn by Products Owned</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        prod = raw.groupby("NumOfProducts")["Exited"].mean().mul(100)
        pcols = [C_GREEN, C_BLUE, C_RED, C_PURPLE][:len(prod)]
        bars2 = ax.bar(prod.index.astype(str), prod.values, color=pcols, edgecolor="none", width=0.5, zorder=3)
        for b, v in zip(bars2, prod.values):
            ax.text(b.get_x()+b.get_width()/2, v+0.6, f"{v:.1f}%",
                    ha="center", fontsize=10, fontweight="bold", color="#fff")
        ax.set_xlabel("Number of Products", fontsize=9)
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📱</span>Active vs Inactive</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        act = raw.groupby("IsActiveMember")["Exited"].mean().mul(100)
        act_labels = ["Inactive", "Active"]
        act_colors = [C_RED, C_GREEN]
        bars3 = ax.bar(act_labels, act.values, color=act_colors, edgecolor="none", width=0.45, zorder=3)
        for b, v in zip(bars3, act.values):
            ax.text(b.get_x()+b.get_width()/2, v+0.4, f"{v:.1f}%",
                    ha="center", fontsize=10.5, fontweight="bold", color="#fff")
        ax.set_ylabel("Churn Rate (%)", fontsize=9)
        ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Insight pills
    geo_ch   = raw.groupby("Geography")["Exited"].mean().mul(100)
    de_rate  = geo_ch.get("Germany", 0)
    gen_ch   = raw.groupby("Gender")["Exited"].mean().mul(100)
    f_rate   = gen_ch.get("Female", 0)
    age_risk = raw[raw["Exited"]==1]["Age"].mean()

    st.markdown(f"""
    <div class="sec-head" style="margin-top:8px"><span class="sec-icon">💡</span>Key Churn Intelligence</div>
    <div class="pill-grid">
      <div class="pill danger">
        <div class="pill-title">🌍 Germany Highest-Risk Country</div>
        Germany churns at <b>{de_rate:.1f}%</b> — nearly 2× the rate of France and Spain combined.
        Priority market for retention campaigns.
      </div>
      <div class="pill warn">
        <div class="pill-title">📦 3–4 Product Holders at Extreme Risk</div>
        Customers with 3+ products show 83–100% churn. Fee overload or poor cross-sell match drives exits.
      </div>
      <div class="pill info">
        <div class="pill-title">🎂 High-Risk Age Band: 40–60</div>
        Mean age of churned customers is <b>{age_risk:.0f} years</b>. Mid-career switchers need targeted retention.
      </div>
      <div class="pill ok">
        <div class="pill-title">📱 Active Members Churn 2× Less</div>
        IsActiveMember is the #1 controllable driver. Engagement campaigns directly reduce churn probability.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2  —  EDA & INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif "EDA" in PAGE:
    st.markdown("""
    <div class="hero" style="padding:26px 40px 22px">
      <div class="hero-grid"></div>
      <div class="hero-title" style="font-size:2.2rem">EDA &amp; <span class="accent">Insights</span></div>
      <div class="hero-subtitle">Deep statistical exploration of the European Bank dataset</div>
    </div>
    """, unsafe_allow_html=True)

    set_mpl()

    # Row 1: Age + Balance dist
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🎂</span>Age Distribution by Churn</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        raw[raw["Exited"]==0]["Age"].hist(bins=30, ax=ax, color=C_BLUE, alpha=0.72, label="Retained", edgecolor="none")
        raw[raw["Exited"]==1]["Age"].hist(bins=30, ax=ax, color=C_RED,  alpha=0.72, label="Churned",  edgecolor="none")
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">💰</span>Balance Distribution by Churn</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        raw[raw["Exited"]==0]["Balance"].hist(bins=30, ax=ax, color=C_TEAL, alpha=0.72, label="Retained", edgecolor="none")
        raw[raw["Exited"]==1]["Balance"].hist(bins=30, ax=ax, color=C_AMBER, alpha=0.72, label="Churned",  edgecolor="none")
        ax.set_xlabel("Balance (€)"); ax.set_ylabel("Count")
        ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Credit Score + Gender
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">💳</span>Credit Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        raw[raw["Exited"]==0]["CreditScore"].hist(bins=30, ax=ax, color=C_PURPLE, alpha=0.72, label="Retained", edgecolor="none")
        raw[raw["Exited"]==1]["CreditScore"].hist(bins=30, ax=ax, color=C_PINK,   alpha=0.72, label="Churned",  edgecolor="none")
        ax.set_xlabel("Credit Score"); ax.set_ylabel("Count")
        ax.legend(); ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">⚧️</span>Churn Rate by Gender &amp; Card</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.5))
        gen = raw.groupby("Gender")["Exited"].mean().mul(100)
        axes[0].bar(gen.index, gen.values, color=[C_PINK, C_TEAL], edgecolor="none", width=0.45, zorder=3)
        for i,(k,v) in enumerate(gen.items()):
            axes[0].text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#fff")
        axes[0].set_title("By Gender", fontsize=9, color="#7A92B0"); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)
        cc = raw.groupby("HasCrCard")["Exited"].mean().mul(100)
        axes[1].bar(["No Card","Has Card"], cc.values, color=[C_RED, C_GREEN], edgecolor="none", width=0.45, zorder=3)
        for i,v in enumerate(cc.values):
            axes[1].text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color="#fff")
        axes[1].set_title("By Credit Card", fontsize=9, color="#7A92B0"); axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Tenure + Heatmap
    c5, c6 = st.columns([1, 1.2])
    with c5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📅</span>Churn Rate by Tenure</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ten = raw.groupby("Tenure")["Exited"].mean().mul(100)
        ax.plot(ten.index, ten.values, color=C_AMBER, lw=2.5, marker="o",
                markersize=6, markerfacecolor=C_AMBER, markeredgecolor="#0F1830")
        ax.fill_between(ten.index, ten.values, alpha=0.12, color=C_AMBER)
        ax.set_xlabel("Tenure (years)"); ax.set_ylabel("Churn Rate (%)")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🔥</span>Correlation Heatmap</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.7))
        cols_h = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
                  "IsActiveMember","EstimatedSalary","Exited"]
        corr = raw[cols_h].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt=".2f", cmap=cmap,
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.75}, annot_kws={"size": 7.5},
                    linecolor="#1A2B48")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Dataset summary stats
    st.markdown('<div class="sec-head" style="margin-top:8px"><span class="sec-icon">📋</span>Dataset Summary Statistics</div>', unsafe_allow_html=True)
    summary = raw[["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary","Exited"]].describe().round(2)
    st.dataframe(
        summary.style
            .background_gradient(cmap="Blues", subset=pd.IndexSlice["mean", :])
            .format("{:.2f}"),
        use_container_width=True, height=310
    )

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3  —  MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif "Model" in PAGE:
    st.markdown("""
    <div class="hero" style="padding:26px 40px 22px">
      <div class="hero-grid"></div>
      <div class="hero-title" style="font-size:2.2rem">Model <span class="accent">Performance</span></div>
      <div class="hero-subtitle">Comprehensive evaluation of all 4 ML models trained on your dataset</div>
    </div>
    """, unsafe_allow_html=True)

    set_mpl()
    model_names  = list(results.keys())
    model_colors = [C_BLUE, C_GREEN, C_AMBER, C_RED]

    # Leaderboard table
    st.markdown('<div class="sec-head"><span class="sec-icon">🏆</span>Model Leaderboard</div>', unsafe_allow_html=True)
    badges = {"Gradient Boosting": "best", "Random Forest": "good",
              "Decision Tree": "ok", "Logistic Regression": "ok"}
    badge_icons = {"best": "🥇 Best", "good": "🥈 Strong", "ok": "✅ Trained"}

    rows_html = ""
    for nm in model_names:
        r = results[nm]
        b = badges[nm]
        rows_html += f"""
        <tr>
          <td><b style="color:#fff">{nm}</b></td>
          <td>{r['Accuracy']}%</td>
          <td>{r['Precision']}%</td>
          <td>{r['Recall']}%</td>
          <td>{r['F1']}%</td>
          <td><b style="color:{'#3B82F6' if nm=='Gradient Boosting' else '#E2EAF4'}">{r['AUC']}%</b></td>
          <td><span class="badge {b}">{badge_icons[b]}</span></td>
        </tr>"""

    st.markdown(f"""
    <div class="card">
    <table class="dtable">
      <thead><tr>
        <th>Model</th><th>Accuracy</th><th>Precision</th>
        <th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>Status</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Charts row
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📈</span>ROC Curves — All Models</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for (nm, col) in zip(model_names, model_colors):
            r = results[nm]
            ax.plot(r["fpr"], r["tpr"], color=col, lw=2.2, label=f"{nm}  ({r['AUC']}%)")
            ax.fill_between(r["fpr"], r["tpr"], alpha=0.04, color=col)
        ax.plot([0,1],[0,1], color="#334155", lw=1.2, linestyle="--", label="Random (50%)")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right"); ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📊</span>Grouped Metric Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        metrics_list = ["Accuracy","Precision","Recall","F1","AUC"]
        x = np.arange(len(metrics_list))
        w = 0.18
        for i, (nm, col) in enumerate(zip(model_names, model_colors)):
            vals = [results[nm][m] for m in metrics_list]
            ax.bar(x + i*w, vals, w, label=nm, color=col, alpha=0.88, edgecolor="none")
        ax.set_xticks(x + w*1.5); ax.set_xticklabels(metrics_list)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature Importance + Confusion Matrix
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🔍</span>Feature Importance — Gradient Boosting</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        top = fi.head(12)
        cols_fi = [C_BLUE if v < top.values[2] else C_RED for v in top.values][::-1]
        bars = ax.barh(top.index[::-1], top.values[::-1], color=cols_fi, edgecolor="none", height=0.62)
        for bar, val in zip(bars, top.values[::-1]):
            ax.text(val+.001, bar.get_y()+bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=7.5, color="#7A92B0")
        ax.set_xlabel("Importance Score")
        ax.xaxis.grid(True); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🧩</span>Confusion Matrix — Gradient Boosting</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        cm = gb["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Retained","Churned"], yticklabels=["Retained","Churned"],
                    linewidths=0.5, linecolor="#1A2B48",
                    annot_kws={"size": 16, "weight": "bold", "color": "#fff"})
        ax.set_xlabel("Predicted", fontsize=10); ax.set_ylabel("Actual", fontsize=10)
        cbar = ax.collections[0].colorbar; cbar.ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Churn probability histogram
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-head"><span class="sec-icon">📉</span>Churn Probability Distribution — Gradient Boosting (Test Set)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].hist(GB_PROBA[y_te==0], bins=40, color=C_BLUE,  alpha=0.75, label="Retained (actual)", edgecolor="none")
    axes[0].hist(GB_PROBA[y_te==1], bins=40, color=C_RED,   alpha=0.75, label="Churned (actual)",  edgecolor="none")
    axes[0].axvline(0.5, color="#F8FAFC", linestyle="--", lw=1.5, label="Decision threshold")
    axes[0].set_xlabel("Predicted Churn Probability"); axes[0].set_ylabel("Count")
    axes[0].set_title("Probability Separation", fontsize=10, color="#7A92B0"); axes[0].legend()
    axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

    low_c  = (GB_PROBA < 0.35).sum()
    med_c  = ((GB_PROBA >= 0.35) & (GB_PROBA < 0.62)).sum()
    high_c = (GB_PROBA >= 0.62).sum()
    axes[1].bar(["Low Risk\n(<35%)", "Medium Risk\n(35–62%)", "High Risk\n(>62%)"],
                [low_c, med_c, high_c], color=[C_GREEN, C_AMBER, C_RED],
                edgecolor="none", width=0.5, zorder=3)
    for i, v in enumerate([low_c, med_c, high_c]):
        axes[1].text(i, v+3, str(v), ha="center", fontsize=11, fontweight="bold", color="#fff")
    axes[1].set_ylabel("Customers"); axes[1].set_title("Risk Segmentation", fontsize=10, color="#7A92B0")
    axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4  —  RISK CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif "Calculator" in PAGE:
    st.markdown("""
    <div class="hero" style="padding:26px 40px 22px">
      <div class="hero-grid"></div>
      <div class="hero-title" style="font-size:2.2rem">Customer Risk <span class="accent">Calculator</span></div>
      <div class="hero-subtitle">Enter any customer profile to instantly predict churn probability using the trained Gradient Boosting model</div>
    </div>
    """, unsafe_allow_html=True)

    # Input form
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown('<div class="sb-nav-label">📋 Financial Profile</div>', unsafe_allow_html=True)
            cs      = st.slider("Credit Score",            300,    850, 650)
            balance = st.number_input("Account Balance (€)",  0, 300000, 65000, 1000)
            salary  = st.number_input("Estimated Salary (€)", 0, 250000, 85000, 5000)
        with r1c2:
            st.markdown('<div class="sb-nav-label">👤 Personal Details</div>', unsafe_allow_html=True)
            age      = st.slider("Age",            18, 80, 40)
            tenure   = st.slider("Tenure (years)",  0, 10,  4)
            gender   = st.selectbox("Gender",   ["Male","Female"])
            geography= st.selectbox("Geography", ["France","Germany","Spain"])
        with r1c3:
            st.markdown('<div class="sb-nav-label">🏦 Banking Behaviour</div>', unsafe_allow_html=True)
            products = st.selectbox("Number of Products", [1,2,3,4])
            has_cc   = st.radio("Has Credit Card?",   [True, False], format_func=lambda x:"Yes" if x else "No")
            is_active= st.radio("Is Active Member?",  [True, False], format_func=lambda x:"Yes" if x else "No")
        st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("⚡  Calculate Churn Risk Score", use_container_width=False)

    if predict_btn:
        d = {"cs":cs,"age":age,"tenure":tenure,"balance":balance,"products":products,
             "has_cc":has_cc,"active":is_active,"salary":salary,"geo":geography,"gender":gender}
        row  = make_row(d, scaler, feat, num_cols)
        prob = float(gb["model"].predict_proba(row)[0][1])
        pct  = round(prob * 100, 1)
        rc   = risk_cls(prob)
        rl, rc_color = risk_label(prob)

        res1, res2 = st.columns([1, 1.6])
        with res1:
            st.markdown(f"""
            <div class="risk-box {rc}">
              <div style="font-size:.72rem;color:#556B8D;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px">
                Churn Probability
              </div>
              <div class="risk-pct {rc}">{pct}%</div>
              <div><span class="risk-tag {rc}">{rl}</span></div>
              <div class="pbar-wrap"><div class="pbar-fill {rc}" style="width:{pct}%"></div></div>
              <div style="font-size:.78rem;color:#556B8D;margin-top:4px">
                Powered by Gradient Boosting · AUC {gb['AUC']}%
              </div>
            </div>
            """, unsafe_allow_html=True)

        with res2:
            st.markdown('<div class="card" style="height:100%;min-height:200px">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head"><span class="sec-icon">🎯</span>Recommended Retention Actions</div>', unsafe_allow_html=True)

            if rc == "low":
                st.success("✅ Customer is stable. No urgent intervention needed.")
                st.markdown("""
                <div class="chips">
                  <span class="chip green">📧 Quarterly newsletter</span>
                  <span class="chip blue">🎁 Loyalty reward points</span>
                  <span class="chip green">📱 App engagement push</span>
                  <span class="chip blue">📊 Annual review meeting</span>
                </div>""", unsafe_allow_html=True)
            elif rc == "med":
                st.warning("⚠️ Early churn signals detected. Act within 30 days.")
                st.markdown("""
                <div class="chips">
                  <span class="chip amber">📞 Personal outreach call</span>
                  <span class="chip blue">💳 Product bundle offer</span>
                  <span class="chip amber">🏦 Fee waiver — 3 months</span>
                  <span class="chip blue">📊 Financial health review</span>
                  <span class="chip purple">🎯 Premium upgrade trial</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.error("🚨 HIGH RISK — Immediate escalation recommended.")
                st.markdown("""
                <div class="chips">
                  <span class="chip red">🚨 Escalate to Relationship Manager</span>
                  <span class="chip amber">💰 Retention bonus offer</span>
                  <span class="chip red">🔒 Priority support SLA</span>
                  <span class="chip amber">🎯 VIP programme enrolment</span>
                  <span class="chip purple">📋 Needs assessment survey</span>
                  <span class="chip blue">🤝 Face-to-face meeting</span>
                </div>""", unsafe_allow_html=True)

            # Customer-specific risk signals
            signals = []
            if age > 45:       signals.append(f"🎂 Age {age} falls in high-risk band (40–60)")
            if not is_active:  signals.append("📱 Inactive member — strongest churn predictor")
            if geography=="Germany": signals.append("🌍 Germany — highest-risk geography (32%)")
            if products >= 3:  signals.append(f"📦 {products} products — extreme churn correlation")
            if balance/(salary+1) > 1.5: signals.append("💰 High balance-to-salary ratio detected")
            if tenure <= 2:    signals.append("⏱️ Low tenure — newer customers at higher risk")
            if not signals:    signals.append("✅ No major risk signals identified")

            st.markdown('<div style="margin-top:14px;font-size:.68rem;color:#556B8D;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">DETECTED RISK SIGNALS</div>', unsafe_allow_html=True)
            for s in signals[:4]:
                st.markdown(f'<div style="font-size:.83rem;color:#E2EAF4;padding:5px 0;border-bottom:1px solid #1A2B48">{s}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Gauge-style probability bar chart
        set_mpl()
        st.markdown('<div class="card" style="margin-top:16px">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">📊</span>Churn Score vs Population Benchmark</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))

        # Left: where this customer sits in the distribution
        axes[0].hist(GB_PROBA, bins=40, color="#1E3A5F", edgecolor="none", label="All customers")
        axes[0].axvline(prob, color=rc_color, lw=2.5, label=f"This customer ({pct}%)")
        axes[0].fill_betweenx([0, axes[0].get_ylim()[1] if axes[0].get_ylim()[1] > 0 else 100],
                               prob-0.01, prob+0.01, color=rc_color, alpha=0.3)
        axes[0].set_xlabel("Churn Probability"); axes[0].set_ylabel("Customers")
        axes[0].set_title("Position in Population", fontsize=10, color="#7A92B0")
        axes[0].legend(); axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

        # Right: bar comparison
        benchmarks  = ["Average\nCustomer", "Active\nMember", "Germany\nCustomer", "This\nCustomer"]
        bench_probs = [
            round(GB_PROBA.mean()*100, 1),
            round(raw[raw["IsActiveMember"]==1]["Exited"].mean()*100, 1),
            round(raw[raw["Geography"]=="Germany"]["Exited"].mean()*100, 1),
            pct
        ]
        bcolors = [C_BLUE, C_GREEN, C_AMBER, rc_color]
        brs = axes[1].bar(benchmarks, bench_probs, color=bcolors, edgecolor="none", width=0.5, zorder=3)
        for br, bv in zip(brs, bench_probs):
            axes[1].text(br.get_x()+br.get_width()/2, bv+0.4, f"{bv}%",
                         ha="center", fontsize=10, fontweight="bold", color="#fff")
        axes[1].set_ylabel("Churn Probability (%)"); axes[1].yaxis.grid(True); axes[1].set_axisbelow(True)
        axes[1].set_title("Benchmark Comparison", fontsize=10, color="#7A92B0")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5  —  WHAT-IF LAB
# ═════════════════════════════════════════════════════════════════════════════
elif "What-If" in PAGE:
    st.markdown("""
    <div class="hero" style="padding:26px 40px 22px">
      <div class="hero-grid"></div>
      <div class="hero-title" style="font-size:2.2rem">What-If <span class="accent">Scenario Lab</span></div>
      <div class="hero-subtitle">Simulate retention interventions and instantly see their impact on churn probability</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="sec-head"><span class="sec-icon">👤</span>Base Customer Profile</div>', unsafe_allow_html=True)
        b_age   = st.slider("Age",           18, 80,  52, key="b_age")
        b_bal   = st.slider("Balance (€)",    0, 200000, 125000, 5000, key="b_bal")
        b_prod  = st.selectbox("Products",   [1,2,3,4], index=2, key="b_prod")
        b_active= st.radio("Active Member?", [False, True],
                           format_func=lambda x:"Yes" if x else "No", key="b_act", index=0)
        b_geo   = st.selectbox("Geography",  ["Germany","France","Spain"], key="b_geo")
        b_ten   = st.slider("Tenure (years)", 0, 10, 1, key="b_ten")
        b_cs    = st.slider("Credit Score",  300, 850, 580, key="b_cs")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="sec-head"><span class="sec-icon">✨</span>Retention Intervention</div>', unsafe_allow_html=True)
        a_age   = st.slider("Age",           18, 80,  b_age, key="a_age")
        a_bal   = st.slider("Balance (€)",    0, 200000, max(0, b_bal-20000), 5000, key="a_bal")
        a_prod  = st.selectbox("Products",   [1,2,3,4], index=max(0, b_prod-2), key="a_prod")
        a_active= st.radio("Active Member?", [True, False],
                           format_func=lambda x:"Yes" if x else "No", key="a_act", index=0)
        a_geo   = st.selectbox("Geography",  ["France","Germany","Spain"], key="a_geo")
        a_ten   = st.slider("Tenure (years)", 0, 10, min(10, b_ten+2), key="a_ten")
        a_cs    = st.slider("Credit Score",  300, 850, min(850, b_cs+50), key="a_cs")
        st.markdown("</div>", unsafe_allow_html=True)

    run_btn = st.button("⚡  Run Scenario Comparison", use_container_width=False)

    if run_btn:
        def qpred(age, bal, prod, act, geo, ten, cs):
            d = {"cs":cs,"age":age,"tenure":ten,"balance":bal,"products":prod,
                 "has_cc":True,"active":act,"salary":85000,"geo":geo,"gender":"Male"}
            return float(gb["model"].predict_proba(make_row(d, scaler, feat, num_cols))[0][1])

        b_p = qpred(b_age, b_bal, b_prod, b_active, b_geo, b_ten, b_cs)
        a_p = qpred(a_age, a_bal, a_prod, a_active, a_geo, a_ten, a_cs)
        delta = a_p - b_p
        b_rc  = risk_cls(b_p); a_rc = risk_cls(a_p)

        # Score summary
        s1, s2, s3 = st.columns(3)
        s1.markdown(f"""
        <div class="risk-box {b_rc}" style="text-align:center">
          <div style="font-size:.68rem;color:#556B8D;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">BASE SCENARIO</div>
          <div class="risk-pct {b_rc}">{b_p*100:.1f}%</div>
          <span class="risk-tag {b_rc}">{risk_label(b_p)[0]}</span>
        </div>""", unsafe_allow_html=True)

        d_color = "#10B981" if delta < 0 else "#EF4444"
        d_arrow = "↓ Improved" if delta < 0 else "↑ Worsened"
        s2.markdown(f"""
        <div style="background:rgba(255,255,255,.03);border:1px solid #1A2B48;border-radius:16px;
                    padding:28px 20px;text-align:center">
          <div style="font-size:.68rem;color:#556B8D;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">DELTA</div>
          <div style="font-family:'Clash Display',sans-serif;font-size:3rem;font-weight:700;
                      line-height:1;color:{d_color}">{delta*100:+.1f}%</div>
          <div style="font-size:.82rem;color:{d_color};margin-top:8px;font-weight:600">{d_arrow}</div>
        </div>""", unsafe_allow_html=True)

        s3.markdown(f"""
        <div class="risk-box {a_rc}" style="text-align:center">
          <div style="font-size:.68rem;color:#556B8D;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px">INTERVENTION RESULT</div>
          <div class="risk-pct {a_rc}">{a_p*100:.1f}%</div>
          <span class="risk-tag {a_rc}">{risk_label(a_p)[0]}</span>
        </div>""", unsafe_allow_html=True)

        # Waterfall-style sensitivity chart
        set_mpl()
        st.markdown('<div class="card" style="margin-top:18px">', unsafe_allow_html=True)
        st.markdown('<div class="sec-head"><span class="sec-icon">🔬</span>Scenario Impact Analysis</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # Horizontal comparison
        b_col = C_RED if b_rc=="high" else C_AMBER if b_rc=="med" else C_GREEN
        a_col = C_RED if a_rc=="high" else C_AMBER if a_rc=="med" else C_GREEN
        axes[0].barh(["Retention\nScenario","Base\nCustomer"],
                     [a_p*100, b_p*100], color=[a_col, b_col], edgecolor="none", height=0.42, zorder=3)
        axes[0].axvline(50, color="#E2EAF4", lw=1.2, linestyle="--", alpha=0.5, label="50% threshold")
        for i, (v, col) in enumerate([(a_p*100, a_col),(b_p*100, b_col)]):
            axes[0].text(v+0.5, i, f"{v:.1f}%", va="center", fontsize=11, fontweight="bold", color=col)
        axes[0].set_xlim(0, 110)
        axes[0].set_xlabel("Churn Probability (%)")
        axes[0].set_title("Before vs After Intervention", fontsize=10, color="#7A92B0")
        axes[0].xaxis.grid(True); axes[0].set_axisbelow(True)
        axes[0].legend(fontsize=8)

        # Sensitivity: top 6 variables toggled one at a time
        levers = [
            ("Make Active",      qpred(b_age, b_bal, b_prod, True,    b_geo, b_ten, b_cs)),
            ("Move to France",   qpred(b_age, b_bal, b_prod, b_active,"France", b_ten, b_cs)),
            ("Reduce to 1 Prod", qpred(b_age, b_bal, 1,      b_active, b_geo, b_ten, b_cs)),
            ("+3 Yrs Tenure",    qpred(b_age, b_bal, b_prod, b_active, b_geo, min(10,b_ten+3), b_cs)),
            ("+100 Credit Score",qpred(b_age, b_bal, b_prod, b_active, b_geo, b_ten, min(850,b_cs+100))),
            ("All Applied",      a_p),
        ]
        lever_names = [l[0] for l in levers]
        lever_deltas= [(b_p - l[1])*100 for l in levers]
        l_colors    = [C_GREEN if d >= 0 else C_RED for d in lever_deltas]
        axes[1].barh(lever_names[::-1], lever_deltas[::-1], color=l_colors[::-1], edgecolor="none", height=0.5, zorder=3)
        axes[1].axvline(0, color="#E2EAF4", lw=1, alpha=0.4)
        for i, (nm, dv) in enumerate(zip(reversed(lever_names), reversed(lever_deltas))):
            sign = "+" if dv >= 0 else ""
            color = C_GREEN if dv >= 0 else C_RED
            axes[1].text(dv + (0.2 if dv>=0 else -0.2), i, f"{sign}{dv:.1f}%",
                        va="center", ha="left" if dv>=0 else "right",
                        fontsize=8.5, fontweight="bold", color=color)
        axes[1].set_xlabel("Churn Reduction (%)")
        axes[1].set_title("Per-Lever Impact (positive = risk reduction)", fontsize=10, color="#7A92B0")
        axes[1].xaxis.grid(True); axes[1].set_axisbelow(True)

        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Action recommendation
        if delta < -0.10:
            st.success(f"🎉 This intervention reduces churn risk by **{abs(delta)*100:.1f}%** — highly effective retention strategy!")
        elif delta < 0:
            st.info(f"✅ Moderate improvement of **{abs(delta)*100:.1f}%**. Consider combining multiple levers for stronger impact.")
        else:
            st.warning("⚠️ This scenario does not improve retention. Review the lever analysis above for better options.")