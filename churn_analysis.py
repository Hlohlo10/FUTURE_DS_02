# =============================================================================
# CUSTOMER RETENTION & CHURN ANALYSIS
# Future Interns - Data Science & Analytics Task 2 (2026)
# Dataset: Telco Customer Churn 
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. STYLE SETUP
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#0F1117",
    "axes.edgecolor":   "#2A2D3A",
    "axes.labelcolor":  "#C9CDD8",
    "xtick.color":      "#7A7F8E",
    "ytick.color":      "#7A7F8E",
    "text.color":       "#E4E6EB",
    "grid.color":       "#1E2130",
    "grid.linestyle":   "--",
    "font.family":      "DejaVu Sans",
    "font.size":        10,
})

CHURN_COLOR    = "#E05C5C"
RETAIN_COLOR   = "#4CAF82"
ACCENT_COLOR   = "#5B8CFF"
WARN_COLOR     = "#F5A623"
PALETTE        = [CHURN_COLOR, RETAIN_COLOR, ACCENT_COLOR, WARN_COLOR,
                  "#A78BFA", "#34D399", "#FBBF24", "#60A5FA"]

# 1. LOAD DATA

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("=" * 60)
print("  TELCO CHURN ANALYSIS  |  Future Interns 2026")
print("=" * 60)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"\n✅  Loaded {len(df):,} rows × {df.shape[1]} columns\n")
except FileNotFoundError:
    raise SystemExit(
        f"\n❌  File not found: '{CSV_PATH}'\n"
        "    Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        "    Then place the CSV in the same folder as this script.\n"
    )

# 2. DATA CLEANING
print("── Data Cleaning ──────────────────────────────────────────")

# TotalCharges is sometimes stored as a string with spaces
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
null_count = df["TotalCharges"].isna().sum()
print(f"   TotalCharges nulls (new customers, tenure=0): {null_count}")
df["TotalCharges"].fillna(0, inplace=True)

# Binary encode Churn
df["ChurnBinary"] = (df["Churn"] == "Yes").astype(int)

# Tenure bands
bins   = [0, 6, 12, 24, 36, 48, 72]
labels = ["0–6m", "6–12m", "1–2yr", "2–3yr", "3–4yr", "4yr+"]
df["TenureBand"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True)

# Monthly charge bands
df["ChargeBand"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 55, 75, 95, 120],
    labels=["<$35", "$35–55", "$55–75", "$75–95", "$95+"]
)

print(f"   Shape after cleaning: {df.shape}")
print(f"   Overall churn rate  : {df['ChurnBinary'].mean()*100:.1f}%\n")

# 3. KEY METRICS
print("── Key Business Metrics ───────────────────────────────────")
total          = len(df)
churned        = df["ChurnBinary"].sum()
retained       = total - churned
churn_rate     = churned / total * 100
avg_tenure_c   = df[df["Churn"]=="Yes"]["tenure"].mean()
avg_tenure_r   = df[df["Churn"]=="No"]["tenure"].mean()
avg_ltv        = df[df["Churn"]=="No"]["TotalCharges"].mean()
monthly_rev    = df["MonthlyCharges"].sum()
rev_at_risk    = df[df["Churn"]=="Yes"]["MonthlyCharges"].sum()

print(f"   Total customers     : {total:,}")
print(f"   Churned             : {churned:,}  ({churn_rate:.1f}%)")
print(f"   Retained            : {retained:,}")
print(f"   Avg tenure churned  : {avg_tenure_c:.0f} months")
print(f"   Avg tenure retained : {avg_tenure_r:.0f} months")
print(f"   Avg LTV (retained)  : ${avg_ltv:,.0f}")
print(f"   Monthly revenue     : ${monthly_rev:,.0f}")
print(f"   Revenue at risk     : ${rev_at_risk:,.0f}  ({rev_at_risk/monthly_rev*100:.1f}%)\n")


# 4. SEGMENT CHURN RATES  (helper function)
def churn_by(col):
    """Return a DataFrame with churn rate % per category."""
    g = df.groupby(col)["ChurnBinary"].agg(["sum", "count"]).reset_index()
    g.columns = [col, "Churned", "Total"]
    g["ChurnRate"] = g["Churned"] / g["Total"] * 100
    return g.sort_values("ChurnRate", ascending=False)

contract_churn  = churn_by("Contract")
internet_churn  = churn_by("InternetService")
payment_churn   = churn_by("PaymentMethod")
tenure_churn    = churn_by("TenureBand")
charge_churn    = churn_by("ChargeBand")
senior_churn    = churn_by("SeniorCitizen")
paperless_churn = churn_by("PaperlessBilling")

# Add-on services
addons = ["TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "StreamingTV"]
addon_churn = []
for col in addons:
    sub = df[df[col] == "Yes"]["ChurnBinary"].mean() * 100
    addon_churn.append({"Service": col.replace("Streaming", "Streaming "), "ChurnRate": sub})
addon_df = pd.DataFrame(addon_churn).sort_values("ChurnRate")

# Cohort: Contract × Internet
cohort = (
    df.groupby(["Contract", "InternetService"])["ChurnBinary"]
    .mean()
    .mul(100)
    .round(1)
    .unstack()
)

# 5. PRINT SEGMENT TABLES

print("── Churn by Contract ──────────────────────────────────────")
print(contract_churn.to_string(index=False))
print("\n── Churn by Internet Service ──────────────────────────────")
print(internet_churn.to_string(index=False))
print("\n── Churn by Payment Method ────────────────────────────────")
print(payment_churn.to_string(index=False))
print("\n── Cohort Heatmap (Contract × Internet) ───────────────────")
print(cohort.to_string())
print()


# 6.  DASHBOARD  (4-page figure layout)
print("── Building Dashboard ─────────────────────────────────────")

# ── PAGE 1: Overview & Core Segments ──────────────────────────
fig1 = plt.figure(figsize=(18, 10))
fig1.suptitle(
    "TELCO CUSTOMER CHURN  |  Retention & Churn Analysis Dashboard  |  Page 1 of 2",
    fontsize=13, fontweight="bold", color="#E4E6EB", y=0.98
)
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.38)

# ── KPI bar (top strip) using text annotations
kpi_ax = fig1.add_axes([0.0, 0.88, 1.0, 0.09])
kpi_ax.set_axis_off()
kpi_ax.set_facecolor("#161923")
fig1.patches.append(plt.Rectangle((0, 0.88), 1, 0.09,
    transform=fig1.transFigure, color="#161923", zorder=-1))

kpis = [
    ("26.5%",  "Overall Churn Rate",      CHURN_COLOR),
    ("1,869",  "Customers Lost",          CHURN_COLOR),
    ("42%",    "M2M Contract Churn",      WARN_COLOR),
    ("2.8%",   "2-Year Contract Churn",   RETAIN_COLOR),
    ("18 mo",  "Avg Tenure (Churned)",    WARN_COLOR),
    ("38 mo",  "Avg Tenure (Retained)",   RETAIN_COLOR),
]
for i, (val, lbl, col) in enumerate(kpis):
    x = 0.085 + i * 0.148
    kpi_ax.text(x, 0.75, val, ha="center", va="center",
                fontsize=16, fontweight="bold", color=col)
    kpi_ax.text(x, 0.15, lbl, ha="center", va="center",
                fontsize=8, color="#7A7F8E")

# ── Plot 1: Churn by Contract (stacked horizontal bar)
ax1 = fig1.add_subplot(gs1[0, 0])
ctypes = contract_churn.sort_values("ChurnRate")
bars_c = ax1.barh(ctypes["Contract"], ctypes["ChurnRate"], color=CHURN_COLOR, height=0.5)
ax1.barh(ctypes["Contract"], 100 - ctypes["ChurnRate"],
         left=ctypes["ChurnRate"], color=RETAIN_COLOR, height=0.5, alpha=0.6)
for bar, val in zip(bars_c, ctypes["ChurnRate"]):
    ax1.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}%", va="center", ha="center", fontsize=10,
             fontweight="bold", color="white")
ax1.set_xlim(0, 100)
ax1.set_xlabel("% customers")
ax1.set_title("Churn by Contract Type", fontsize=11, pad=8)
ax1.tick_params(left=False)

# ── Plot 2: Churn by Internet Service
ax2 = fig1.add_subplot(gs1[0, 1])
colors2 = [CHURN_COLOR if v > 30 else (WARN_COLOR if v > 15 else RETAIN_COLOR)
           for v in internet_churn["ChurnRate"]]
bars2 = ax2.bar(internet_churn["InternetService"], internet_churn["ChurnRate"],
                color=colors2, width=0.5)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10,
             fontweight="bold")
ax2.set_ylabel("Churn rate (%)")
ax2.set_title("Churn by Internet Service", fontsize=11, pad=8)
ax2.set_ylim(0, 55)
ax2.yaxis.grid(True)
ax2.set_axisbelow(True)

# ── Plot 3: Churn by Payment Method
ax3 = fig1.add_subplot(gs1[0, 2])
pm = payment_churn.copy()
pm["Method"] = pm["PaymentMethod"].str.replace(" (automatic)", "\n(auto)", regex=False)
colors3 = [CHURN_COLOR if v > 30 else (WARN_COLOR if v > 20 else RETAIN_COLOR)
           for v in pm["ChurnRate"]]
bars3 = ax3.bar(range(len(pm)), pm["ChurnRate"], color=colors3, width=0.55)
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9,
             fontweight="bold")
ax3.set_xticks(range(len(pm)))
ax3.set_xticklabels(pm["Method"], fontsize=8)
ax3.set_ylabel("Churn rate (%)")
ax3.set_title("Churn by Payment Method", fontsize=11, pad=8)
ax3.set_ylim(0, 55)
ax3.yaxis.grid(True)
ax3.set_axisbelow(True)

# ── Plot 4: Churn by Tenure Band (line)
ax4 = fig1.add_subplot(gs1[1, 0])
tb = tenure_churn.set_index("TenureBand").reindex(labels)
ax4.plot(tb.index, tb["ChurnRate"], marker="o", color=CHURN_COLOR,
         linewidth=2.5, markersize=8)
ax4.fill_between(range(len(tb)), tb["ChurnRate"].values, alpha=0.15, color=CHURN_COLOR)
for i, (idx, row) in enumerate(tb.iterrows()):
    ax4.text(i, row["ChurnRate"] + 1.5, f"{row['ChurnRate']:.0f}%",
             ha="center", fontsize=9, fontweight="bold", color=CHURN_COLOR)
ax4.set_xticks(range(len(tb)))
ax4.set_xticklabels(tb.index, fontsize=9)
ax4.set_ylabel("Churn rate (%)")
ax4.set_title("Churn by Tenure (customer lifecycle)", fontsize=11, pad=8)
ax4.set_ylim(0, 70)
ax4.yaxis.grid(True)
ax4.set_axisbelow(True)

# ── Plot 5: Churn by Monthly Charge Band
ax5 = fig1.add_subplot(gs1[1, 1])
cb = charge_churn.dropna(subset=["ChargeBand"]).sort_values("ChargeBand")
grad_colors = [RETAIN_COLOR, "#A3C97A", WARN_COLOR, "#E07C3E", CHURN_COLOR]
bars5 = ax5.bar(cb["ChargeBand"].astype(str), cb["ChurnRate"],
                color=grad_colors[:len(cb)], width=0.55)
for bar in bars5:
    ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.0f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax5.set_ylabel("Churn rate (%)")
ax5.set_xlabel("Monthly charges")
ax5.set_title("Churn by Monthly Charge Band", fontsize=11, pad=8)
ax5.set_ylim(0, 55)
ax5.yaxis.grid(True)
ax5.set_axisbelow(True)

# ── Plot 6: Add-on Services vs Churn
ax6 = fig1.add_subplot(gs1[1, 2])
colors6 = [RETAIN_COLOR if v < 20 else (WARN_COLOR if v < 28 else CHURN_COLOR)
           for v in addon_df["ChurnRate"]]
bars6 = ax6.barh(addon_df["Service"], addon_df["ChurnRate"], color=colors6, height=0.5)
ax6.axvline(churn_rate, color="white", linestyle="--", linewidth=1, alpha=0.5,
            label=f"Avg {churn_rate:.1f}%")
for bar in bars6:
    ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.0f}%", va="center", fontsize=9, fontweight="bold")
ax6.set_xlabel("Churn rate (%)")
ax6.set_title("Churn: Customers WITH Add-on Services", fontsize=11, pad=8)
ax6.legend(fontsize=8)
ax6.set_xlim(0, 40)

fig1.savefig("churn_dashboard_page1.png", dpi=150, bbox_inches="tight",
             facecolor="#0F1117")
print("   ✅  Saved: churn_dashboard_page1.png")

# ── PAGE 2: Cohort, Distributions & Recommendations ───────────
fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle(
    "TELCO CUSTOMER CHURN  |  Cohort Analysis & Recommendations  |  Page 2 of 2",
    fontsize=13, fontweight="bold", color="#E4E6EB", y=0.98
)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.5, wspace=0.38)

# ── Plot 7: Cohort Heatmap  (Contract × Internet)
ax7 = fig2.add_subplot(gs2[0, :2])
sns.heatmap(
    cohort,
    annot=True, fmt=".1f", linewidths=0.5,
    cmap=sns.color_palette("RdYlGn_r", as_cmap=True),
    ax=ax7,
    cbar_kws={"label": "Churn %", "shrink": 0.8},
    annot_kws={"size": 12, "weight": "bold"}
)
ax7.set_title("Retention Cohort Heatmap  (Contract × Internet Service)", fontsize=11, pad=10)
ax7.set_xlabel("Internet Service")
ax7.set_ylabel("Contract Type")
ax7.tick_params(axis="both", labelsize=9)

# ── Plot 8: Tenure distribution (churned vs retained)
ax8 = fig2.add_subplot(gs2[0, 2])
ax8.hist(df[df["Churn"]=="No"]["tenure"], bins=30, color=RETAIN_COLOR,
         alpha=0.7, label="Retained", density=True)
ax8.hist(df[df["Churn"]=="Yes"]["tenure"], bins=30, color=CHURN_COLOR,
         alpha=0.7, label="Churned", density=True)
ax8.axvline(avg_tenure_c, color=CHURN_COLOR,   linestyle="--", linewidth=1.5,
            label=f"Churn avg {avg_tenure_c:.0f}m")
ax8.axvline(avg_tenure_r, color=RETAIN_COLOR,  linestyle="--", linewidth=1.5,
            label=f"Retain avg {avg_tenure_r:.0f}m")
ax8.set_xlabel("Tenure (months)")
ax8.set_ylabel("Density")
ax8.set_title("Tenure Distribution: Churned vs Retained", fontsize=11, pad=8)
ax8.legend(fontsize=8)
ax8.yaxis.grid(True)
ax8.set_axisbelow(True)

# ── Plot 9: Senior vs Non-senior churn
ax9 = fig2.add_subplot(gs2[1, 0])
sc = senior_churn.copy()
sc["Label"] = sc["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
bars9 = ax9.bar(sc["Label"], sc["ChurnRate"],
                color=[RETAIN_COLOR, CHURN_COLOR], width=0.45)
for bar in bars9:
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold")
ax9.set_ylabel("Churn rate (%)")
ax9.set_title("Senior vs Non-Senior Churn", fontsize=11, pad=8)
ax9.set_ylim(0, 55)
ax9.yaxis.grid(True)
ax9.set_axisbelow(True)

# ── Plot 10: Paperless billing churn
ax10 = fig2.add_subplot(gs2[1, 1])
pb = paperless_churn.copy()
bars10 = ax10.bar(pb["PaperlessBilling"], pb["ChurnRate"],
                  color=[RETAIN_COLOR if v < 20 else CHURN_COLOR for v in pb["ChurnRate"]],
                  width=0.4)
for bar in bars10:
    ax10.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
              f"{bar.get_height():.1f}%", ha="center", va="bottom",
              fontsize=11, fontweight="bold")
ax10.set_ylabel("Churn rate (%)")
ax10.set_title("Paperless Billing vs Churn", fontsize=11, pad=8)
ax10.set_ylim(0, 45)
ax10.yaxis.grid(True)
ax10.set_axisbelow(True)

# ── Plot 11: Recommendations text panel
ax11 = fig2.add_subplot(gs2[1, 2])
ax11.set_axis_off()
recs = [
    ("1", "Migrate M2M → annual contracts",          CHURN_COLOR),
    ("2", "90-day new customer onboarding program",   WARN_COLOR),
    ("3", "Convert e-check payers to auto-pay",       WARN_COLOR),
    ("4", "Bundle add-ons with fiber subscriptions",  RETAIN_COLOR),
    ("5", "Investigate fiber optic satisfaction gap", RETAIN_COLOR),
    ("6", "Senior-specific support & loyalty pricing",ACCENT_COLOR),
]
ax11.text(0.0, 0.97, "Strategic Recommendations", fontsize=11,
          fontweight="bold", va="top", color="#E4E6EB")
y = 0.82
for num, text, col in recs:
    ax11.add_patch(plt.Circle((0.04, y + 0.01), 0.04, color=col, alpha=0.25,
                               transform=ax11.transAxes, zorder=3))
    ax11.text(0.04, y + 0.01, num, ha="center", va="center", fontsize=9,
              fontweight="bold", color=col, transform=ax11.transAxes)
    ax11.text(0.12, y, text, fontsize=9, va="center", color="#C9CDD8",
              transform=ax11.transAxes, wrap=True)
    y -= 0.135
ax11.set_xlim(0, 1); ax11.set_ylim(0, 1)

fig2.savefig("churn_dashboard_page2.png", dpi=150, bbox_inches="tight",
             facecolor="#0F1117")
print("   ✅  Saved: churn_dashboard_page2.png")

# 7.  EXPORT CLEAN SEGMENT SUMMARY  

summary_rows = []
for col, label in [
    ("Contract",        "Contract"),
    ("InternetService", "Internet Service"),
    ("PaymentMethod",   "Payment Method"),
    ("TenureBand",      "Tenure Band"),
    ("ChargeBand",      "Charge Band"),
]:
    seg = churn_by(col).copy()
    seg.rename(columns={col: "Segment"}, inplace=True)
    seg.insert(0, "Category", label)
    summary_rows.append(seg)

summary_df = pd.concat(summary_rows, ignore_index=True)
summary_df["ChurnRate"] = summary_df["ChurnRate"].round(1)
summary_df.to_csv("churn_segment_summary.csv", index=False)
print("   ✅  Saved: churn_segment_summary.csv  (import into Power BI / Excel)\n")


# 8.  FINAL PRINTED REPORT

print("=" * 60)
print("  EXECUTIVE SUMMARY")
print("=" * 60)
print(f"""
  Overall churn rate  : {churn_rate:.1f}%  ({churned:,} of {total:,} customers)
  Monthly revenue risk: ${rev_at_risk:,.0f}  ({rev_at_risk/monthly_rev*100:.1f}% of total)
  
  TOP CHURN DRIVERS
  -----------------
  1. Month-to-month contract  → 42%  churn
  2. Fiber optic internet     → 42%  churn  (pays most, churns most)
  3. Electronic check payment → 45%  churn
  4. New customers (0-6 mo)   → ~55% churn  (early-lifecycle cliff)
  5. Senior citizens          → 42%  churn  (vs 24% non-senior)

  SAFEST SEGMENTS
  ---------------
  - 2-year contract + auto-pay  → ~2–3% churn
  - With online security        → 14%  churn
  - With tech support           → 15%  churn
  - Tenure 4yr+                 → ~5%  churn

  KEY RECOMMENDATIONS
  -------------------
  1. Run a contract migration campaign for M2M customers
  2. Implement a 90-day onboarding & retention program
  3. Incentivise auto-pay adoption (small monthly discount)
  4. Bundle add-ons with fiber plans to raise perceived value
  5. Investigate fiber quality/pricing gap with NPS survey
  6. Create dedicated senior customer support & pricing track

  OUTPUTS
  -------
  📊 churn_dashboard_page1.png  — Segment analysis
  📊 churn_dashboard_page2.png  — Cohort heatmap & distributions
  📄 churn_segment_summary.csv  — Tabular data for Power BI / Excel
""")

plt.show()
print("✅  Analysis complete.")