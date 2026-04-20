import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('charts', exist_ok=True)

# ── Load & Clean ─────────────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['ChurnBinary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# ── Colors ───────────────────────────────────────────────────────
BG     = '#0d0d0d'
PANEL  = '#1a1a1a'
RED    = '#ff4d4d'
GREEN  = '#2ecc71'
ORANGE = '#f39c12'
TEAL   = '#1abc9c'
GRAY   = '#888888'
WHITE  = '#ffffff'

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

# ════════════════════════════════════════════════════════════════
# PAGE 1
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor=BG)
fig.text(0.5, 0.97,
         'TELCO CUSTOMER CHURN  |  Retention & Churn Analysis Dashboard  |  Page 1 of 2',
         ha='center', va='top', color=WHITE, fontsize=14, fontweight='bold')

kpis = [
    ('26.5%', 'Overall Churn Rate', RED),
    ('1,869',  'Churned Customers', RED),
    ('42%',   'M2M Contract Churn', ORANGE),
    ('2.8%',  'Two-Year Churn',     GREEN),
    ('18 mo', 'Avg Tenure (Churned)',  ORANGE),
    ('38 mo', 'Avg Tenure (Retained)', GREEN),
]
for i, (val, label, col) in enumerate(kpis):
    x = 0.08 + i * 0.155
    fig.text(x, 0.91, val,   ha='center', color=col,  fontsize=18, fontweight='bold')
    fig.text(x, 0.875, label, ha='center', color=GRAY, fontsize=7.5)

fig.add_artist(plt.Line2D([0.03, 0.97], [0.865, 0.865],
               color='#333333', linewidth=0.8, transform=fig.transFigure))

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.05, right=0.97,
                       top=0.85, bottom=0.06,
                       hspace=0.45, wspace=0.35)

# Chart 1: Churn by Contract
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1)
contracts = ['Month-to-month', 'One year', 'Two year']
churn_c   = df.groupby('Contract')['ChurnBinary'].mean().reindex(contracts) * 100
retain_c  = 100 - churn_c
for i, (c, r, col) in enumerate(zip(churn_c, retain_c, [RED, ORANGE, GREEN])):
    ax1.barh(i, c, color=col, height=0.5)
    ax1.barh(i, r, left=c, color='#2d6a4f', height=0.5)
    ax1.text(c/2, i, f'{c:.0f}%', ha='center', va='center',
             color=WHITE, fontsize=9, fontweight='bold')
ax1.set_yticks(range(3))
ax1.set_yticklabels(contracts, color=WHITE, fontsize=8)
ax1.set_xlim(0, 100)
ax1.set_xlabel('% customers', color=WHITE, fontsize=8)
ax1.set_title('Churn by Contract Type', color=WHITE, fontsize=10, fontweight='bold')

# Chart 2: Churn by Internet Service
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2)
internet_order = ['Fiber optic', 'DSL', 'No']
churn_i = df.groupby('InternetService')['ChurnBinary'].mean().reindex(internet_order) * 100
bars = ax2.bar(internet_order, churn_i.values, color=[RED, ORANGE, GREEN], width=0.5)
for bar, val in zip(bars, churn_i.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax2.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax2.set_ylim(0, 55)
ax2.set_title('Churn by Internet Service', color=WHITE, fontsize=10, fontweight='bold')

# Chart 3: Churn by Payment Method
ax3 = fig.add_subplot(gs[0, 2])
style_ax(ax3)
pay_labels = ['Electronic check', 'Mailed check', 'Bank transfer\n(auto)', 'Credit card\n(auto)']
pay_keys   = ['Electronic check', 'Mailed check',
              'Bank transfer (automatic)', 'Credit card (automatic)']
churn_p = df.groupby('PaymentMethod')['ChurnBinary'].mean().reindex(pay_keys) * 100
bars = ax3.bar(pay_labels, churn_p.values,
               color=[RED, ORANGE, GREEN, TEAL], width=0.5)
for bar, val in zip(bars, churn_p.values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax3.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax3.set_ylim(0, 55)
ax3.set_title('Churn by Payment Method', color=WHITE, fontsize=10, fontweight='bold')
ax3.tick_params(axis='x', labelsize=7)

# Chart 4: Churn by Tenure
ax4 = fig.add_subplot(gs[1, 0])
style_ax(ax4)
bins   = [0, 6, 12, 24, 36, 48, 72]
labels = ['0-6m','6-12m','1-2yr','2-3yr','3-4yr','4yr+']
df['TG'] = pd.cut(df['tenure'], bins=bins, labels=labels)
churn_t = df.groupby('TG', observed=True)['ChurnBinary'].mean() * 100
ax4.plot(labels, churn_t.values, color=RED, linewidth=2, zorder=2)
ax4.fill_between(labels, churn_t.values, alpha=0.15, color=RED)
for x, y in zip(labels, churn_t.values):
    col = RED if y > 26.5 else GREEN
    ax4.scatter(x, y, color=col, s=60, zorder=3)
    ax4.text(x, y + 1.5, f'{y:.0f}%', ha='center', color=WHITE,
             fontsize=8, fontweight='bold')
ax4.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax4.set_ylim(0, 70)
ax4.set_title('Churn by Tenure (customer lifecycle)', color=WHITE,
              fontsize=10, fontweight='bold')

# Chart 5: Churn by Monthly Charge Band
ax5 = fig.add_subplot(gs[1, 1])
style_ax(ax5)
charge_bins   = [0, 35, 55, 75, 95, 200]
charge_labels = ['<$35','$35-55','$55-75','$75-95','$95+']
df['CB'] = pd.cut(df['MonthlyCharges'], bins=charge_bins, labels=charge_labels)
churn_cb = df.groupby('CB', observed=True)['ChurnBinary'].mean() * 100
cb_colors = [GREEN if v < 26.5 else ORANGE if v < 33 else RED
             for v in churn_cb.values]
bars = ax5.bar(charge_labels, churn_cb.values, color=cb_colors, width=0.5)
for bar, val in zip(bars, churn_cb.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f}%', ha='center', color=WHITE, fontsize=9, fontweight='bold')
ax5.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax5.set_xlabel('Monthly charges', color=WHITE, fontsize=8)
ax5.set_ylim(0, 55)
ax5.set_title('Churn by Monthly Charge Band', color=WHITE,
              fontsize=10, fontweight='bold')

# Chart 6: Add-on Services
ax6 = fig.add_subplot(gs[1, 2])
style_ax(ax6)
addons       = ['StreamingTV','DeviceProtection','OnlineBackup','TechSupport','OnlineSecurity']
addon_labels = ['Streaming TV','DeviceProtection','OnlineBackup','TechSupport','OnlineSecurity']
addon_churn  = [df[df[a]=='Yes']['ChurnBinary'].mean() * 100 for a in addons]
avg_line     = df['ChurnBinary'].mean() * 100
addon_colors = [RED if v > avg_line else GREEN for v in addon_churn]
bars = ax6.barh(addon_labels, addon_churn, color=addon_colors, height=0.5)
for bar, val in zip(bars, addon_churn):
    ax6.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}%', va='center', color=WHITE, fontsize=9, fontweight='bold')
ax6.axvline(x=avg_line, color=WHITE, linestyle='--', linewidth=1, alpha=0.6)
ax6.text(avg_line + 0.3, -0.6, f'Avg {avg_line:.1f}%', color=WHITE, fontsize=7)
ax6.set_xlim(0, 40)
ax6.set_xlabel('Churn rate (%)', color=WHITE, fontsize=8)
ax6.set_title('Churn: Customers WITH Add-on Services', color=WHITE,
              fontsize=10, fontweight='bold')

plt.savefig('charts/churn_dashboard_page1.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Page 1 saved!")

# ════════════════════════════════════════════════════════════════
# PAGE 2
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12), facecolor=BG)
fig.text(0.5, 0.97,
         'TELCO CUSTOMER CHURN  |  Cohort Analysis & Recommendations  |  Page 2 of 2',
         ha='center', va='top', color=WHITE, fontsize=14, fontweight='bold')

gs = gridspec.GridSpec(2, 3, figure=fig,
                       left=0.05, right=0.97,
                       top=0.91, bottom=0.06,
                       hspace=0.45, wspace=0.38)

# Chart 1: Cohort Heatmap
ax1 = fig.add_subplot(gs[0, 0:2])
style_ax(ax1)
contracts = ['Month-to-month', 'One year', 'Two year']
internets = ['DSL', 'Fiber optic', 'No']
heatmap_data = pd.pivot_table(df, values='ChurnBinary',
                               index='Contract', columns='InternetService',
                               aggfunc='mean') * 100
heatmap_data = heatmap_data.reindex(index=contracts, columns=internets)
im = ax1.imshow(heatmap_data.values, cmap='RdYlGn_r',
                aspect='auto', vmin=0, vmax=60)
ax1.set_xticks(range(3))
ax1.set_yticks(range(3))
ax1.set_xticklabels(internets, color=WHITE, fontsize=9)
ax1.set_yticklabels(contracts, color=WHITE, fontsize=9)
ax1.set_xlabel('Internet Service', color=WHITE, fontsize=9)
ax1.set_ylabel('Contract Type', color=WHITE, fontsize=9)
ax1.set_title('Retention Cohort Heatmap  (Contract x Internet Service)',
              color=WHITE, fontsize=11, fontweight='bold')
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f'{heatmap_data.values[i,j]:.1f}',
                 ha='center', va='center', color=WHITE,
                 fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
cbar.ax.tick_params(colors=WHITE, labelsize=8)
cbar.set_label('Churn %', color=WHITE, fontsize=8)

# Chart 2: Tenure Distribution
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2)
churned  = df[df['Churn']=='Yes']['tenure']
retained = df[df['Churn']=='No']['tenure']
ax2.hist(retained, bins=35, alpha=0.6, color=GREEN, label='Retained', density=True)
ax2.hist(churned,  bins=35, alpha=0.6, color=RED,   label='Churned',  density=True)
ax2.axvline(churned.mean(),  color=RED,   linestyle='--', linewidth=1.5,
            label=f'Churn avg {churned.mean():.0f}m')
ax2.axvline(retained.mean(), color=GREEN, linestyle='--', linewidth=1.5,
            label=f'Retain avg {retained.mean():.0f}m')
ax2.set_xlabel('Tenure (months)', color=WHITE, fontsize=8)
ax2.set_ylabel('Density', color=WHITE, fontsize=8)
ax2.set_title('Tenure Distribution: Churned vs Retained',
              color=WHITE, fontsize=10, fontweight='bold')
ax2.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.8)

# Chart 3: Senior vs Non-Senior
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3)
senior_churn    = df[df['SeniorCitizen']==1]['ChurnBinary'].mean() * 100
nonsenior_churn = df[df['SeniorCitizen']==0]['ChurnBinary'].mean() * 100
bars = ax3.bar(['Senior','Non-Senior'], [senior_churn, nonsenior_churn],
               color=[RED, '#ff8080'], width=0.4)
for bar, val in zip(bars, [senior_churn, nonsenior_churn]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=WHITE,
             fontsize=11, fontweight='bold')
ax3.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax3.set_ylim(0, 55)
ax3.set_title('Senior vs Non-Senior Churn', color=WHITE,
              fontsize=10, fontweight='bold')

# Chart 4: Paperless Billing
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4)
paper_yes = df[df['PaperlessBilling']=='Yes']['ChurnBinary'].mean() * 100
paper_no  = df[df['PaperlessBilling']=='No']['ChurnBinary'].mean() * 100
bars = ax4.bar(['Yes','No'], [paper_yes, paper_no],
               color=[RED, GREEN], width=0.4)
for bar, val in zip(bars, [paper_yes, paper_no]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=WHITE,
             fontsize=11, fontweight='bold')
ax4.set_ylabel('Churn rate (%)', color=WHITE, fontsize=8)
ax4.set_ylim(0, 45)
ax4.set_xlabel('Paperless Billing', color=WHITE, fontsize=8)
ax4.set_title('Paperless Billing vs Churn', color=WHITE,
              fontsize=10, fontweight='bold')

# Recommendations Box
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor(PANEL)
for spine in ax5.spines.values():
    spine.set_edgecolor('#333333')
ax5.set_xticks([])
ax5.set_yticks([])
ax5.text(0.05, 0.93, 'Strategic Recommendations', color=WHITE,
         fontsize=11, fontweight='bold', transform=ax5.transAxes)
recs = [
    (RED,      '1', 'Migrate M2M -> annual contracts'),
    (ORANGE,   '2', '90-day new customer onboarding program'),
    (ORANGE,   '3', 'Convert e-check payers to auto-pay'),
    (GREEN,    '4', 'Bundle add-ons with fiber subscriptions'),
    (GREEN,    '5', 'Investigate fiber optic satisfaction gap'),
    ('#3498db','6', 'Senior-specific support & loyalty pricing'),
]
for i, (col, num, text) in enumerate(recs):
    y = 0.78 - i * 0.13
    circle = plt.Circle((0.07, y + 0.02), 0.045, color=col,
                         transform=ax5.transAxes, clip_on=False)
    ax5.add_patch(circle)
    ax5.text(0.07, y + 0.02, num, ha='center', va='center',
             color=WHITE, fontsize=9, fontweight='bold',
             transform=ax5.transAxes)
    ax5.text(0.16, y + 0.02, text, va='center', color=WHITE,
             fontsize=8.5, transform=ax5.transAxes)

plt.savefig('charts/churn_dashboard_page2.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.close()
print("Page 2 saved!")
print("\nBoth dashboard pages saved in the 'charts' folder!")