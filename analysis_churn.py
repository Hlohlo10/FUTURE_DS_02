import pandas as pd
import matplotlib.pyplot as plt
import os
 
# Create charts folder
os.makedirs('charts', exist_ok=True)
 
# Load clean data
df = pd.read_csv('telco_churn_clean.csv')
print("Data loaded!")
print(f"Total Customers: {len(df):,}")
print(f"Churned Customers: {df['ChurnBinary'].sum():,}")
print(f"Overall Churn Rate: {df['ChurnBinary'].mean() * 100:.2f}%")
 
# CHART 1: Churn Rate by Contract Type
churn_contract = df.groupby('Contract')['ChurnBinary'].mean() * 100
 
plt.figure(figsize=(10, 5))
bars = plt.bar(churn_contract.index, churn_contract.values,
               color=['#e74c3c', '#3498db', '#2ecc71'])
plt.title('Churn Rate by Contract Type', fontsize=16, fontweight='bold')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_contract.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/churn_by_contract.png')
plt.close()
print("Chart 1 saved: Churn by Contract")
 
#CHART 2: Churn Rate by Payment Method
churn_payment = df.groupby('PaymentMethod')['ChurnBinary'].mean() * 100
 
plt.figure(figsize=(10, 5))
bars = plt.bar(churn_payment.index, churn_payment.values, color='steelblue')
plt.title('Churn Rate by Payment Method', fontsize=16, fontweight='bold')
plt.xlabel('Payment Method')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=15)
for bar, val in zip(bars, churn_payment.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/churn_by_payment.png')
plt.close()
print("Chart 2 saved: Churn by Payment Method")
 
#CHART 3: Churn Rate by Tenure Group
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                            labels=['0-12 months', '13-24 months',
                                    '25-48 months', '49-72 months'])
churn_tenure = df.groupby('TenureGroup', observed=True)['ChurnBinary'].mean() * 100
 
plt.figure(figsize=(10, 5))
bars = plt.bar(churn_tenure.index.astype(str), churn_tenure.values,
               color=['#e74c3c', '#e67e22', '#3498db', '#2ecc71'])
plt.title('Churn Rate by Customer Tenure', fontsize=16, fontweight='bold')
plt.xlabel('Tenure Group')
plt.ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_tenure.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/churn_by_tenure.png')
plt.close()
print("Chart 3 saved: Churn by Tenure")
 
#CHART 4: Churn Rate by Internet Service
churn_internet = df.groupby('InternetService')['ChurnBinary'].mean() * 100
 
plt.figure(figsize=(10, 5))
bars = plt.bar(churn_internet.index, churn_internet.values,
               color=['#9b59b6', '#e74c3c', '#2ecc71'])
plt.title('Churn Rate by Internet Service Type', fontsize=16, fontweight='bold')
plt.xlabel('Internet Service')
plt.ylabel('Churn Rate (%)')
for bar, val in zip(bars, churn_internet.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('charts/churn_by_internet.png')
plt.close()
print("Chart 4 saved: Churn by Internet Service")
 
#CHART 5: Monthly Charges - Churned vs Retained
churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
retained = df[df['Churn'] == 'No']['MonthlyCharges']
 
plt.figure(figsize=(10, 5))
plt.hist(retained, bins=30, alpha=0.6, label='Retained', color='#2ecc71')
plt.hist(churned, bins=30, alpha=0.6, label='Churned', color='#e74c3c')
plt.title('Monthly Charges: Churned vs Retained Customers',
          fontsize=16, fontweight='bold')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.savefig('charts/monthly_charges_churn.png')
plt.close()
print("Chart 5 saved: Monthly Charges Distribution")
 
print("\nAll charts saved in the 'charts' folder!")