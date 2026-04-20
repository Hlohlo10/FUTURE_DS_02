import pandas as pd
 
# Loading the data
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Original shape:", df.shape)
 
# Fix TotalCharges column (has blank spaces, should be numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
 
# Drop rows where TotalCharges is blank
df = df.dropna(subset=['TotalCharges'])
print("After removing blank TotalCharges:", df.shape)
 
# Step 3: Convert Churn to binary (1 = churned, 0 = stayed)
df['ChurnBinary'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
 
# Step 4: Create tenure groups for cohort analysis
df['TenureGroup'] = pd.cut(df['tenure'],
                            bins=[0, 12, 24, 48, 72],
                            labels=['0-12 months', '13-24 months',
                                    '25-48 months', '49-72 months'])
 
# Save clean file
df.to_csv('telco_churn_clean.csv', index=False)
print("\nDone! Clean file saved.")
print(f"Overall churn rate: {df['ChurnBinary'].mean() * 100:.2f}%")
