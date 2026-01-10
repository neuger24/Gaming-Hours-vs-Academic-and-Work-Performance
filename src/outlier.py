import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

print("--- Analisi Statistica degli Outlier (Metodo IQR) ---")


for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"Colonna '{col}': {len(outliers)} outlier rilevati.")


plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, y=col, color='skyblue')
    plt.title(f'Boxplot di {col}')

plt.tight_layout()
plt.show()