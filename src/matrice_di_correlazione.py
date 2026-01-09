import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')


numeric_df = df.select_dtypes(include=['float64', 'int64'])


corr_matrix = numeric_df.corr()


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice di Correlazione delle Variabili Numeriche')
plt.show()


print(corr_matrix)