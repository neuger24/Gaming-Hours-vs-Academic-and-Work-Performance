import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Caricamento del dataset
df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

# 2. Selezione delle sole variabili numeriche
# La correlazione di Pearson si applica a dati quantitativi
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# 3. Calcolo della matrice di correlazione
corr_matrix = numeric_df.corr()

# 4. Visualizzazione con Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice di Correlazione delle Variabili Numeriche')
plt.show()

# 5. Visualizzazione dei valori testuali
print(corr_matrix)