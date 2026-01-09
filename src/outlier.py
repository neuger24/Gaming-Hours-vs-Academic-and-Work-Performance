import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Caricamento del dataset
# Assicurati di aver caricato il file 'Gaming_Hours_vs_Performance.csv' nella cartella di Colab
df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

# 2. Selezione delle sole colonne numeriche per l'analisi
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

print("--- Analisi Statistica degli Outlier (Metodo IQR) ---")

# 3. Ciclo per calcolare gli outlier per ogni colonna
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identificazione dei record fuori dai limiti
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"Colonna '{col}': {len(outliers)} outlier rilevati.")

# 4. Visualizzazione Grafica con Boxplot
plt.figure(figsize=(16, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(data=df, y=col, color='skyblue')
    plt.title(f'Boxplot di {col}')

plt.tight_layout()
plt.show()