import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Gaming_Hours_vs_Performance versione 1.1.csv', sep=';')

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='Performance_Impact', data=df, palette='magma')
plt.title('Distribuzione dell\'Impatto sulle Performance per Genere')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Performance_Impact', y='Daily_Gaming_Hours', data=df, palette='viridis')
plt.title('Relazione tra Ore di Gioco Giornaliere e Impatto')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Performance_Impact', y='Sleep_Hours', data=df, palette='plasma')
plt.title('Relazione tra Ore di Sonno e Impatto')
plt.show()