import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')

print("Info Dataset:")

print(df.info())

missing_values = df.isnull().sum()

print("\nMissing Values per colonna:\n", missing_values[missing_values > 0])

plt.figure(figsize=(8, 5))

sns.countplot(x='Performance_Impact', data=df, order=['Negative', 'Neutral', 'Positive'])

plt.title('Distribuzione della Variabile Target (Performance_Impact)')

plt.show()
