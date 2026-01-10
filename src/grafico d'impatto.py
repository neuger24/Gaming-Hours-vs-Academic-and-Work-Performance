import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


df = pd.read_csv('Gaming_Hours_vs_Performance.csv', sep=';')
df = df.drop(columns=['User_ID', 'Weekly_Gaming_Hours'])


numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)


X = df.drop('Performance_Impact', axis=1)
y = LabelEncoder().fit_transform(df['Performance_Impact'])
X = pd.get_dummies(X, columns=['Gender', 'Occupation', 'Game_Type', 'Primary_Gaming_Time'], drop_first=True, dtype=int)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)


dt = DecisionTreeClassifier(random_state=42).fit(X_resampled, y_resampled)
rf = RandomForestClassifier(random_state=42).fit(X_resampled, y_resampled)


models = [(dt, 'Decision Tree', 'viridis'), (rf, 'Random Forest', 'magma')]

for model, name, palette in models:

    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)


    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi.values, y=fi.index, palette=palette)
    plt.title(f'Feature Importance: {name} (Dataset Bilanciato)')
    plt.xlabel('Importance Score')
    plt.ylabel('Variabili')
    plt.tight_layout()
    plt.show()