import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"adult 3.csv")
df = df[['age', 'education', 'occupation', 'hours-per-week', 'income']]

# Clean
df.dropna(inplace=True)
df = df[df['education'].isin(['Bachelors', 'HS-grad', 'Masters'])]
df = df[df['occupation'].isin(['Tech-support', 'Sales', 'Exec-managerial'])]

# Encode
edu_map = {'HS-grad': 0, 'Bachelors': 1, 'Masters': 2}
occ_map = {'Tech-support': 0, 'Sales': 1, 'Exec-managerial': 2}
income_map = {'<=50K': 0, '>50K': 1}

df['education'] = df['education'].map(edu_map)
df['occupation'] = df['occupation'].map(occ_map)
df['income'] = df['income'].map(income_map)

# Train
X = df[['age', 'education', 'occupation', 'hours-per-week']]
y = df['income']

model = RandomForestClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)