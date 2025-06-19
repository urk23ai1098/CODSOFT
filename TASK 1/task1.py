import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

def load_data():
    return pd.read_csv('Titanic-Dataset.csv')

df = load_data()
df.head()

def get_summary(df):
    return {
        "shape": df.shape,
        "data_types": df.dtypes,
        "missing_values": df.isnull().sum(),
        "unique_counts": df.nunique(),
        "summary_stats": df.describe()
    }

summary = get_summary(df)

summary

def preprocess_data(df):
    df = df.copy()
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    scaler = MinMaxScaler()
    cols_to_scale = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    return X, y, scaler

X, y, scaler = preprocess_data(df)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_save_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    return model

model = train_and_save_model(X_train, y_train)

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)
    report = classification_report(y, predictions)
    return acc, cm, report

acc, cm, report = evaluate_model(model, X_test, y_test)

print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

df['Survived'].value_counts().plot(kind='bar')
plt.xticks([0, 1], ['Did Not Survive', 'Survived'])
plt.title("Survival Count")
plt.show()

df['Age'].hist(bins=30, edgecolor='black')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()

df['Fare'].hist(bins=40, edgecolor='black')
plt.xlabel("Fare")
plt.ylabel("Count")
plt.title("Fare Distribution")
plt.show()

pd.crosstab(df['Sex'], df['Survived']).plot(kind='bar')
plt.title("Survival by Sex")
plt.show()

pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar')
plt.title("Survival by Passenger Class")
plt.show()

df.boxplot(column='Age', by='Pclass')
plt.title("Age vs Pclass")
plt.suptitle("")
plt.show()

df_corr = df.copy()
df_corr['Sex'] = df_corr['Sex'].map({'male': 0, 'female': 1})
df_corr['Embarked'] = df_corr['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df_corr.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
corr = df_corr.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

pd.crosstab(df['Embarked'], df['Survived']).plot(kind='bar')
plt.title("Survival by Port of Embarkation")
plt.show()