import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('IRIS.csv')

le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])  # 0: setosa, 1: versicolor, 2: virginica

x = data.drop(columns='species')
y = data['species']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

model_lg = LogisticRegression(C=1, max_iter=100, multi_class='multinomial', solver='saga')
model_lg.fit(x_train, y_train)


print("Training Accuracy:", model_lg.score(x_train, y_train))
print("Testing Accuracy:", model_lg.score(x_test, y_test))

y_pred = model_lg.predict(x_test)

print("\nTrue label distribution in test set:")
print(y_test.value_counts(normalize=True))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(6, 4))
disp.plot(cmap='RdBu', values_format='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

results_df = pd.DataFrame({
    'Actual': le.inverse_transform(y_test), 
    'Predicted': le.inverse_transform(y_pred)
})

melted_results = results_df.melt(var_name='Type', value_name='Species')

plt.figure(figsize=(8, 5))
sns.countplot(data=melted_results, x='Species', hue='Type', palette='RdBu')
plt.title('Actual vs Predicted Species Count')
plt.xlabel('Species')
plt.ylabel('Count')
plt.legend(title='Label Type')
plt.tight_layout()
plt.show()
