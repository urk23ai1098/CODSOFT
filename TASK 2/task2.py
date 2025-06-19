import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('IRIS.csv')
data.head()

data.info()

data.isnull().sum()

data.columns

data.sepal_length.min()

data.sepal_length.max()

data.sepal_width.min()

data.sepal_width.max()

data.petal_length.min()

data.petal_length.max()

data.petal_width.min()

data.petal_width.max()

data.species.value_counts()

data.sepal_length.nunique()/data.shape[0]*100

data.sepal_width.nunique()/data.shape[0]*100

data.petal_length.nunique()/data.shape[0]*100

data.petal_width.nunique()/data.shape[0]*100

data.species.value_counts(normalize=True)*100

color = sns.color_palette('RdBu')
plt.pie(data['species'].value_counts(), labels=data['species'].unique(), autopct='%1.f%%',colors=color);

data.groupby('species')['sepal_length'].mean()

color = sns.color_palette('RdBu')
data.groupby('species')['sepal_length'].mean().plot(kind='bar',color=color)

data.groupby('species')['sepal_width'].mean()

color = sns.color_palette('RdBu')
data.groupby('species')['sepal_width'].mean().plot(kind='bar', color=color)

data.groupby('species')['petal_length'].mean()

color = sns.color_palette('RdBu')
data.groupby('species')['petal_length'].mean().plot(kind='bar',color=color)

data.groupby('species')['petal_width'].mean()

color = sns.color_palette('RdBu')
data.groupby('species')['petal_width'].mean().plot(kind='bar',color=color)

le =LabelEncoder()
data['species'] = le.fit_transform(data['species'])

data

data.sample(10)

x=data.drop(columns='species',axis=1)
y=data.species

x

y

model_params={
    'linear_regression':{
        'model':LinearRegression(),
        'params':{
            'fit_intercept':[True,False],
            'positive':[True,False]
        }
    },
    'Logistic_Regression':{
        'model':LogisticRegression(),
        'params':{
                'penalty':['l1', 'l2'],
                'C':[0.01, 0.1, 1, 10],
                'solver':['saga'],
                'max_iter':[100, 200, 500],
                'multi_class':['multinomial']
            }
        },
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[50,100,200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]

        }
    }


}

score=[]
for model_name,mp in model_params.items():
    clf=GridSearchCV(mp['model'],mp['params'],cv=5,return_train_score=True)
    clf.fit(x,y)
    score.append({
        'model': model_name,
        'train_score': clf.cv_results_['mean_train_score'][clf.best_index_],
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
df = pd.DataFrame(score,columns=['model','train_score','best_score','best_params'])
df

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

model_lg=LogisticRegression(C=1,max_iter=100,multi_class='multinomial',solver='saga')

model_lg.fit(x_train,y_train)

model_lg.score(x_train,y_train)

model_lg.score(x_test,y_test)

print(y_test.value_counts(normalize=True))