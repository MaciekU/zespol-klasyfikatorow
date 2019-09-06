# A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally), whereas a micro-average will aggregate the contributions of all classes to compute the average metric. In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance (i.e you may have many more examples of one class than of other classes). 

import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def Bagging(wartosci,klasy,splits):
    kfold = StratifiedKFold(n_splits=splits)
    model = BaggingClassifier(n_estimators=100, bootstrap=False)
    results = cross_val_score(model, wartosci, klasy, cv=kfold, scoring="f1_micro")
    result = results.mean()
    std = results.std()  # standard deviation 
    print("Bagging F1_mean:",result,std)
    return

def RandomForest(wartosci,klasy,splits):
    kfold = StratifiedKFold(n_splits=splits)
    model = RandomForestClassifier(n_estimators=100, criterion="gini",max_features=8)  # entropy max_features=3
    results = cross_val_score(model, wartosci, klasy, cv=kfold, scoring="f1_micro")
    result = results.mean()
    std = results.std() #standard deviation 
    print("RandomForest F1_mean:",result,std)
    return

def AdaBoost(wartosci, klasy, splits):
    kfold = StratifiedKFold(n_splits=splits)
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.2)
    results = cross_val_score(model, wartosci, klasy, cv=kfold, scoring="f1_micro")
    result = results.mean()
    std = results.std() #standard deviation 
    print("Boosting F1_mean:",result,std)
    return

dane_diabetes = pandas.read_csv('pima_indians_diabetes.txt', header=None, delimiter=',', engine='python')
dane_diabetes.columns = ['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 'Serum_insu', 'BMI', 'Diabetes_func', 'Age', 'Class']
wartosci = dane_diabetes[['No_pregnant', 'Plasma_glucose', 'Blood_pres', 'Skin_thick', 'Serum_insu', 'BMI', 'Diabetes_func', 'Age']].values
klasy = dane_diabetes['Class'].values

print("Diabetes:")
Bagging(wartosci,klasy,10)
RandomForest(wartosci,klasy,10)
AdaBoost(wartosci, klasy,10)


dane_glass = pandas.read_csv('glass.data', header=None, delimiter=',', engine='python')
dane_glass.columns = ['Id number','RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type of glass' ]
dane_glass.drop('Id number',axis=1, inplace=True)
wartosci = dane_glass[['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']].values
klasy = dane_glass['Type of glass'].values

print("Glass:")
Bagging(wartosci,klasy,9)
RandomForest(wartosci,klasy,9)
AdaBoost(wartosci, klasy,9)

dane_wine = pandas.read_csv('wine.data', header=None, delimiter=',', engine='python')
dane_wine.columns = ['Class','Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
wartosci = dane_wine[['Alcohol', 'Malic Acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']].values
klasy = dane_wine['Class'].values

print("Wine:")
Bagging(wartosci,klasy,10)
RandomForest(wartosci,klasy,10)
AdaBoost(wartosci, klasy,10)
