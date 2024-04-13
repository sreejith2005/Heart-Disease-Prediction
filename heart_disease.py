import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import RocCurveDisplay,confusion_matrix, classification_report, f1_score, recall_score, precision_score

import pickle

df = pd.read_csv("heart-disease.csv")
df.head()

#splitting the data 
np.random.seed(42)
X = df.drop("target", axis =1)
y = df["target"]
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2)

models = {"RF":RandomForestClassifier(),
          "LR": LogisticRegression(),
          "DT": DecisionTreeClassifier()}

def fit_and_score(models, X_train,y_train,X_test, y_test):
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)

    return model_scores

fit_and_score(models, X_train=X_train,y_train=y_train,X_test=X_test, y_test=y_test)

#tuning
LR_grid = {"C":[1],
          "solver": ["liblinear"]}

LR_rs = RandomizedSearchCV(LogisticRegression(), param_distributions = LR_grid, cv=5, verbose=2, n_iter=20)

# LR_rs.best_params_

# clf = LogisticRegression(solver = "liblinear", C=1)
LR_rs.fit(X_train, y_train)
LR_rs.score(X_test, y_test)

y_preds = LR_rs.predict(X_test)
y_preds

RocCurveDisplay.from_estimator(estimator=LR_rs, X=X_test, y=y_test)

report = classification_report(y_test,y_preds)
report

conf = confusion_matrix(y_test, y_preds)
conf

#plottitng with seaborn

def plot_conf(y_test, y_preds):
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds), annot = True, cbar=False)
plot_conf(y_test, y_preds)



clf = LogisticRegression(solver= 'liblinear', C= 1)

cv_acc = cross_val_score(clf,X,y,scoring='accuracy')
cv_acc = np.mean(cv_acc)
cv_acc

cv_f1 = cross_val_score(clf,X,y,scoring='f1')
cv_f1 = np.mean(cv_f1)
cv_f1

cv_recall = cross_val_score(clf,X,y,scoring='accuracy')
cv_recall = np.mean(cv_recall)
cv_recall

cv_precision = cross_val_score(clf,X,y,scoring='accuracy')
cv_precision = np.mean(cv_precision)
cv_precision

clf.fit(X_train, y_train)
clf.coef_

feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict

features_df = pd.DataFrame(feature_dict, index=[0])
features_df.T.plot.bar(title = "Feature Importance", legend=False)

pickle.dump(clf, open("Heart-Disease-Project.pkl", "wb"))

load_pickle_model = pickle.load(open("Heart-Disease-Project.pkl", "rb"))
