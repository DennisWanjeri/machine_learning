#!/usr/bin/python3
"""Ensemble modelling"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# preparing the dataset
data = pd.read_csv('titanic.csv')
print(data.head())

# preprocess function
def preprocess(data):
    """preprocesses the titanic dataset"""
    def fix_age(age):
        """fix age"""
        if np.isnan(age):
            return -1
        elif age < 1:
            return age*100
        else:
            return age
    # data.dropna(inplace=True)
    data.loc[:, 'Age'] = data.Age.apply(fix_age)
    data.loc[:, 'Sex'] = data.Sex.apply(lambda s: int(s == 'female'))
    embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C', 'Emb_Q', 'Emb_S']]
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return pd.concat([data[cols], embarked], axis=1).values

# Split dataset into training and validation sets
train, val = train_test_split(data, test_size=0.2, random_state=11)
x_train = preprocess(train)
y_train = train['Survived'].values
print(np.any(np.isnan(x_train)))
print(np.all(np.isfinite(x_train)))
x_val = preprocess(val)
y_val = val['Survived'].values

# bootstrap bagging
dt_params = {
    'criterion': 'entropy',
    'random_state': 11
}
dt = DecisionTreeClassifier(**dt_params)
# bagging classifier
bc_params = {
    'base_estimator': dt,
    'n_estimators':50,
    'max_samples':0.5,
    'random_state':11,
    'n_jobs': -1
}
bc = BaggingClassifier(**bc_params)
# fitting the Bagging classifier model
bc.fit(x_train, y_train)
bc_preds_train = bc.predict(x_train)
bc_preds_val = bc.predict(x_val)

print('Bagging Classifier:\n> Accuracy on training data = {:.4f}\n>\
Accuracy on validation data = {:.4f}'.format(accuracy_score(y_true=y_train, y_pred= bc_preds_train),\
                                             accuracy_score(y_true=y_val, y_pred=bc_preds_val)
                                             ))
# fitting decision tree to the training model
dt.fit(x_train, y_train)
dt_preds_train = dt.predict(x_train)
dt_preds_val = dt.predict(x_val)
print('Decision Tree:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=dt_preds_train),
    accuracy_score(y_true=y_val, y_pred=dt_preds_val)
))

# Building the Ensemble Model using Random Forest
rf_params = {
    'n_estimators': 100,
    'criterion': 'entropy',
    'max_features': 0.5,
    'min_samples_leaf': 10,
    'random_state': 11,
    'n_jobs': -1
}
rf = RandomForestClassifier(**rf_params)
rf.fit(x_train, y_train)
rf_preds_train = rf.predict(x_train)
rf_preds_val = rf.predict(x_val)

print('Random Forest:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=rf_preds_train),
    accuracy_score(y_true=y_val, y_pred=rf_preds_val)
))

# classification using adaptive boosting
from sklearn.ensemble import AdaBoostClassifier
dt_params = {
    'max_depth': 1,
    'random_state': 11
}
dt = DecisionTreeClassifier(**dt_params)
ab_params = {
    'n_estimators': 100,
    'base_estimator': dt,
    'random_state': 11
}
ab = AdaBoostClassifier(**ab_params)

# fit the model
ab.fit(x_train, y_train)
ab_preds_train = ab.predict(x_train)
ab_preds_val = ab.predict(x_val)
print('Adaptive Boosting:\n> Accuracy on training data = {:.4f}\n>  Accuracy on validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=ab_preds_train),
    accuracy_score(y_true=y_val, y_pred=ab_preds_val)
))

# prediction accuracy of the model for a varying no. of base estimators
ab_params = {
    'base_estimator': dt,
    'random_state': 11
}
n_estimator_values = list(range(10, 210, 10))
train_accuracies, val_accuracies = [], []

for n_estimators in n_estimator_values:
    ab = AdaBoostClassifier(n_estimators=n_estimators, **ab_params)
    ab.fit(x_train, y_train)
    ab_preds_train = ab.predict(x_train)
    ab_preds_val = ab.predict(x_val)

    train_accuracies.append(accuracy_score(y_true=y_train, y_pred=ab_preds_train))
    val_accuracies.append(accuracy_score(y_true=y_val, y_pred=ab_preds_val))

# plotting a line graph to visualize the prediction accuracies trends
plt.figure(figsize=(10,7))
plt.plot(n_estimator_values, train_accuracies, label='Train')
plt.plot(n_estimator_values, val_accuracies, label='Validation')

plt.ylabel('Accuracy score')
plt.xlabel('n_estimators')
plt.legend()
plt.savefig('accuracy_visualization.png')

# GradientBoosting Classifier
gbc_params = {
    'n_estimators': 100,
    'max_depth': 3,
    'min_samples_leaf': 5,
    'random_state': 11
}
gbc = GradientBoostingClassifier(**gbc_params)
# fit the model
gbc.fit(x_train, y_train)
gbc_preds_train = gbc.predict(x_train)
gbc_preds_val = gbc.predict(x_val)

print('Gradient Boosting Classifier:\n> Accuracy on training data = {:.4f}n> Accuracy on validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=gbc_preds_train),
    accuracy_score(y_true=y_val, y_pred=gbc_preds_val)
    ))

# stacked model

# Base models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
# Stacking model
from sklearn.linear_model import LogisticRegression
# create a new training set with additional columns for predictions
x_train_with_metapreds = np.zeros((x_train.shape[0], x_train.shape[1] + 2))
x_train_with_metapreds[:, :-2] = x_train
x_train_with_metapreds[:, -2:] = -1
print(x_train_with_metapreds)

kf = KFold(n_splits=5, random_state=11)

for train_indices, val_indices in kf.split(x_train):
    kfold_x_train, kfold_x_val = x_train[train_indices], x_train[val_indices]
    kfold_y_train, kfold_y_val = y_train[train_indices], y_train[val_indices]
    svm = LinearSVC(random_state=11, max_iter=1000)
    svm.fit(kfold_x_train, kfold_y_train)
    svm_pred = svm.predict(kfold_x_val)

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(kfold_x_train, kfold_y_train)
    knn_pred = knn.predict(kfold_x_val)

    x_train_with_metapreds[val_indices, -2] = svm_pred
    x_train_with_metapreds[val_indices, -1] = knn_pred

# create a new validation set
x_val_with_metapreds = np.zeros((x_val.shape[0], x_val.shape[1]+2))
x_val_with_metapreds[:, :-2] = x_val
x_val_with_metapreds[:, -2:] = -1
print(x_val_with_metapreds)

# fit base models on the complete training
svm = LinearSVC(random_state=11, max_iter=1000)
svm.fit(x_train, y_train)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
svm_pred = svm.predict(x_val)
knn_pred = knn.predict(x_val)
x_val_with_metapreds[:, -2] = svm_pred
x_val_with_metapreds[:, -1] = knn_pred

#train stacked model
lr = LogisticRegression(random_state=11)
lr.fit(x_train_with_metapreds, y_train)
lr_preds_train = lr.predict(x_train_with_metapreds)
lr_preds_val = lr.predict(x_val_with_metapreds)
print('Stacked Classifier:\n> Accuracy on training data = {:.4f}\n> Accuracy on validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=lr_preds_train),
    accuracy_score(y_true=y_val, y_pred=lr_preds_val)
    ))

# compare accuracy with basemodels
print('SVM:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=svm.predict(x_train)),
    accuracy_score(y_true=y_val, y_pred=svm_pred)
))
print('kNN:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=knn.predict(x_train)),
    accuracy_score(y_true=y_val, y_pred=knn_pred)
    ))
