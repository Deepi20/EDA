from sklearn.datasets import load_iris
data = load_iris()
data
import pandas as pd
pd.read_csv(data.filename)
from yellowbrick.features import ParallelCoordinates
viz = ParallelCoordinates(features=data.feature_names,
                          classes=data.target_names,
                          normalize='standard')
viz.fit(data.data, data.target)
viz.transform(data.data)
viz.poof()
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
scaler.fit(X_train)
X1 = scaler.transform(X_train)

clf = DecisionTreeClassifier()
clf.fit(X1, y_train)
X2 = scaler.transform(X_test)
clf.predict(X2)
from sklearn.pipeline import Pipeline
decision_pipeline = Pipeline([
    ('normalize', StandardScaler()),
    ('decision', DecisionTreeClassifier())
])
decision_pipeline.fit(X_train, y_train)
decision_pipeline.predict(X_test)
decision_pipeline.score(X_test, y_test)
scaler2 = decision_pipeline.steps[0][1]
clf2 = decision_pipeline.steps[1][1]
clf2.predict(scaler2.transform(X_test))
clf2.predict(X_test)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
clf = DecisionTreeClassifier()
clf
param_range = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9]
}
clf = GridSearchCV(DecisionTreeClassifier(),
                   param_range,
                   cv=3,
                   verbose=5)
clf.fit(X_train, y_train)
clf.best_estimator_
clf.predict(X_test)
clf.score(X_test, y_test)
clf = RandomizedSearchCV(DecisionTreeClassifier(),
                         param_range,
                         cv=3,
                         verbose=5,
                         n_iter=5)
