from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def dtc_model(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier(criterion="gini", splitter="best")
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return dtc, accuracy
