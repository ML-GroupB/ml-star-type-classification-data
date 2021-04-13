from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def rfc_model(X_train, X_test, y_train, y_test):
    rfc = RandomForestClassifier(n_estimators=500)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return rfc, accuracy
