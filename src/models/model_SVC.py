from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# Function that create and test model and then return the model fit to whole dataset

def svc_linear(X_train, X_test, y_train, y_test):
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return svc, accuracy


"""

# Testing which kernel is the best

def training(function_type, c_number):
    svm = SVC(kernel=function_type, C=c_number)
    svm.fit(X_train, y_train)

    predictions = svm.predict(X_test)

    return accuracy_score(y_test, predictions)

c_values = [i/5 for i in range(1, 501, 5)]

acc_values = [0.0]*len(c_values)

types = {'linear', 'sigmoid', 'rbf', 'poly'}
for tpe in types:
    for i in range(len(acc_values)):
        acc_values[i] = training(tpe, c_values[i])
    plt.plot(c_values, acc_values)
    plt.title(label=tpe)
    plt.show()
"""
