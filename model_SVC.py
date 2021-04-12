import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Function that create and test model and then return the model fit to whole dataset

def svc_linear(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    y_predict = svc.predict(X_test)
    print(accuracy_score(y_test, y_predict))
    svc.fit(X, y)
    return svc


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

if __name__ == '__main__':
    stars_data = pd.read_csv('Stars.csv')
    stars_names = {'Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants'}
    stars_data.Color.loc[stars_data.Color == 'Blue-white'] = 'Blue-White'
    stars_data.Color.loc[stars_data.Color == 'Blue White'] = 'Blue-White'
    stars_data.Color.loc[stars_data.Color == 'Blue white'] = 'Blue-White'
    stars_data.Color.loc[stars_data.Color == 'yellow-white'] = 'White-Yellow'
    stars_data.Color.loc[stars_data.Color == 'Yellowish White'] = 'White-Yellow'
    stars_data.Color.loc[stars_data.Color == 'white'] = 'White'
    stars_data.Color.loc[stars_data.Color == 'yellowish'] = 'Yellowish'

    stars_data.Color = pd.Categorical(stars_data.Color)
    stars_data.Color = stars_data.Color.cat.codes
    stars_data.Spectral_Class = pd.Categorical(stars_data.Spectral_Class)
    stars_data.Spectral_Class = stars_data.Spectral_Class.cat.codes

    X = stars_data.drop(columns="Type")
    y = stars_data["Type"]

    svc_linear(X, y)
