import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

##################
stars_data = pd.read_csv("Stars.csv")
stars_names = ['Red Dwarf', 'Brown Dwarf', 'White Dwarf', 'Main Sequence', 'Super Giants', 'Hyper Giants']

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

##################


# 70-80% is usual but sometimes it gets even more (or lower)
def gnb_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    gnb = GaussianNB().fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    return gnb


# Usually 85-90% 
if __name__ == '__main__':
    for i in range(10):
        model = gnb_model(X, y)
