import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
def knn_model(X, y):
    accuracy = []
    i_range = range(1, 60)

    for j in range(1, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        for i in i_range:
            knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute').fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            # print(str(i) + ' ' + str(acc))
            accuracy.append(acc)
        plt.plot(i_range, accuracy)
        plt.title(j)
        plt.show()
        print(accuracy.index(max(accuracy))+1)  # 1 - 3 is best n_neighbours
        # print(max(accuracy))
        accuracy.clear()


if __name__ == '__main__':
    knn_model(X, y)
