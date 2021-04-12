import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# 70-80% is usual but sometimes it gets even even more <3
def knn_model(X_train, X_test, y_train, y_test):
    i_range = range(1, 60)
    max_accuracy = 0
    accuracies = []

    knn = KNeighborsClassifier(algorithm='brute')
    knn_max = knn

    for i in i_range:
        knn.n_neighbors = i
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # print(str(i) + ' ' + str(acc))
        accuracies.append(acc)

        if acc > max_accuracy:
            max_accuracy = acc
            knn_max = knn

    # plt.plot(i_range, accuracies)
    # plt.title("Accuracies")
    # plt.show()

    return knn_max, max_accuracy
