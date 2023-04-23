from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('resto_preprocessed.csv', delimiter=";")


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')


def display_knn_plot(x_data, y_data, classifier, title):
    plt.figure()
    plot_decision_regions(x_data, y_data, classifier=classifier)
    plt.xlabel(dataset.columns[2])
    plt.ylabel(dataset.columns[5])
    plt.legend(loc='upper left')
    plt.title(title)
    plt.show()


x = dataset.iloc[:, [2, 5]].values
y = dataset.iloc[:, 6].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
correct_predictions = np.trace(cm)
total_samples = len(y_test)
accuracy = correct_predictions / total_samples
print("Accuracy : ", accuracy)

# Overlapping markers might indicate that KNN is not the best algorithm for this dataset
display_knn_plot(x_train, y_train, classifier,
                 'KNN Decision Regions (Training set)')
display_knn_plot(x_test, y_test, classifier,
                 'KNN Decision Regions (Testing set)')
