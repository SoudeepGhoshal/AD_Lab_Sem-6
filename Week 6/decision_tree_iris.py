from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def visualize_tree(clf, feature_names, class_names):
    plt.figure(figsize=(10, 5))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=6,
    )
    plt.title("Decision Tree for Iris Species Classification", fontsize=16)
    plt.show()

def dec_tree(train_size):
    iris = load_iris()
    X = iris.data  # Features: sepal length, sepal width, petal length, petal width
    y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)
    feature_names = iris.feature_names
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=42)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    #visualize_tree(clf, feature_names, class_names)

    return accuracy

if __name__ == '__main__':
    acc = dec_tree(0.7)