from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def decision_tree():
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred


def kNN_classifier():
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred


def mini_batch_gradient_descent():
    model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred


def naive_bayes_classifier():
    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred


def svm():
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    return y_pred


def evaluate(y_pred, model):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f'Model: {model}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print()


def main():
    y_pred = decision_tree()
    evaluate(y_pred, 'Decision Tree')

    y_pred = kNN_classifier()
    evaluate(y_pred, 'kNN Classifier')

    y_pred = mini_batch_gradient_descent()
    evaluate(y_pred, 'Mini-Batch Gradient Descent')

    y_pred = naive_bayes_classifier()
    evaluate(y_pred, 'Naive Bayes Classifier')

    y_pred = svm()
    evaluate(y_pred, 'Support Vector Machine')

if __name__ == '__main__':
    main()