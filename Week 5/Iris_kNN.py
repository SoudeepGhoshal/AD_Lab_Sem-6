import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Classifier with k=3
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
y_pred_knn_3 = knn_3.predict(X_test)
conf_matrix_knn_3 = confusion_matrix(y_test, y_pred_knn_3)
accuracy_knn_3 = accuracy_score(y_test, y_pred_knn_3)

# KNN Classifier with k=13
knn_13 = KNeighborsClassifier(n_neighbors=13)
knn_13.fit(X_train, y_train)
y_pred_knn_13 = knn_13.predict(X_test)
conf_matrix_knn_13 = confusion_matrix(y_test, y_pred_knn_13)
accuracy_knn_13 = accuracy_score(y_test, y_pred_knn_13)

# KNN Classifier with k=20
knn_20 = KNeighborsClassifier(n_neighbors=20)
knn_20.fit(X_train, y_train)
y_pred_knn_20 = knn_20.predict(X_test)
conf_matrix_knn_20 = confusion_matrix(y_test, y_pred_knn_20)
accuracy_knn_20 = accuracy_score(y_test, y_pred_knn_20)

# Logistic Regressor
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

print(f'Accuracy of KNN (k=3): {accuracy_knn_3 * 100:.2f}%')
print(f'Accuracy of KNN (k=13): {accuracy_knn_13 * 100:.2f}%')
print(f'Accuracy of KNN (k=20): {accuracy_knn_20 * 100:.2f}%')
print(f'Accuracy of Logistic Regression: {accuracy_log_reg * 100:.2f}%')

fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Plot for KNN (k=3)
axes[0][0].imshow(conf_matrix_knn_3, interpolation='nearest', cmap=plt.cm.Blues)
axes[0][0].set_title('Confusion Matrix (KNN k=3)')
axes[0][0].set_xticks(np.arange(3))
axes[0][0].set_yticks(np.arange(3))
axes[0][0].set_xticklabels(iris.target_names)
axes[0][0].set_yticklabels(iris.target_names)
axes[0][0].set_xlabel('Predicted Label')
axes[0][0].set_ylabel('True Label')
for i in range(3):
    for j in range(3):
        axes[0][0].text(j, i, conf_matrix_knn_3[i, j], ha="center", va="center", color="white")

# Plot for KNN (k=13)
axes[0][1].imshow(conf_matrix_knn_13, interpolation='nearest', cmap=plt.cm.Blues)
axes[0][1].set_title('Confusion Matrix (KNN k=13)')
axes[0][1].set_xticks(np.arange(3))
axes[0][1].set_yticks(np.arange(3))
axes[0][1].set_xticklabels(iris.target_names)
axes[0][1].set_yticklabels(iris.target_names)
axes[0][1].set_xlabel('Predicted Label')
axes[0][1].set_ylabel('True Label')
for i in range(3):
    for j in range(3):
        axes[0][1].text(j, i, conf_matrix_knn_13[i, j], ha="center", va="center", color="white")

# Plot for KNN (k=20)
axes[1][0].imshow(conf_matrix_knn_20, interpolation='nearest', cmap=plt.cm.Blues)
axes[1][0].set_title('Confusion Matrix (KNN k=20)')
axes[1][0].set_xticks(np.arange(3))
axes[1][0].set_yticks(np.arange(3))
axes[1][0].set_xticklabels(iris.target_names)
axes[1][0].set_yticklabels(iris.target_names)
axes[1][0].set_xlabel('Predicted Label')
axes[1][0].set_ylabel('True Label')
for i in range(3):
    for j in range(3):
        axes[1][0].text(j, i, conf_matrix_knn_20[i, j], ha="center", va="center", color="white")

# Plot for Logistic Regression
axes[1][1].imshow(conf_matrix_log_reg, interpolation='nearest', cmap=plt.cm.Blues)
axes[1][1].set_title('Confusion Matrix (Logistic Regression)')
axes[1][1].set_xticks(np.arange(3))
axes[1][1].set_yticks(np.arange(3))
axes[1][1].set_xticklabels(iris.target_names)
axes[1][1].set_yticklabels(iris.target_names)
axes[1][1].set_xlabel('Predicted Label')
axes[1][1].set_ylabel('True Label')
for i in range(3):
    for j in range(3):
        axes[1][1].text(j, i, conf_matrix_log_reg[i, j], ha="center", va="center", color="white")

plt.tight_layout()
plt.show()

accuracies = {
    "KNN (k=3)": accuracy_knn_3,
    "KNN (k=13)": accuracy_knn_13,
    "KNN (k=20)": accuracy_knn_20,
    "Logistic Regression": accuracy_log_reg
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nThe best performing model is: {best_model}.")