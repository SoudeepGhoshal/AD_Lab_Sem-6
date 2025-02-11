from decision_tree_iris import dec_tree
from naive_bayes_classifier_iris import naive_bayes
import pandas as pd
import matplotlib.pyplot as plt

def comp():
    accuracy_dec = {}
    accuracy_nai = {}

    for i in range(5, 10):
        accuracy_dec.update({i/10:dec_tree(i/10)})
        accuracy_nai.update({i/10:naive_bayes(i/10)})

    df = pd.DataFrame({
        'Key': accuracy_dec.keys(),
        'Decision Tree': accuracy_dec.values(),
        'Naive Bayes': accuracy_nai.values()
    })

    print (df)

    plt.figure(figsize=(8, 4))
    plt.plot(df['Key'], df['Decision Tree'], marker='o', label='Decision Tree', linewidth=2)
    plt.plot(df['Key'], df['Naive Bayes'], marker='s', label='Naive Bayes', linewidth=2)

    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.xlabel('Training Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(df['Key'])
    plt.ylim(0.85, 1.01)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    comp()