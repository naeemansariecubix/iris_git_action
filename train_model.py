# Corrected code with fixes applied

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score
sns.set(style='white')

# Load Data
dataset = pd.read_csv('iris.csv')

# Feature names
dataset.columns = [colname.strip(' (cm)').replace(" ", "_") for colname in dataset.columns.tolist()]
features_names = dataset.columns.tolist()[:4]

# Feature Engineering
dataset['sepal_legth_width_ratio'] = dataset['sepal_length'] / dataset['sepal_width']
dataset['petal_legth_width_ratio'] = dataset['petal_length'] / dataset['petal_width']

# Select Features
dataset = dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
                   'sepal_legth_width_ratio', 'petal_legth_width_ratio', 'target']]

# Training
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=44)

# X_train, y_train, X_test, y_test
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')

# Logistic Regression
logreg = LogisticRegression(C=0.0001, solver='lbfgs', max_iter=100, multi_class='multinomial')
logreg.fit(X_train, y_train)
predictions_lr = logreg.predict(X_test)
cm = confusion_matrix(y_test, predictions_lr)

f1 = f1_score(y_test, predictions_lr, average='micro')
prec = precision_score(y_test, predictions_lr, average='micro')
recall = recall_score(y_test, predictions_lr, average='micro')

train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
predictions_rf = rf_clf.predict(X_test)

f1_rf = f1_score(y_test, predictions_rf, average='micro')
prec_rf = precision_score(y_test, predictions_rf, average='micro')
recall_rf = recall_score(y_test, predictions_rf, average='micro')

train_acc_rf = rf_clf.score(X_train, y_train) * 100
test_acc_rf = rf_clf.score(X_test, y_test) * 100

# Confusion Matrix Plotting
def plot_cm(cm, target_name, title="Confusion Matrix", cmap=None, normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_name))
    plt.xticks(tick_marks, target_name, rotation=45)
    plt.yticks(tick_marks, target_name)

    with np.errstate(divide='ignore', invalid='ignore'):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:0.4f}" if normalize else f"{int(cm[i, j]):,}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel(f'Predicted Label\naccuracy={accuracy:0.4f}; misclass={misclass:0.4f}')
    plt.savefig('ConfusionMatrix.png', dpi=120)
    plt.show()

target_name = ['setosa', 'versicolor', 'virginica']
plot_cm(cm, target_name, title="Confusion Matrix", cmap=None, normalize=True)

# Feature Importance Plot
importances = rf_clf.feature_importances_
labels = dataset.drop('target', axis=1).columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=['feature', 'importance'])
features = feature_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='importance', y='feature', data=features)
ax.set_xlabel('Importance', fontsize=14)
ax.set_ylabel('Feature', fontsize=14)
ax.set_title('Random Forest Feature Importances', fontsize=14)
plt.tight_layout()
plt.savefig('FeatureImportance.png')
plt.close()

# Save Scores to a Text File
with open('scores.txt', "w") as score:
    score.write(f"Random Forest Train Var: {train_acc_rf:.1f}%\n")
    score.write(f"Random Forest Test Var: {test_acc_rf:.1f}%\n")
    score.write(f"F1 Score: {f1_rf:.1f}%\n")
    score.write(f"Recall Score: {recall_rf:.1f}%\n")
    score.write(f"Precision Score: {prec_rf:.1f}%\n\n")
    score.write(f"Logistic Regression Train Var: {train_acc_lr:.1f}%\n")
    score.write(f"Logistic Regression Test Var: {test_acc_lr:.1f}%\n")
    score.write(f"F1 Score: {f1:.1f}%\n")
    score.write(f"Recall Score: {recall:.1f}%\n")
    score.write(f"Precision Score: {prec:.1f}%\n")
