import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# from sklearn.inspection import plot_partial_dependence

# Define the URL of the dataset and column names
url_rf = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
column_names = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
                'coarse_aggregate', 'fine_aggregate', 'age', 'compressive_strength']

# Load the dataset
data = pd.read_excel(url_rf, names=column_names)

# Regression Task: Predicting 'compressive_strength' (Continuous)
X_reg = data.drop(columns=['compressive_strength'])
y_reg = data['compressive_strength']
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Classification Task: Binary Classification based on 'compressive_strength' (Discrete)
threshold = 40  # Adjust the threshold as needed
data['compressive_strength_label'] = (data['compressive_strength'] >= threshold).astype(int)
X_clf = data.drop(columns=['compressive_strength', 'compressive_strength_label'])
y_clf = data['compressive_strength_label']
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)


# Support Vector Classifier (SVC) for classification
svc_model = SVC(kernel='linear')
svc_model.fit(X_clf_train, y_clf_train)
svc_predictions = svc_model.predict(X_clf_test)
svc_accuracy = accuracy_score(y_clf_test, svc_predictions)

# Random Forest Classifier
rf_clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf_model.fit(X_clf_train, y_clf_train)
rf_clf_predictions = rf_clf_model.predict(X_clf_test)
rf_clf_accuracy = accuracy_score(y_clf_test, rf_clf_predictions)

# # ROC Curve for Random Forest Classifier
# rf_probs = rf_clf_model.predict_proba(X_clf_test)[:, 1]
# fpr, tpr, thresholds = roc_curve(y_clf_test, rf_probs)
# roc_auc = auc(fpr, tpr)


# # Plot ROC Curve for Random Forest Classifier
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.title('Random Forest Classifier - ROC Curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig('assets/random_forest_roc_curve.png')


# # Partial Dependence Plot for Random Forest Classifier
# feature_importance = rf_clf_model.feature_importances_
# sorted_idx = np.argsort(feature_importance)[::-1]
# plt.figure()
# plt.bar(range(X_clf_train.shape[1]), feature_importance[sorted_idx])
# plt.xticks(range(X_clf_train.shape[1]), X_clf_train.columns[sorted_idx], rotation=90)
# plt.xlabel('Feature')
# plt.ylabel('Feature Importance')
# plt.title('Random Forest Classifier - Feature Importance')
# plt.tight_layout()
# plt.show()
# plt.savefig('assets/random_forest_feature_importance.png')

# Plot Decision Boundary for SVC
plt.figure()
plt.scatter(X_clf_train.iloc[:, 0], X_clf_train.iloc[:, 1], c=y_clf_train, alpha=0.8)
plt.xlabel('Cement')
plt.ylabel('Blast Furnace Slag')
plt.title('SVC - Decision Boundary')
plt.tight_layout()
plt.show()
plt.savefig('assets/svc_decision_boundary.png')


print('SVM Classifier Accuracy: ', svc_accuracy)
print('Random Forest Classifier Accuracy: ', rf_clf_accuracy)