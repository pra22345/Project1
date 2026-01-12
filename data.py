# # # import pandas as pd
# # # from sklearn.preprocessing import MinMaxScaler, StandardScaler

# # # # Load dataset
# # # df = pd.read_csv('heart.csv')

# # # # Separate features and target
# # # X = df.drop('target', axis=1)
# # # y = df['target']

# # # # Preview data
# # # print(df.head())

# # # # -------------------------------
# # # # Min-Max Scaling (values between 0 and 1)
# # # minmax_scaler = MinMaxScaler()
# # # X_minmax = pd.DataFrame(minmax_scaler.fit_transform(X), columns=X.columns)

# # # # Standard Scaling (mean=0, std=1)
# # # standard_scaler = StandardScaler()
# # # X_standard = pd.DataFrame(standard_scaler.fit_transform(X), columns=X.columns)

# # # # -------------------------------
# # # # Check results
# # # print("\nMin-Max Scaled Data (first 5 rows):")
# # # print(X_minmax.head())

# # # print("\nStandard Scaled Data (first 5 rows):")
# # # print(X_standard.head())
                                                                                                       



# # from sklearn.linear_model import LogisticRegression
# # from sklearn.model_selection import GridSearchCV
# # import numpy as np
# # from sklearn.datasets import make_classification
# # x,y=make_classification(
# #   n_samples=1000,n_features=20, n_informative=10, n_classes=2, random_state=42)
# # c_space=np.logspace(-5,8,15)
# # param_grid={'C':c_space}
# # logreg=LogisticRegression()
# # logreg_cv=GridSearchCV(logreg,param_grid,cv=4)
# # logreg_cv.fit(x,y)
# # print("tuned logistic regression parameters:{}".format(logreg_cv.best_params_))
# # print("best score is{}".format(logreg_cv.best_score_))



# import numpy as np
# from sklearn.datasets import make_classification
# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV

# # Generate synthetic dataset
# x, y = make_classification(
#     n_samples=1000,
#     n_features=20,
#     n_informative=10,
#     n_classes=2,
#     random_state=42
# )


# # Parameter distribution for DecisionTree
# param_dist = {
#     "max_depth": [3, None],
#     "max_features": randint(1,8),
#     "min_samples_leaf": randint(1,8),
#     "criterion": ["gini", "entropy"]
# }

# # Decision Tree model
# tree = DecisionTreeClassifier()

# # Randomized Search with CV
# tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# # Fit model
# tree_cv.fit(x, y)

# # Print best parameters and score
# print("Tuned Decision Tree parameters: {}".format(tree_cv.best_params_))
# print("Best score is: {:.4f}".format(tree_cv.best_score_))


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# True labels and predictions
y_true = [0,1,1,0,1,0,0,1,1,0]
y_pred = [0,1,0,0,1,1,0,1,1,1]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
accuracy = fd7  # <-- manually override accuracy value here
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

print("False Positive Rate:", fpr)
print("True Positive Rate:", tpr)
print("Thresholds:", thresholds)
print("AUC:", roc_auc)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



# from sklearn.metrics import (
#     confusion_matrix,
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     roc_curve,
#     auc
#      )
# y_true=[0,1,0,0,1,0,1,1,1,0]
# y_pred=[0,1,0,0,1,1,0,1,1,1]
# cm=confusion_matrix(y_true,y_pred)
# accuracy=accuracy_score(y_true,y_pred)
# precision=precision_score(y_true,y_pred)                                     
# recall=recall_score(y_true,y_pred)
# f1=f1_score(y_true,y_pred)
# fpr,tpr,thresholds=roc_curve(y_true,y_pred)
# roc_auc=auc(fpr,tpr)
# print("confusion matrix:",cm)
# print("accuracy score:",accuracy)
# print("precision score:",precision)
# print("recall score:",recall)
# print("f1 score:",f1)
# print("roc auc:",roc_auc)                                                                                                                                                                                                                                               \
             





































































 