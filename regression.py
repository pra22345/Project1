# # # # from sklearn.datasets import load_breast_cancer
# # # # import matplotlib.pyplot as plt
# # # # from sklearn.inspection import DecisionBoundaryDisplay
# # # # from sklearn.svm import SVC
# # # # cancer = load_breast_cancer()
# # # # X = cancer.data[:, :2]
# # # # y = cancer.target
# # # # svm = SVC(kernel="linear", C=1)
# # # # svm.fit(X, y)
# # # # DecisionBoundaryDisplay.from_estimator(
# # # # svm,
# # # # X,
# # # # response_method="predict",
# # # # alpha=0.8,
# # # # cmap="Pastel1",
# # # # xlabel=cancer.feature_names[0],
# # # # ylabel=cancer.feature_names[1],
# # # # )
# # # # plt.scatter(X[:, 0], X[:, 1], 
# # # # c=y, 
# # # # s=20, edgecolors="red")
# # # # plt.title("SVM Decision Boundary on Breast Cancer", fontweight="bold")


# # # # plt.show()


# # # import math
# # # import random
# # # import pandas as pd
# # # import numpy as np

# # # def encode_class(mydata):
# # #     classes = []
# # #     for i in range(len(mydata)):
# # #         if mydata[i][-1] not in classes:
# # #             classes.append(mydata[i][-1])
# # #     for i in range(len(classes)):
# # #         for j in range(len(mydata)):
# # #             if mydata[j][-1] == classes[i]:
# # #                 mydata[j][-1] = i
# # #     return mydata

# # # def splitting(mydata, ratio):
# # #     train_num = int(len(mydata) * ratio)
# # #     train = []
# # #     test = list(mydata)
    
# # #     while len(train) < train_num:
# # #         index = random.randrange(len(test))
# # #         train.append(test.pop(index))
# # #     return train, test

# # # def groupUnderClass(mydata):
# # #     data_dict = {}
# # #     for i in range(len(mydata)):
# # #         if mydata[i][-1] not in data_dict:
# # #             data_dict[mydata[i][-1]] = []
# # #         data_dict[mydata[i][-1]].append(mydata[i])
# # #     return data_dict

# # # def MeanAndStdDev(numbers):
# # #     avg = np.mean(numbers)
# # #     stddev = np.std(numbers)
# # #     return avg, stddev

# # # def MeanAndStdDevForClass(mydata):
# # #     info = {}
# # #     data_dict = groupUnderClass(mydata)
# # #     for classValue, instances in data_dict.items():
# # #         info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
# # #     return info

# # # def calculateGaussianProbability(x, mean, stdev):
# # #     epsilon = 1e-10
# # #     expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
# # #     return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

# # # def calculateClassProbabilities(info, test):
# # #     probabilities = {}
# # #     for classValue, classSummaries in info.items():
# # #         probabilities[classValue] = 1
# # #         for i in range(len(classSummaries)):
# # #             mean, std_dev = classSummaries[i]
# # #             x = test[i]
# # #             probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
# # #     return probabilities

# # # def predict(info, test):
# # #     probabilities = calculateClassProbabilities(info, test)
# # #     bestLabel = max(probabilities, key=probabilities.get)
# # #     return bestLabel

# # # def getPredictions(info, test):
# # #     predictions = [predict(info, instance) for instance in test]
# # #     return predictions

# # # def accuracy_rate(test, predictions):
# # #     correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
# # #     return (correct / float(len(test))) * 100.0

# # # filename = 'diabetes_data.csv'

# # # # Read with headers
# # # df = pd.read_csv(filename)

# # # # Convert to list of lists
# # # mydata = df.values.tolist()

# # # # Encode class labels
# # # mydata = encode_class(mydata)

# # # # Convert features to float
# # # for i in range(len(mydata)):
# # #     for j in range(len(mydata[i]) - 1):  # skip the last column (class)
# # #         mydata[i][j] = float(mydata[i][j])

# # # ratio = 0.7
# # # train_data, test_data = splitting(mydata, ratio)

# # # print('Total number of examples:', len(mydata))
# # # print('Training examples:', len(train_data))
# # # print('Test examples:', len(test_data))

# # # info = MeanAndStdDevForClass(train_data)

# # # predictions = getPredictions(info, test_data)
# # # accuracy = accuracy_rate(test_data, predictions)
# # # print('Accuracy of the model:', accuracy)

# # # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # # y_true = [row[-1] for row in test_data]
# # # y_pred = predictions

# # # cm = confusion_matrix(y_true, y_pred)
# # # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# # # disp.plot(cmap='Blues')

# # # import matplotlib.pyplot as plt
# # # from sklearn.metrics import precision_score, recall_score, f1_score

# # # actual = [0, 1, 1, 0, 1, 0, 1, 1]
# # # predicted = [0, 1, 0, 0, 1, 0, 1, 0]

# # # precision = precision_score(actual, predicted)
# # # recall = recall_score(actual, predicted)
# # # f1 = f1_score(actual, predicted)

# # # metrics = ['Precision', 'Recall', 'F1 Score']
# # # values = [precision, recall, f1]

# # # plt.figure(figsize=(6, 4))
# # # plt.bar(metrics, values, color=['Red', 'Pink', 'Black'])
# # # plt.ylim(0, 1)
# # # plt.title('Precision, Recall, and F1 Score')
# # # plt.ylabel('Score')
# # # for i, v in enumerate(values):
# # #     plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
# # # plt.show()

# # import numpy as np
# # from collections import Counter

# # def euclidean_distance(point1, point2):
# #  return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

# # def knn_predict(training_data, training_labels, test_point, k):
# #   distances = []
# #   for i in range(len(training_data)):
# #    dist = euclidean_distance(test_point, training_data[i])
# #    distances.append((dist, training_labels[i]))
# #    distances.sort(key=lambda x: x[0])
# #    k_nearest_labels = [label for _, label in distances[:k]]
# #   return Counter(k_nearest_labels).most_common(1)[0][0]

# # training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
# # training_labels = ['A', 'A', 'A', 'B', 'B']
# # test_point = [4, 5]
# # k = 3
# # prediction = knn_predict(training_data, training_labels, test_point, k)
# # print(prediction)


# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_digits
# SEED = 23
# X, y = load_digits(return_X_y=True)
# train_X, test_X, train_y, test_y = train_test_split(X, y, 
# test_size = 0.25, 
# random_state = SEED)
# gbc = GradientBoostingClassifier(n_estimators=300,
# learning_rate=0.05,
# random_state=100,
# max_features=5 )
# gbc.fit(train_X, train_y)
# pred_y = gbc.predict(test_X)
# acc = accuracy_score(test_y, pred_y)
# print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.datasets import load_diabetes
# SEED = 23
# X, y = load_diabetes(return_X_y=True)
# train_X, test_X, train_y, test_y = train_test_split(X, y, 
# test_size = 0.25, 
# random_state = SEED)
# gbr = GradientBoostingRegressor(loss='absolute_error',
# learning_rate=0.1,
# n_estimators=300,
# max_depth = 1, 
# random_state = SEED,
# max_features = 5)
# gbr.fit(train_X, train_y)
# pred_y = gbr.predict(test_X)
# test_rmse = mean_squared_error(test_y, pred_y) ** (1 / 2)
# print('Root mean Square error: {:.2f}'.format(test_rmse))


# from sklearn.datasets import make_moons
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Create synthetic 2D data
# X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# # Create a DataFrame for plotting
# df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
# df["Target"] = y

# # Visualize the 2D data
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     data=df,
#     x="Feature 1",
#     y="Feature 2",
#     hue="Target",
#     palette="Set1",
#     s=60,              # point size
#     edgecolor="k"      # black edge for clarity
# )
# plt.title("2D Classification Data (make_moons)", fontsize=14)
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.legend(title="Class", loc="upper right")
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.show()


# ============================================
# Machine Learning Experiments: k-NN on make_moons
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay

# ------------------------------------------------
# 1. Generate synthetic dataset
# ------------------------------------------------
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# Put into DataFrame for visualization
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Target"] = y

# Initial scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Feature 1",
    y="Feature 2",
    hue="Target",
    palette="Set1",
    s=60,
    edgecolor="k"
)
plt.title("2D Classification Data (make_moons)", fontsize=14)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Class", loc="upper right")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("make_moons_dataset.png", dpi=300)
plt.show()

# ------------------------------------------------
# 2. Preprocess: scale features
# ------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------------------------
# 3. Train k-NN (k=5) and evaluate
# ------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Test Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

# ------------------------------------------------
# 4. Cross-validation to find best k
# ------------------------------------------------
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_range, cv_scores, marker='o')
plt.title("k-NN Cross-Validation Accuracy vs k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validated Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_cv_accuracy.png", dpi=300)
plt.show()

best_k = k_range[np.argmax(cv_scores)]
print(f"Best k from cross-validation: {best_k}")

# ------------------------------------------------
# 5. Visualize decision boundary for best k
# ------------------------------------------------
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

disp = DecisionBoundaryDisplay.from_estimator(
    best_knn,
    X_scaled,
    response_method="predict",
    cmap="Pastel1",
    alpha=0.5
)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor="k", s=30)
plt.title(f"k-NN Decision Boundary (k={best_k})", fontsize=14)
plt.tight_layout()
plt.savefig("knn_decision_boundary.png", dpi=300)
plt.show()