import numpy as np
import sklearn.model_selection as test_train_split

np.random.seed(42)  # random seed makes sure the random values generated are the same every time the code is run

n_samples = 100

a = np.random.randint(0, 1000, n_samples) # Creates an array of 100 random integers between 0 and 1000
b = np.random.randint(0, 1000, n_samples) # Creates an array of 100 random integers between 0 and 1000
c = np.random.randint(0, 1000, n_samples) # Creates an array of 100 random integers between 0 and 1000


X = np.column_stack([a, b, c])
# print(X)
Y = a + 2*b + 3*c


X_train, X_test, Y_train, Y_test = test_train_split.train_test_split(X, Y, test_size=0.2, random_state=42)
# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)

# Machine Learning Begins
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression(n_jobs=-1) # n_jobs=-1 means use all available cores for training
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

predictions = np.column_stack([Y_test, Y_pred])
print(predictions)


print("Mean Squared Error (MSE):", mean_squared_error(Y_test, Y_pred))
print("R-squared (R2) Score:", r2_score(Y_test, Y_pred))

import matplotlib.pyplot as plt

X_test_feature = X_test[:, 0]
sorted_idx = np.argsort(X_test_feature)

plt.figure(figsize=(8, 5))

plt.scatter(X_test_feature, Y_test, color='blue', label='Actual Y')
plt.scatter(X_test_feature, Y_pred, color='red', label='Predicted Y', marker='x')

plt.plot(X_test_feature[sorted_idx], Y_pred[sorted_idx], color='green', label='Regression Line', linewidth=2)

plt.xlabel("Feature a")
plt.ylabel("Target Y")
plt.title("Linear Regression Model Visualization (Using Feature a)")
plt.legend()
plt.grid(True)
plt.show()
