import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

# Define the URL of the dataset and column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
column_names = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age', 'compressive_strength']

# Load the dataset from the URL
df = pd.read_excel(url, names=column_names)

# Split the dataset into features (X) and target (y) for both problems
X_regression = df.drop(['compressive_strength'], axis=1)
y_regression = df['compressive_strength']
y_classification = (y_regression >= y_regression.median()).astype(int)

# Split the datasets into train and test sets
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_regression, y_classification, test_size=0.2, random_state=42)

# Standardize the data for both problems
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

# Regression problem: Two-layer Neural Network with 2 hidden units with 50 neurons each
reg = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
reg.fit(X_reg_train_scaled, y_reg_train)
y_reg_pred = reg.predict(X_reg_test_scaled)
mse = mean_squared_error(y_reg_test, y_reg_pred)
print("Regression Mean Squared Error:", mse)
# print("The input is: ", X_reg_test_scaled)
print("For input the regression prediction is", y_reg_pred[0])
print("\n")

# Classification problem: Two-layer Neural Network with 2 hidden units with 50 neurons each
clf = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
clf.fit(X_class_train_scaled, y_class_train)
y_class_pred = clf.predict(X_class_test_scaled)
accuracy = accuracy_score(y_class_test, y_class_pred)
print("Classification Accuracy:", accuracy)
# print("The input is: ", X_class_test_scaled)
print("For input the classification is predicted as", "good" if y_class_pred[0] else "bad")
