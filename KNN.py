import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('train.csv')

# add to every na something
data['HomePlanet'].fillna(0, inplace=True)
data['CryoSleep'].fillna(False, inplace=True)
data['Destination'].fillna('TRAPPIST-1e', inplace=True)
data['VIP'].fillna(0, inplace=True)

# Convert string columns to float
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype('category').cat.codes.astype('float64')

# Fill in missing values with mean
data.fillna(data.mean(), inplace=True)

# Extract the target variable
y = data['Transported']

# Keep all columns
X = data.drop(['Transported', 'PassengerId'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN model and fit it to the training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Load the test data
test_data = pd.read_csv('test.csv')

# Keep only the PassengerId column
passenger_ids = test_data['PassengerId']

# Convert string columns to float
for col in test_data.columns:
    if test_data[col].dtype == 'object':
        test_data[col] = test_data[col].astype('category').cat.codes.astype('float64')

# Fill in missing values with mean
test_data.fillna(test_data.mean(), inplace=True)

# Predict on the test data
X_test = test_data.drop('PassengerId', axis=1)
y_pred = knn.predict(X_test)

# Create a new DataFrame with PassengerId and predicted Transported values
output = pd.DataFrame({'PassengerId': passenger_ids, 'Transported': y_pred})

# Save the output to a new CSV file
output.to_csv('submission.csv', index=False)
