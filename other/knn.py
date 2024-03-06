import psycopg2
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

# Establish a connection to the database
conn = psycopg2.connect(
    dbname="your_database",
    user="your_username",
    password="your_password",
    host="localhost",
    port="5432"
)

# Create a cursor object
cur = conn.cursor()

# Fetch training data from the database
cur.execute("SELECT features, label FROM training_data;")
training_data = cur.fetchall()

# Split the data into features (X) and label (y)
X = np.array([x[0] for x in training_data])
y = np.array([x[1] for x in training_data])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the SVM model
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

# Fine-tuning the model
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Predicting the test set results
y_pred = grid.predict(X_test)

# Evaluating the model
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix: ', cm)
print('Accuracy: ', accuracy_score(y_test, y_pred))

# Fetch new data from the database
cur.execute("SELECT features FROM new_data;")
new_data = cur.fetchall()

# Preprocess and predict the new data
new_data = scaler.transform(new_data)
new_pred = grid.predict(new_data)

print('Predictions for new data: ', new_pred)

# Close the cursor and connection
cur.close()
conn.close()