import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the data
with open('Spam.json') as f:
    spam = json.load(f)
with open('NotSpam.json') as f:
    not_spam = json.load(f)
with open('sampleEmails.json') as f:
    sample_emails = json.load(f)

# Prepare the data
emails = spam + not_spam
labels = [1]*len(spam) + [0]*len(not_spam)

# Vectorize the data
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(emails)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# Train the SVM model
model = svm.SVC()
model.fit(X_train, y_train)

# Fine-tuning the model
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Evaluate the model
sample_features = vectorizer.transform(sample_emails)
y_pred = grid.predict(sample_features)
print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy: ', accuracy_score(y_test, y_pred))

# Process new email
with open('newEmail.txt', 'r') as f:
    new_email = f.read()
new_features = vectorizer.transform([new_email])
new_pred = grid.predict(new_features)

if new_pred[0] == 1:
    print('The new email is spam.')
else:
    print('The new email is not spam.')