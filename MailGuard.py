# Build a Naive Bayes Classifier to predict whether an email is spam or not spam based on its content.
# The model will analyze the frequency of words and phrases in the email text to learn patterns commonly found in spam messages. 
# Using a labeled email dataset, train and test the model to evaluate its accuracy, precision, and recall in identifying spam emails.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt

# Load dataset (adjusted for the Kaggle version)
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['Category', 'Message']]
data.columns = ['Category', 'Text']

# Encode labels
data['Label'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['Text'], data['Label'], test_size=0.2, random_state=0)

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Predictions
y_pred = nb.predict(X_test_tfidf)

# Evaluation
accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred), 3)
recall = round(recall_score(y_test, y_pred), 3)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Try a custom email
sample = ["Congratulations! You've won a free trip to Dubai. Click below to claim."]
sample_vec = tfidf.transform(sample)
sample_pred = "Spam" if nb.predict(sample_vec)[0] == 1 else "Not Spam"
print("\nCustom Email Prediction:", sample_pred)

# ----------- Visualization Section -----------

metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]

plt.figure(figsize=(7,5))
plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
plt.title('Model Performance Metrics', pad=15)
plt.tight_layout()
plt.show()


plt.figure(figsize=(4, 3))
color = 'red' if sample_pred == 'Spam' else 'green'
plt.bar(['Custom Email'], [1], color=color)
plt.title(f"Custom Email Prediction: {sample_pred}")
plt.ylim(0, 1.5)
plt.text(0, 1.05, sample_pred, ha='center', fontweight='bold')
plt.show()
