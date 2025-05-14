import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset f
df = pd.read_csv("fake_or_real_news.csv")

print('\n\t\t\t1)INFO OF DATA BEFORE PREPROCESSING:')
df.info()

###### PREPROCESSING #########

# 1. Clean the text column
df['text'] = df['text'].str.strip()
df = df[df['text'].notna() & (df['text'] != '')]  # Remove rows with empty text

# 2. Convert label to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})  # Convert to numerical

# 3. Remove duplicates
df = df.drop_duplicates(subset=['text'])  # Remove duplicate articles

# 4. create text length feature
df['text_length'] = df['text'].apply(len)

# 5. Remove outliers based on text length
# Calculate Z-scores for text length
z_scores = (df['text_length'] - df['text_length'].mean()) / df['text_length'].std()
df = df[(z_scores.abs() < 3)]  # Keep only rows within 3 standard deviations

# Print info of updated dataset after preprocessing
print('\n\n\t\t\t2)INFO OF DATA AFTER PREPROCESSING:')
df.info()

# convert text to numerical features (using TF-IDF or CountVectorizer)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_text = tfidf.fit_transform(df['text'])
X_length = df['text_length'].values.reshape(-1, 1)

# Combine features
import numpy as np
X = np.hstack((X_text.toarray(), X_length))
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluate models
print("\nKNN Performance:")
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\nDecision Tree Performance:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))