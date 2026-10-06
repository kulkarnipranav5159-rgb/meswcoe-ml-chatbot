import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Load Dataset
df = pd.read_csv('mescoe_dataset.csv')

# 2. Preprocessing & Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['intent']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 3. Model 1: Multinomial Naive Bayes
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# 4. Model 2: Logistic Regression
model_lr = LogisticRegression(max_iter=200)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# 5. Evaluate Performance Metrics
acc_nb = accuracy_score(y_test, y_pred_nb)
acc_lr = accuracy_score(y_test, y_pred_lr)

prec_nb, rec_nb, f1_nb, _ = precision_recall_fscore_support(y_test, y_pred_nb, average='weighted', zero_division=0)
prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted', zero_division=0)

# Create metrics comparison table
metrics_df = pd.DataFrame({
    'Model': ['Multinomial Naive Bayes', 'Logistic Regression'],
    'Accuracy': [acc_nb, acc_lr],
    'Precision': [prec_nb, prec_lr],
    'Recall': [rec_nb, rec_lr],
    'F1-Score': [f1_nb, f1_lr]
})

print("=== Model Performance Comparison ===")
print(metrics_df)

# 6. Save the Trained Model & Vectorizer
best_model = model_lr if acc_lr >= acc_nb else model_nb
joblib.dump(best_model, 'mescoe_chatbot_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Save metrics CSV for the Streamlit App
metrics_df.to_csv('model_metrics.csv', index=False)
print("\nModel saved successfully as 'mescoe_chatbot_model.pkl'!")