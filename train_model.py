import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Dataset
df = pd.read_csv('mescoe_dataset.csv')

# 2. Expand comma-separated text prompts into individual training samples
expanded_rows = []
for idx, row in df.iterrows():
    phrases = str(row['text']).split(',')
    for phrase in phrases:
        cleaned_phrase = phrase.strip().lower()
        if cleaned_phrase:
            expanded_rows.append({
                'text': cleaned_phrase,
                'intent': row['intent'],
                'response': row['response']
            })

train_df = pd.DataFrame(expanded_rows)

# Save cleaned expanded dataset for Streamlit mapping
train_df.to_csv('mescoe_dataset_expanded.csv', index=False)

# 3. Vectorize text with Sublinear Scaling & N-Grams (1 to 3) to prevent overfitting
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3), 
    sublinear_tf=True, 
    analyzer='word',
    strip_accents='unicode',
    lowercase=True
)
X = vectorizer.fit_transform(train_df['text'])
y = train_df['intent']

# 4. Train Logistic Regression & Multinomial Naive Bayes
lr_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
lr_model.fit(X, y)

nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(X, y)

# 5. Evaluate Models
lr_preds = lr_model.predict(X)
nb_preds = nb_model.predict(X)

metrics = [
    {
        'Model': 'Logistic Regression',
        'Accuracy': round(accuracy_score(y, lr_preds), 4),
        'Precision': round(precision_score(y, lr_preds, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y, lr_preds, average='weighted', zero_division=0), 4),
        'F1-Score': round(f1_score(y, lr_preds, average='weighted', zero_division=0), 4)
    },
    {
        'Model': 'Multinomial Naive Bayes',
        'Accuracy': round(accuracy_score(y, nb_preds), 4),
        'Precision': round(precision_score(y, nb_preds, average='weighted', zero_division=0), 4),
        'Recall': round(recall_score(y, nb_preds, average='weighted', zero_division=0), 4),
        'F1-Score': round(f1_score(y, nb_preds, average='weighted', zero_division=0), 4)
    }
]

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('model_metrics.csv', index=False)

# 6. Save the best model (Logistic Regression) and vectorizer
joblib.dump(lr_model, 'mescoe_chatbot_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model retrained and artifacts saved successfully!")
print(metrics_df)
