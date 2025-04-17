
import pandas as pd                                  # Pandas library data manipulation sathi upyog keli 
from sklearn.model_selection import train_test_split # Data la yraining ani testing set madhe split karnyasathi 
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF vectorizer text la numerical form madhe convert karto 
from sklearn.linear_model import LogisticRegression  # Logistic Regression classifier use karnya sathi 
import pickle                                        # Python objects ko file me save karne ke liye (serialization)
import re                                            # Regular expressions for text cleaning
import string                                        # String related utilities (punctuation, etc.)

# CSV file 'spam.csv' ko load karte , jyachyat spam aani ham messages ahet 
df = pd.read_csv("spam.csv")

# CSV file chya columns la rename karte ahe : 'label' (spam/ham) aur 'message'
df.columns = ['label', 'message']

# 'label' column madhe spam la 1 aur ham la 0 chya form mdhe ghet ahe 
df['label'] = df['label'].map({'ham': 0, 'spam': 1})0

# he function text la clean krta ahe 
# - sgdya characters la lowercase mdhe convert krt ahe
# - URLs, punctuation, digits, aur extra spaces hata deta hai.
def clean_text(text):
    text = text.lower()  # Text la lowercase mdhe convert karta hai
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs la remove krt ahe 
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Punctuation marks remove krt ahe 
    text = re.sub(r'\d+', '', text)  # Numbers  ko remove kart ahe 
    text = re.sub(r'\s+', ' ', text).strip()  # Extra spaces hata deta hai aur text ko trim karta hai
    return text

# "message" column ke sabhi messages ko clean_text function se process karke "clean_message" naam ka naya column banate hain
df['clean_message'] = df['message'].apply(clean_text)

# Data ko training aur testing set me 80:20 ratio me split karte hain, random_state se reproducibility maintain hoti hai
X_train, X_test, y_train, y_test = train_test_split(df['clean_message'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorizer banate hain, jo text ko numerical vectors me convert karta hai.
# ngram_range=(1, 2) se single words aur word pairs dono liye jaate hain.
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.9)
# Training set ke text data ko vectorizer se transform karte hain
X_train_vec = vectorizer.fit_transform(X_train)
# Testing set ke liye bhi vectorizer ko apply karte hain (fit nahi, sirf transform)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression model banate hain.
# C=2 is parameter se regularization ko control karte hain, max_iter=1000 iteration limit deta hai.
model = LogisticRegression(C=2, max_iter=1000)
# Model ko training set ke vectorized data aur corresponding labels (spam/ham) se train karte hain
model.fit(X_train_vec, y_train)

# Trained vectorizer ko 'vectorizer.pkl' file me save kar dete hain taki baad mein use ho sake.
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
# Trained classifier ko 'spam_classifier.pkl' file me save kar dete hain.
with open("spam_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

# Console par message print karta hai jab model training complete ho jaye.
print("✅ Model retrained with enhancements. Ready to predict smartly!")
