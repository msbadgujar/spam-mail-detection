import streamlit as st     # Streamlit library se web app banate hain
import pickle              # Pickle module se saved model aur vectorizer ko load karte hain

# "spam_classifier.pkl" file se trained model (Logistic Regression classifier) ko load karte hain.
with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# "vectorizer.pkl" file se trained TF-IDF vectorizer ko load karte hain.
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit page ke configuration set karte hain: title aur layout
st.set_page_config(page_title="Smart Spam Mail Detector", layout="centered")

# Page ka title aur description display karte hain
st.title("📧 Smart Spam Mail Detection System")
st.markdown("Enter any message below and this tool will predict whether it's **SPAM** or **NOT SPAM** with confidence.")

# Text area jahan user apna message type karta hai
input_text = st.text_area("✉️ Enter your message here:", height=150)

# "Detect Spam" button dabane pe evaluation start hota hai
if st.button("🧠 Detect Spam"):
    if input_text.strip() == "":  # Agar input field blank hai to warning de
        st.warning("⚠️ Please enter a message to detect.")
    else:
        # Input text ko vectorizer ke through numerical format (vector) me convert karte hain.
        input_vector = vectorizer.transform([input_text])
        # Model se prediction lete hain: 1 matlab spam, 0 matlab ham (non-spam)
        prediction = model.predict(input_vector)[0]
        # predict_proba ke through confidence score bhi nikalte hain.
        probability = model.predict_proba(input_vector)[0][prediction]

        # Agar prediction 1 hai to spam message, aur confidence score display karte hain
        if prediction == 1:
            st.error(f"🧨 This message is **SPAM**! ({probability * 100:.2f}% confidence)")
        else:
            st.success(f"✅ This message is **NOT SPAM**. ({probability * 100:.2f}% confidence)")
