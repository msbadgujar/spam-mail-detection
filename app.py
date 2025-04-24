import streamlit as st    
import pickle            

with open("spam_classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
title aur layout
st.set_page_config(page_title="Smart Spam Mail Detector", layout="centered")

st.title("📧 Smart Spam Mail Detection System")
st.markdown("Enter any message below and this tool will predict whether it's **SPAM** or **NOT SPAM** with confidence.")

input_text = st.text_area("✉️ Enter your message here:", height=150)

if st.button("🧠 Detect Spam"):
    if input_text.strip() == "":  # Agar input field blank hai to warning de
        st.warning("⚠️ Please enter a message to detect.")
    else:
     
        input_vector = vectorizer.transform([input_text])
       
        prediction = model.predict(input_vector)[0]
  
        probability = model.predict_proba(input_vector)[0][prediction]

 
        if prediction == 1:
            st.error(f"🧨 This message is **SPAM**! ({probability * 100:.2f}% confidence)")
        else:
            st.success(f"✅ This message is **NOT SPAM**. ({probability * 100:.2f}% confidence)")
