import streamlit as st
import joblib
import os

# Paths for the saved model and vectorizer
MODEL_PATH = "naive_bayes_spam_model.joblib"
VECTORIZER_PATH = "count_vectorizer.joblib"

# Set the browser title and favicon
st.set_page_config(
    page_title="Naive Bayes Spam Email Detection App",  # Browser tab title
    page_icon="📧",  # Favicon (emoji or path to an image file)
    layout="centered",  # Layout can be "centered" or "wide"
)

# Function to load the model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    # Check if files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        st.error("❌ Required model files are missing! Please upload the correct files.")
        return None, None
    # Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

# Load the model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Streamlit app title and description
st.title("🔍 Spam Email Detection App")
st.subheader("📧 Analyze your email messages to detect if they're **Spam** or **Ham**.")
st.write("This app uses a Naive Bayes classifier to predict the likelihood of spam.")

# Input text box
user_input = st.text_area("💬 Type your email message here:")

# Prediction functionality
if st.button("Check Message"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    elif model is None or vectorizer is None:
        st.error("❌ Model is not properly loaded. Please check your setup.")
    else:
        # Transform the input text
        input_vectorized = vectorizer.transform([user_input])

        # Make predictions
        prediction = model.predict(input_vectorized)[0]  # 0 for Ham, 1 for Spam
        prediction_proba = model.predict_proba(input_vectorized)[0]  # Probabilities

        # Display results
        if prediction == 1:  # Spam
            st.error(f"❌ The message is classified as **Spam**.")
            st.write(f"📊 Spam probability: **{prediction_proba[1]:.2f}**")
        else:  # Ham
            st.success(f"✅ The message is classified as **Ham**.")
            st.write(f"📊 Ham probability: **{prediction_proba[0]:.2f}**")

# Footer with credits
st.markdown("---")
st.markdown("💡 Built with Streamlit by Forhad Hossain | Naive Bayes Spam Detection App")