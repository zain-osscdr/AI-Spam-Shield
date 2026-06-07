import gradio as gr
import re
import joblib
import nltk


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



nltk.download("stopwords")
nltk.download("wordnet")

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Example Messages

SPAM_EXAMPLE = """
Congratulations! You have won a FREE iPhone.
Click here to claim your reward.
"""

OFFER_EXAMPLE = """
Limited Time Offer!

Buy now and save 50%.

Click here to claim your discount.
"""

NORMAL_EXAMPLE = """
Hi Team,

Please find the meeting agenda attached.

Regards
"""

# Prediction Function
def predict(message):

    if len(message.strip()) == 0:
        return {"⚠️ Empty Message": 0.0}, "0.00%"

    text = message.lower()

    text = re.sub(r'[^a-z\s]', ' ', text)

    words = text.split()

    stop_words = set(stopwords.words("english"))

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    final_text = " ".join(words)

    vector = tfidf.transform([final_text])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    if prediction == 1:

        confidence = probabilities[1]

        return {
            "🚨 Spam": float(confidence)
        }, f"{confidence * 100:.2f}%"

    else:

        confidence = probabilities[0]

        return {
            "✅ Ham (Not Spam)": float(confidence)
        }, f"{confidence * 100:.2f}%"

# Example Functions

def load_spam():
    return SPAM_EXAMPLE

def load_offer():
    return OFFER_EXAMPLE

def load_normal():
    return NORMAL_EXAMPLE

# Interface

with gr.Blocks(
    title="AI Spam Shield",
    css_paths="style.css"
) as demo:

    # Theme Toggle
    gr.HTML("""
<div class="theme-toggle-wrap">
    <button class="theme-btn" onclick="
        document.body.classList.toggle('light-mode');
        this.textContent = document.body.classList.contains('light-mode')
            ? '🌙 Dark Mode'
            : '☀️ Light Mode';
    ">☀️ Light Mode</button>
</div>
""")

    # Header
    gr.HTML("""
<div class="contain">
<div class="hero-card">

    <div class="hero-top-bar">
        <div class="hero-logo-area">
            <div class="hero-logo-icon">🛡</div>
            <span class="hero-logo-text">AI Spam Shield</span>
        </div>
        <div class="hero-nav-badges">
            <div class="nav-badge">🛡 Spam Detection</div>
            <div class="nav-badge">⚡ Fast Prediction</div>
            <div class="nav-badge">🔒 User Friendly</div>
        </div>
    </div>

    <div class="hero-body">
        <div class="hero-text">
            <h1>AI Spam Shield</h1>
            <h3>Smart Email Spam Detection System</h3>
            <p>Protect your inbox from unwanted messages.</p>
            <p>Classify messages as Spam or Ham with confidence scores.</p>
        </div>
        <div class="hero-image-area">
            <img src="assets/artificial-intelligence-14089_64.png" alt="Shield" />
        </div>
    </div>

</div>
</div>
""")

    # Main content
    with gr.Column(elem_classes="contain"):

        # Input Section
        gr.Markdown("## 📧 Paste Your Message")

        message = gr.Textbox(
            lines=12,
            label="Message Content",
            placeholder="Paste your message here..."
        )

        btn = gr.Button("🚀 Predict Now")

        gr.Markdown("""
### Supported Message Types

- SMS Messages

- Email Messages

- Promotional Messages

- Business Communication
""")

        # Result Section
        gr.Markdown("## 🎯 Prediction Result")

        with gr.Group(elem_classes="result-card"):

            prediction = gr.Label(
                label="Prediction Result"
            )

            confidence = gr.Textbox(
                label="Confidence Score",
                interactive=False
            )

        btn.click(
            fn=predict,
            inputs=message,
            outputs=[prediction, confidence]
        )

        # Example Messages
        gr.Markdown("## 🧪 Try Example Messages")

        with gr.Row():

            spam_btn = gr.Button("🚨 Spam Example")
            offer_btn = gr.Button("💰 Offer Example")
            normal_btn = gr.Button("📩 Normal Email")

        spam_btn.click(fn=load_spam, outputs=message)
        offer_btn.click(fn=load_offer, outputs=message)
        normal_btn.click(fn=load_normal, outputs=message)

demo.launch()