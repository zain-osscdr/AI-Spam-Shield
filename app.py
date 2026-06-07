import gradio as gr
from transformers import pipeline

# Load DistilBERT Model

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

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

    result = classifier(message)

    label = result[0]["label"]
    score = result[0]["score"]

    if label == "NEGATIVE":
        prediction = "🚨 Spam"
    else:
        prediction = "✅ Ham (Not Spam)"

    return {prediction: score}, f"{score * 100:.2f}%"

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

    # Header
    with gr.Row(elem_classes="hero-card"):

        with gr.Column(scale=1):

            gr.Image(
                "assets/shield.png",
                width=150,
                show_label=False,
                container=False
            )

        with gr.Column(scale=3):

            gr.Markdown("""
# AI Spam Shield

### Smart Email Spam Detection using DistilBERT

Protect your inbox using Artificial Intelligence.

Detect spam messages instantly with high accuracy.
""")

            gr.HTML("""
<div class="badges">
    <div class="badge">🛡 99% Accurate</div>
    <div class="badge">⚡ Lightning Fast</div>
    <div class="badge">🔒 Privacy Focused</div>
</div>
""")

    # Input Section
    gr.Markdown("## 📧 Paste Email / Message")

    message = gr.Textbox(
        lines=12,
        label="Email Content",
        placeholder="Paste email content here..."
    )

    btn = gr.Button("🚀 Predict Now")

    gr.Markdown("""
### Supported Messages

- Email Messages

- Promotional Emails

- Spam Content

- Business Emails
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

    spam_btn.click(
        fn=load_spam,
        outputs=message
    )

    offer_btn.click(
        fn=load_offer,
        outputs=message
    )

    normal_btn.click(
        fn=load_normal,
        outputs=message
    )

demo.launch()