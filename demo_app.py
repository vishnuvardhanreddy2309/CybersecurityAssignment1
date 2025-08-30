import joblib
import gradio as gr

# Load saved model and vectorizer
model = joblib.load("fake_news_model.joblib")
tfidf = joblib.load("tfidf_vectorizer.joblib")

def fake_news_checker(text):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    return f"This news is likely: {pred.upper()}"

demo = gr.Interface(
    fn=fake_news_checker,
    inputs=gr.Textbox(lines=3, placeholder="Paste news headline or statement here..."),
    outputs="text",
    title="📰 Fake News Detection (TF-IDF + Logistic Regression)",
    description="Enter a news headline or short article statement to check if it's real or fake."
)

if __name__ == "__main__":
    demo.launch()
