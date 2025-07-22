from flask import Flask, request, render_template_string
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the FAQ data from Excel
faq_df = pd.read_excel("FAQ_Dataset.xlsx")
faq_df.dropna(subset=["Question", "Answer"], inplace=True)
faq_df["Question"] = faq_df["Question"].astype(str).str.strip()
faq_df["Answer"] = faq_df["Answer"].astype(str).str.strip()

# Encode all the FAQ questions for semantic search
faq_embeddings = model.encode(faq_df["Question"], convert_to_tensor=True)

# Flask app setup
app = Flask(__name__)

# HTML template for basic chatbot UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>HR FAQ Chatbot</title></head>
<body>
  <h2>HR FAQ Chatbot</h2>
  <form method="POST">
    <label>Ask your question:</label><br>
    <input type="text" name="question" size="80"><br><br>
    <input type="submit" value="Submit">
  </form>
  {% if response %}
    <h3>Response:</h3>
    <p>{{ response }}</p>
  {% endif %}
</body>
</html>
"""

# Route for home
@app.route("/", methods=["GET", "POST"])
def home():
    response = None
    if request.method == "POST":
        user_question = request.form["question"].strip()
        if user_question:
            query_embedding = model.encode(user_question, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)[0]
            top_index = similarity_scores.argmax().item()
            best_score = similarity_scores[top_index].item()

            if best_score >= 0.6:  # Adjust this threshold if needed
                response = faq_df.iloc[top_index]["Answer"]
            else:
                response = "Sorry, I couldn't find a relevant answer. Please contact HR."

    return render_template_string(HTML_TEMPLATE, response=response)

if __name__ == "__main__":
    app.run(debug=True)
