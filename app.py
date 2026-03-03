from flask import Flask, render_template, request
from model_loader import load_datasets, recommend_items
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# -----------------------------------------------------------------------------------
# ✅ Load the model and datasets ONLY ONCE when the server starts (Not on every request)
# -----------------------------------------------------------------------------------
print("\n⏳ Loading model and datasets... Please wait.\n")
model = SentenceTransformer('all-MiniLM-L6-v2')   # Load model once
courses, youtube, books, skills = load_datasets()  # Load dataset once
print("\n✅ Model and datasets loaded successfully! Server ready.\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    skill = request.form.get('skill', '').strip()

    if not skill:
        return render_template('results.html', error="⚠ Please enter a skill to get recommendations.")

    # Call recommend function with pre-loaded data & model
    recommendations = recommend_items(skill, courses, youtube, books, skills, model)

    return render_template('results.html', skill=skill, **recommendations)

if __name__ == "__main__":
    app.run(debug=True)
