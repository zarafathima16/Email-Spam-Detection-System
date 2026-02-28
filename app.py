from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Sample training data
emails = [
    "Congratulations you won a lottery",
    "Claim your free prize now",
    "Limited time offer buy now",
    "Win money instantly click here",
    "Meeting scheduled tomorrow",
    "Project discussion at 5pm",
    "Let's complete the assignment",
    "Important office update"
]

labels = [1,1,1,1,0,0,0,0]  # 1 = Spam, 0 = Not Spam

# Train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

model = MultinomialNB()
model.fit(X, labels)

# HTML UI
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Email Spam Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1f4037, #99f2c8);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .card {
            background: white;
            padding: 40px;
            border-radius: 15px;
            width: 400px;
            text-align: center;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        }

        h1 {
            margin-bottom: 20px;
            color: #1f4037;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ccc;
            margin-bottom: 15px;
            font-size: 14px;
        }

        button {
            padding: 10px 20px;
            background: #1f4037;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            background: #14532d;
        }

        .result {
            margin-top: 20px;
            font-size: 18px;
            font-weight: bold;
        }

        .spam {
            color: red;
        }

        .notspam {
            color: green;
        }
    </style>
</head>
<body>

<div class="card">
    <h1>📧 AI Spam Detector</h1>

    <form method="POST">
        <textarea name="message" placeholder="Enter your email content here..." required></textarea>
        <br>
        <button type="submit">Analyze Email</button>
    </form>

    {% if prediction %}
        <div class="result {{ css_class }}">
            {{ prediction }}
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    css_class = ""

    if request.method == "POST":
        message = request.form["message"]
        vector_input = vectorizer.transform([message])

        # ✅ Prediction with probability
        prediction_value = model.predict(vector_input)[0]
        probability = model.predict_proba(vector_input)[0]

        spam_percent = round(probability[1] * 100, 2)
        ham_percent = round(probability[0] * 100, 2)

        if prediction_value == 1:
            prediction = f"🚫 Spam Email ({spam_percent}% confident)"
            css_class = "spam"
        else:
            prediction = f"✅ Not Spam Email ({ham_percent}% confident)"
            css_class = "notspam"

    return render_template_string(html_page, prediction=prediction, css_class=css_class)

if __name__ == "__main__":
    app.run(debug=True)
