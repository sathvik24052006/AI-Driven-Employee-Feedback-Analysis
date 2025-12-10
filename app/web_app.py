from flask import Flask, render_template, request, jsonify
from predict_service import predict_feedback
import os

# Tell Flask where templates folder is
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../templates"))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/analyze", methods=["POST"])
def analyze_text():
    feedback_text = request.form.get("feedback", "").strip()

    if not feedback_text:
        return render_template("index.html", prediction=None, error="Please enter feedback text.")

    result = predict_feedback(feedback_text)

    return render_template("index.html", prediction=result, input_text=feedback_text)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    if not data or "feedback" not in data:
        return jsonify({"error": "JSON must include 'feedback' field."}), 400

    result = predict_feedback(data["feedback"])
    return jsonify(result)


if __name__ == "__main__":
    print("\n🚀 Flask app running at: http://127.0.0.1:5000\n")
    app.run(debug=True)

