from flask import Flask, render_template, request
import pickle

model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    study = float(request.form["study"])
    social = float(request.form["social"])
    sleep = float(request.form["sleep"])
    attendance = float(request.form["attendance"])
    assignment = float(request.form["assignment"])

    result = model.predict([[study, social, sleep, attendance, assignment]])

    return render_template("index.html", prediction=round(result[0],2))

if __name__ == "__main__":
    app.run(debug=True)