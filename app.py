from flask import Flask, render_template, request
import pickle
import numpy as np

# ========== 1. Initialize Flask ==========
app = Flask(__name__)

# ========== 2. Load Model ==========
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, le = pickle.load(f)

# ========== 3. Home Route ==========
@app.route('/')
def home():
    return render_template('index.html')

# ========== 4. Prediction Route ==========
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    gender = request.form['gender']
    married = request.form['married']
    dependents = request.form['dependents']
    education = request.form['education']
    self_employed = request.form['self_employed']
    applicant_income = float(request.form['applicant_income'])
    coapplicant_income = float(request.form['coapplicant_income'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    credit_history = float(request.form['credit_history'])
    property_area = request.form['property_area']

    # Manual Encoding
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    education_map = {"Graduate": 1, "Not Graduate": 0}
    self_emp_map = {"Yes": 1, "No": 0}
    property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    input_data = [
        gender_map[gender],
        married_map[married],
        dependents_map[dependents],
        education_map[education],
        self_emp_map[self_employed],
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_map[property_area]
    ]

    # Convert to array and scale
    X_input = np.array(input_data).reshape(1, -1)
    X_input_scaled = scaler.transform(X_input)

    # Prediction
    prediction = model.predict(X_input_scaled)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Not Approved"

    return render_template('index.html', prediction_text=result)

# ========== 5. Run App ==========
if __name__ == '__main__':
    app.run(debug=True)
