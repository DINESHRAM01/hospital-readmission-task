from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    try:
        features = [
            float(request.form['age']),
            float(request.form['time_in_hospital']),
            float(request.form['n_lab_procedures']),
            float(request.form['n_procedures']),
            float(request.form['n_medications']),
            float(request.form['n_outpatient']),
            float(request.form['n_inpatient']),
            float(request.form['n_emergency']),
            

        ]
        final_features = np.array(features).reshape(1, -1)
        #prediction = model.predict(final_features)
        #return render_template('index.html', prediction_text=f"Readmission Prediction: {prediction[0]}")
        prediction = model.predict(final_features)[0]

        label = (
            "✅ Not Readmitted"
            if prediction == 0 else
                    "⚠️ Readmitted (Within 30 Days)"
                        )

        return render_template('index.html', prediction_text=label)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")
if __name__ == '__main__':
    app.run(debug=True)

