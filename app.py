from flask import Flask, request, jsonify
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("bmi_model.pkl")

@app.route('/api/bmi', methods=['POST'])
def BMI():
    try:
        # Check if form data is present
        if not all(key in request.form for key in ('Gender', 'Height', 'Weight')):
            return jsonify({'error': 'Missing data. Please provide Gender, Height, and Weight.'}), 400
        
        # Get form data
        Gender = int(request.form.get('Gender'))
        Height = int(request.form.get('Height'))
        Weight = int(request.form.get('Weight'))
        
        # Validate input data
        if Gender not in [0, 1]:
            return jsonify({'error': 'Gender must be 0 (female) or 1 (male).'}), 400
        if Height <= 0 or Weight <= 0:
            return jsonify({'error': 'Height and Weight must be positive numbers.'}), 400

        # Prepare the input for the model
        x = np.array([[Gender, Height, Weight]])

        # Predict using the model
        prediction = model.predict(x)

        # Return the corresponding index based on the prediction
        bmi_index = int(prediction[0])
        index_labels = {
            0: 'อ่อนแอมาก',
            1: 'อ่อนแอ',
            2: 'ปกติ',
            3: 'น้ำหนักเกิน',
            4: 'โรคอ้วน',
            5: 'โรคอ้วนรุนแรง'
        }

        return jsonify({'Index': index_labels.get(bmi_index, 'Unknown BMI category')})

    except ValueError:
        return jsonify({'error': 'Invalid input. Please ensure all inputs are numeric.'}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
