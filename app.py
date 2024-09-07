import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and encoder
model = pickle.load(open('best_model.pkl', 'rb'))
onehotEnc = pickle.load(open('onehotEnc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the request
        age = float(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']
        
        # Prepare the input data
        cat_data = [[sex, smoker, region]]  # Encapsulate in list for correct shape
        num_data = [[age, bmi, children]]  # Encapsulate in list for correct shape
        
        # Convert categorical data to DataFrame
        cat_df = pd.DataFrame(cat_data, columns=['sex', 'smoker', 'region'])
        num_df = pd.DataFrame(num_data, columns=['age', 'bmi', 'children'])
        
        # Apply one-hot encoding to categorical features
        cat_encoded = onehotEnc.transform(cat_df)
        print(cat_encoded)
        
        # Combine numerical and one-hot encoded features
        final_input = np.hstack([num_df.values, cat_encoded])
        print(final_input)
        
        # Make prediction
        output = model.predict(final_input)
        print(output)
        
        # Return the result
        return render_template("prediction.html", prediction_text=f"The premium prediction is: Rs.{output[0]:.2f}")
    
    except Exception as e:
        return render_template("prediction.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
