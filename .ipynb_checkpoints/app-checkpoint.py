import pickle
import json
from flask import Flask, request, jsonify, render_template
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from the .pkl file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the columns expected in the input data
columns = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last',
           'gender', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num',
           'unix_time', 'merch_lat', 'merch_long']

# Route for homepage with options to upload a JSON or enter custom values
@app.route('/')
def home():
    return render_template('index.html', columns=columns)

# Route for handling file upload and making prediction
@app.route('/predict_json', methods=['POST'])
def predict_json():
    if 'file' not in request.files:
        return "No file part in the request", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Check if the uploaded file is a JSON file
    if not file.filename.endswith('.json'):
        return "ERROR: NOT A JSON FILE. <br> Please select a JSON file.", 400

    try:
        # Parse JSON file
        data = json.load(file)
    except json.JSONDecodeError:
        return "ERROR: Invalid JSON file. <br> Please check the file format.", 400
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data], columns=columns)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return jsonify({'prediction': int(prediction)})

# Route for handling manual input of data
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    # Collect form data
    input_data = {column: request.form[column] for column in columns}
    
    # Convert input_data into DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Convert numeric columns to correct data type
    numeric_columns = ['Unnamed: 0', 'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    input_df[numeric_columns] = input_df[numeric_columns].apply(pd.to_numeric)

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
