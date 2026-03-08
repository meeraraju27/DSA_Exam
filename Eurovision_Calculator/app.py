from flask import Flask, request, render_template
import pickle  
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD MODELS/FILES HERE ---
with open('eurovision_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)


@app.route('/')
def home():
    """Renders the main page (index.html)."""
    # Pass an empty string so the template doesn't error on first load
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and returns it."""
    
    try:
        # 1. Get data from form
        english = int(request.form['Song.In.English'])
        danceability = float(request.form['danceability']) 
        energy = float(request.form['energy'])
        gender = request.form['Artist.gender']
        group_solo = request.form['Group.Solo']

        # 2. Create input vector matching training
        input_data = {name: 0.5 for name in feature_names}
        input_data['Song.In.English'] = english
        input_data['danceability'] = danceability
        input_data['energy'] = energy
        
        # One-hot categoricals
        for name in feature_names:
            if 'Artist.gender_' + gender in name:
                input_data[name] = 1.0
            if 'Group.Solo_' + group_solo in name:
                input_data[name] = 1.0

                # Predict
        # Predict
        input_df = pd.DataFrame([input_data])[feature_names]
        output = model.predict(input_df)[0]
        prediction_text = f"Predicted Points: {output}"

    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    # --- 3. Render the page again with the prediction ---
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)