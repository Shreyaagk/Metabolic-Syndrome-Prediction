from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset and train the model
df = pd.read_csv(r"C:\Users\shrey\OneDrive\Desktop\OJT\Metabolic Syndrome.csv")
df.ffill(inplace=True)  # Fill missing values
df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})

# Encode categorical features
le_Marital = LabelEncoder()
le_Race = LabelEncoder()
df['Marital'] = le_Marital.fit_transform(df['Marital'])
df['Race'] = le_Race.fit_transform(df['Race'])

# Drop unnecessary columns
df.drop(['seqn', 'Income'], axis=1, inplace=True)

# Define features and target variable
X = df.drop("MetabolicSyndrome", axis=1)
y = df["MetabolicSyndrome"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Save trained model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(rfc, model_file)

# Load model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and convert form data to float
        feature_names = ["Age", "Sex", "Marital", "Race", "WaistCirc", "BMI", "Albuminuria",
                         "UrAlbCr", "UricAcid", "BloodGlucose", "HDL", "Triglycerides"]

        features = [float(request.form.get(name, 0)) for name in feature_names]

        # Ensure input shape matches model expectations
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Interpret result
        result = "Person has Metabolic Syndrome" if prediction == 1 else "Person does not have Metabolic Syndrome"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False when deploying
