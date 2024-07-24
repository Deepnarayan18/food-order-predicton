from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        a = int(request.form['age'])
        b = int(request.form['gender'])
        c = int(request.form['marital_status'])
        d = int(request.form['occupation'])
        e = int(request.form['income'])
        f = int(request.form['education'])
        g = int(request.form['family_size'])
        h = int(request.form['pin_code'])
        i = int(request.form['review'])
        
        # Create feature array
        features = np.array([[a, b, c, d, e, f, g, h, i]])
        
        # Make prediction
        prediction = model.predict(features)
        
        return render_template('index.html', prediction_text='Customer will order again: {}'.format(prediction[0]))
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
