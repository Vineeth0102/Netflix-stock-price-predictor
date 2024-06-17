from flask import Flask, request, render_template
import pickle
import pandas as pd
app = Flask(__name__)

# Load the trained model
with open('NetflixPred.pkl', 'rb') as file:
    NetflixPred = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    opening_value = float(request.form['Opening_value'])
    low_value = float(request.form['Low_value'])
    highest_value = float(request.form['Highest_value'])
    volume = float(request.form['Volume'])

    prediction = predict_close(opening_value, highest_value, low_value, volume)
    return render_template('home.html', result=prediction)

def predict_close(Open, High, Low, Volume):
    temp_array = [Open, High, Low, Volume]
    feature_names = ['Open', 'High', 'Low', 'Volume']
    temp_df = pd.DataFrame([temp_array], columns=feature_names)
    return NetflixPred.predict(temp_df)[0].round(2)

if __name__ == '__main__':
    app.run(debug=True)
