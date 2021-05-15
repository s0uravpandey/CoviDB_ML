import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('India/ind_svr_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    dataset = pd.read_csv('India/ind_vaccinations.csv')
    X = dataset.iloc[0:88,5].values
    y = dataset.iloc[0:112,3].values
    y=y.reshape(-1,1)
    X=X.reshape(-1,1)
    from sklearn.preprocessing import StandardScaler
    sc_y=StandardScaler()
    sc_X=StandardScaler()
    X=sc_X.fit_transform(X)
    y=sc_y.fit_transform(y)
    y_pred=sc_y.inverse_transform(model.predict(sc_X.transform(np.array([[int(x) for x in request.form.values()]]))))
    output=round(y_pred[0])
    return render_template('index.html',prediction_text=output,prediction_percent=round(output/13000000))
    

if __name__ == "__main__":
    app.run(debug=True)
