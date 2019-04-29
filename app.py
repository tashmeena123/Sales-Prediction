from flask import Flask, render_template, jsonify, request
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
global lr

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST","GET"])
def predict():

    try:            
        query = np.array((float(request.form['Item_Identifier']),
                          float(request.form['Item_Weight']),
                          float(request.form['Item_MRP']),
                          int(request.form['Item_Fat_Content']),
                          float(request.form['Item_Visibility']),
                          int(request.form['Item_type'])))
        query = query.reshape(1, -1)
        print(query)
        print(query.shape)
        # query = np.array(query).reshape(-1,)
        prediction = (lr.predict(query))

        print(prediction)
        return render_template("predict.html",prediction=prediction)

    except:

    	return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) 
    except:
        port = 12345 

    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    
    app.run(port=port, debug=True)


 











