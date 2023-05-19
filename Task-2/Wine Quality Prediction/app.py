from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import json
import pandas as pd

regmodel = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
  return render_template('home.html')
  

@app.route('/predict' , methods=['POST'])
def predict():
  # json_ = request.json
  # data = pd.DataFrame(json_)
  
  # output=regmodel.predict(data)
  # print(output)
  # output = str(output)
  # return jsonify({'prediction':output})
  
  float_features = [float(x) for x in request.form.values()]
  features = [np.array(float_features)]
  prediction = regmodel.predict(features)
  return render_template("home.html",prediction_text = "wine quality is {}".format(prediction))

if __name__ == " __main__":
  app.run(debug=True)  
