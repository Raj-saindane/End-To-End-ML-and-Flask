from flask import Flask,render_template,request
import pandas as pd
import pickle
import sklearn 
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

app=Flask(__name__)
model=pickle.load(open('a_model.pkl','rb'))

@app.route('/',methods=['GET'])
def home_fun():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    mpg=int(request.form['mpg'])
    cylinder=int(request.form['cylinder'])
    displacement=float(request.form['displacement'])

    data={'mpg':mpg,
          'cylinder':cylinder,
          'displacement':displacement}
    
    df=pd.DataFrame([data])
    prediction=model.predict(df)
    output=round(prediction[0],2)
    #if output<0:
        #return render_template('home.html',prediction_text="Due to wrong vlaue we cannot make Yor prediction")
    #else:
        #return render_template('home.html',prediction_text="here are your output{}".format(output))
    #else:
    #render_template("home.html")

if __name__=="__main__":
    app.run(debug=True)


    