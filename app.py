import pickle
import sys
try:
    from flask import Flask, request, jsonify, url_for, render_template
except ImportError:
    print("Flask is not installed. Install it using: pip install flask")
    sys.exit(1)
try:
    import numpy as np
except ImportError:
    print("NumPy is not installed. Install it using: pip install numpy")
    sys.exit(1)
import pandas as pd
app = Flask(__name__)
classifymodel=pickle.load(open('conv2d.pkl','rb'))
@app.route('/')
def home():
    return render_template('pet.html')
@app.route('/predict',methods=['POST'])
def predict():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=classifymodel.predict(np.array(list(data.values())).reshape(1,-1))
    print(output)
    return jsonify(output[0])
if __name__=='__main__':
    app.run(debug=True)
