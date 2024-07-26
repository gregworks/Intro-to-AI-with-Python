import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from joblib import load
from sklearn import svm

app = Flask(__name__)
clf3 = load('my_model.joblib')

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.io.json.json_normalize(data)
    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    df['feature_1'] = df['feature_1'].astype(float)
    df['feature_2'] = df['feature_2'].astype(float)
    df['feature_3'] = df['feature_3'].astype(float)
    df['feature_4'] = df['feature_4'].astype(float)
    prediction = clf3.predict(df[feature_cols])
    output = str(prediction[0])
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5010, debug=True)