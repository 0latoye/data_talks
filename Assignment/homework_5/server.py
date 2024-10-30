from flask import Flask, request, jsonify
import requests
import waitress
import pickle
import socket
import numpy
url = '192.168.2.2'

def predict_single(customer,dv,model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

with open('model1.bin','rb') as model_in:
    model = pickle.load(model_in)

with open('dv.bin','rb') as dv_in:
    dv = pickle.load(dv_in)

app = Flask(__name__)

# @app.route('/')
# def model():
#     global client
#     # client = {"job": "student", "duration": 280, "poutcome": "failure"}
#     client = {"job": "management", "duration": 400, "poutcome": "success"}
#     model_path = r'C:\Users\hk3to\OneDrive\Documents\Data_Talks\Machine learning\work\Assignment\homework_5\model1.bin'
#     dv_path = r'C:\Users\hk3to\OneDrive\Documents\Data_Talks\Machine learning\work\Assignment\homework_5\dv.bin'
    
#     model = pickle.load(open(model_path,'rb'))
#     dv = pickle.load(open(dv_path,'rb'))
#     X = dv.transform(client)
#     prediction = list(model.predict_proba(X))
#     pred_value = round(prediction[0][1],3)



#     return jsonify({'Prediction':pred_value})
#     # return "The root works"

@app.route('/predict',methods = ['POST'])
def predict():
    client = request.get_json()
    prediction = predict_single(client,dv,model)
    return jsonify({'prediction':prediction})

hostname = socket.gethostname()
ip_addr = socket.gethostbyname(hostname)

if __name__ == "__main__":
    # app.run(debug=True,host ='0.0.0.0')
    app.run(host =ip_addr,port=8080, debug=True)