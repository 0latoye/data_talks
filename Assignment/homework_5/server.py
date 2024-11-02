from flask import Flask, request, jsonify
import waitress
import pickle
import socket

url = '192.168.2.2'

path = r'work\Assignment\homework_5'
def predict_single(customer,dv,model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

with open(f'{path}\model1.bin','rb') as model_in:
    model = pickle.load(model_in)

with open(f'{path}\dv.bin','rb') as dv_in:
    dv = pickle.load(dv_in)

app = Flask(__name__)


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