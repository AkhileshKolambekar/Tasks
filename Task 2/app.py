from flask import Flask, request,jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route('/predict',methods = ['POST'])
def predict():
    try:    
        input_data = {key: float(request.form[key]) for key in request.form.keys()}

        prediction = model.predict([list(input_data.values())])[0]
        return jsonify({'Prediction':prediction})
    except Exception as e:
         return jsonify({'error':str(e)})

if __name__ == "__main__":
    app.run(debug = True)