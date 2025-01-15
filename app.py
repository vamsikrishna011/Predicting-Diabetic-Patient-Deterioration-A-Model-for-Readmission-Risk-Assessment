import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score as acc

from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
# app = Flask(__name__)
# @app.route('/test')

from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

# ml_file = open("./Models/final_model_XGB.pkl", "rb")
# ml_model = joblib.load(ml_file)

@app.route("/")
def home():

    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            #print("Hi")
            race = int(request.form['race'])
            gender = int(request.form['gender'])
            age = float(request.form['age'])
            admission_type_id = float(request.form['admissiontypeid'])
            discharge_disposition_id = float(request.form['dischargedispositionid'])
            time_in_hospital = float(request.form['timeinhospital'])
            num_lab_procedures = float(request.form['numlabprocedures'])
            num_procedures = float(request.form['numprocedures'])
            num_medications = float(request.form['nummedications'])
            number_emergency  = float(request.form['numberemergency'])
            diag_1 = int(request.form['diag1'])
            diag_2 = int(request.form['diag2'])
            diag_3 = int(request.form['diag3'])
            number_diagnoses = float(request.form['numberdiagnoses'])
            max_glu_serum = int(request.form['maxgluserum'])
            A1Cresult = int(request.form['acresult'])
            metformin = int(request.form['metformin'])
            repaglinide  = int(request.form['repaglinide'])
            nateglinide  = int(request.form['nateglinide'])
            chlorpropamide = int(request.form['chlorpropamide'])
            glimepiride = int(request.form['glimepiride'])
            acetohexamide = int(request.form['acetohexamide'])
            #print("middle1")
            glipizide = int(request.form['glipizide'])
            glyburide = int(request.form['glyburide'])
            tolbutamide = int(request.form['tolbutamide'])
            pioglitazone = int(request.form['pioglitazone'])
            rosiglitazone = int(request.form['rosiglitazone'])
            acarbose = int(request.form['acarbose'])
            miglitol = int(request.form['miglitol'])
            troglitazone  = int(request.form['troglitazone'])
            tolazamide = int(request.form['tolazamide'])
            insulin = int(request.form['insulin'])
            #print("mid2")
            glyburidemetformin = int(request.form['glyburidemetformin'])
            glipizidemetformin  = int(request.form['glipizidemetformin'])
            metforminpioglitazone = int(request.form['metforminpioglitazone'])
            change = int(request.form['change'])
            diabetesMed  = int(request.form['diabetesmed'])

            #print("end")

            args_arr = [[race, gender, age, admission_type_id, discharge_disposition_id, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_emergency, diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, insulin, glyburidemetformin, glipizidemetformin, metforminpioglitazone, change, diabetesMed]]

            #print(args_arr)
            main_arr = np.array(args_arr)

            import joblib
            ml_model = joblib.load('./Models/XGB.joblib')

            mod_predict = ml_model.predict(main_arr)
            x_main = mod_predict[0]
            print(x_main)

        except ValueError:

            return "Check whether the values entered correctly or not"


    return render_template('predict.html', predit = x_main)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
