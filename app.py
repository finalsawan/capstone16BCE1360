from flask import Flask,render_template,request
import joblib
import numpy as np
regression_model_load=open('decision_tree_classifier.pkl','rb')
regression_model=joblib.load(regression_model_load)
app=Flask(__name__)

@app.route('/')

def home():

    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])

def predict():
    if request.method == 'POST':
        try:
            id=float(request.form['id'])
            hour     =float(request.form['hour'])
            C1=float(request.form['C1'])
            banner_pos =float(request.form['banner_pos'])
            site_id       =float(request.form['site_id'])
            site_domain        =float(request.form['site_domain'])
            site_category=float(request.form['site_category'])
            app_id=float(request.form['app_id'])
            app_domain=float(request.form['app_domain'])
            app_category=float(request.form['app_category'])
            device_id=float(request.form['device_id'])
            device_ip=float(request.form['device_ip'])
            device_model     =float(request.form['device_model'])
            device_type=float(request.form['device_type'])
            device_conn_type =float(request.form['device_conn_type'])
            C14=float(request.form['C15'])
            C15=float(request.form['C15'])
            C16=float(request.form['C16'])
            C17=float(request.form['C17'])
            C18=float(request.form['C18'])
            C19=float(request.form['C19'])
            C20=float(request.form['C20'])
            C21=float(request.form['C21'])

            preds_args=[id,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,device_ip,device_model,
            device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21]
            preds_args_arr=np.array(preds_args)
            preds_args_arr=preds_args_arr.reshape(1,-1)
            
            preds=regression_model.predict(preds_args_arr)
            model_prediction=int(preds)
        except ValueError:
            return 'Check the values of the model'

    return render_template('predict.html',prediction=model_prediction)


if __name__ == "__main__":
    app.run()