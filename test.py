from flask import Flask,render_template,request
import joblib
import numpy as np

preds_args= [1, 1, 4, 2, 3,1,2, 1, 2, 3, 4,1, 1, 4, 2, 3,1,2, 1, 2, 3, 4,2]
preds_args_arr=np.array(preds_args)
preds_args_arr=preds_args_arr.reshape(1,-1)

regression_model_load=open('decision_tree_classifier.pkl','rb')
regression_model=joblib.load(regression_model_load)
preds=regression_model.predict(preds_args_arr)
model_prediction=int(preds)
print(model_prediction)