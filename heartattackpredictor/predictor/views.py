from django.shortcuts import render
from django.http import HttpResponseRedirect
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score 

def home(request):
    return render(request,'home.html')

def results(request,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,thal):
    file=pd.read_csv("C:\\Users\\Akshay Sharma\\projects\\heartattackpredictor\\templates\\heartattack.csv",na_values=['?'])
    file.drop(columns=['ca'],inplace=True)
    file['thalach'].fillna(file['thalach'].mean(),inplace=True)
    file.fillna(file.median(),inplace=True)
    file=file.rename(columns={'num       ':'num'})
    Y=file['num']
    X=file.drop('num',axis='columns')
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)
    model=RandomForestClassifier()
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)
    accuracy=f1_score(Y_test,Y_pred)
    k=pd.DataFrame(np.array([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,thal]).reshape(1,-1))
    Y_pred=model.predict(k)[0]
    return render(request,'result.html',{'Y_pred':Y_pred})

    


def info(request):
    if request.method=='POST':
        age=request.POST['age']
        sex=request.POST['sex']
        if sex=='M':
            sex=1
        else:
            sex=0
        cp=request.POST['cp']
        trestbps=request.POST['trestbps']
        chol=request.POST['chol']
        fbs=request.POST['fbs']
        restecg=request.POST['restecg']
        thalach=request.POST['thalach']
        exang=request.POST['exang']
        oldpeak=request.POST['oldpeak']
        slope=request.POST['slope']
        thal=request.POST['thal']
    return results(request,age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,thal)

