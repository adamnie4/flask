#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import pickle
from flask import Flask, request
from flask_restful import Api
from modelowanie import modelowanie 

modelowanie()

with open("model.pkl", 'rb') as wczytywanie:
    model_wczytany = pickle.load(wczytywanie)
    
app = Flask(__name__)
api = Api(app)

@app.route('/')
def home():
    return 'Adam Niewiadomski 116961, przejdź do zakladki /predykcja/'



@app.route('/predykcja/', methods=['GET'])

def predykcja():
    pl = request.args.get("pl", "2.2")
    sl = request.args.get("sl", "3.5")
    
    res = model_wczytany.predict([sl, pl])
    mapper = {'0': 'setosa',
              '1': 'versicolor'}
    
    return mapper[f"{res}"]


app.run(port='5003')

