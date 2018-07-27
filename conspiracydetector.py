from flask import Flask, request, render_template
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.externals import joblib
from nltk import word_tokenize
import string
import re

app = Flask(__name__)

clf = joblib.load("model_notokenizer.pkl")
parameters = clf.named_steps['clf'].get_params()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze',methods=['POST','GET'])
def analyze():
    if request.method=='POST':
        result=request.form
        input_text = result['input_text']
        predicted = clf.predict([input_text])
        # print(predicted)
        certainty = clf.decision_function([input_text])
        
        # Is it bonkers?
        if predicted[0]:
            verdict = "Not too nuts!"
        else:
            verdict = "Bonkers!"
        
        return render_template('result.html',prediction=[input_text, verdict, float(certainty), parameters])
    
if __name__ == '__main__':
    #app.debug = True
    app.run()
