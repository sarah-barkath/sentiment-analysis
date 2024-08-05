from typing import final
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
import tensorflow as tf
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoConfig
import torch.nn.functional as F
from IPython.display import HTML

# Use a pipeline as a high-level helper
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

#tokenization using given model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model_tf = TFAutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

MODEL=f"cardiffnlp/twitter-roberta-base-sentiment-latest"
config = AutoConfig.from_pretrained(MODEL)

# Generic code for model pipeline and tokenization ends here
#insert excel sheet for analysis of multiple user inputs
def xl_to_df(file):
    df=pd.read_excel(file)
    print(df)
    return df

#inserting all sentences into a data frame
def one_col(df):
    df_new = df.iloc[:,[1]]
    print(df_new)
    print(len(df_new))
    return df_new

# one user input + adding into data frame
def sentence_input(user_input):
    df = pd.DataFrame({'sentence': [user_input]})
    print(df)
    df_new = df
    return df_new

#removes all special characters
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

#tokenization; gives an array of tensor(s)

def token_input_total(dataframe):
    ip=[]
    for i in dataframe:
        text = i
        text = preprocess(text)
        input = tokenizer(text, return_tensors="pt")
        ip.append(input)
    return ip

#row-wise sentiment analysis
#def token_input(df_new):
#    ip=[]
#    for i, row in df_new.iterrows():
#        text = row[0]
#        text = preprocess(text)
#        input = tokenizer(text, return_tensors="pt")
#        ip.append(input)
#    return ip

#row-wise
def token_input(dataframe):
    ip=[]
    for i, row in dataframe.iterrows():
        text = row[0]
        text = preprocess(text)
        input = tokenizer(text, return_tensors="pt")
        ip.append(input)
    return ip

#calculates logits for sentence(s)
def logit_calc(inputs):
    logits_list = []
    for i in inputs:
        with torch.no_grad():
            outputs = model(**i)
            logits = outputs.logits
            logits_list.append(logits)
    return logits_list

#printing sentiment results
def final_results(prob):
    scores = prob[0].detach().numpy()
    s = scores*100
    print(s)
    ranking = np.argsort(s)
    ranking = ranking[::-1] #descending
    sentiment = []
    percentage  = []
    for i in range(s.shape[0]):
        l = config.id2label[ranking[i]]  #0-negative 1-neutral 2-positive
        sc = s[ranking[i]]
        sentiment.append(l)
        percentage.append(np.round(float(sc), 3))
        print(f"{i+1}) {l} {np.round(float(sc), 4)}") #descending order of labelled scores
    print(sentiment, percentage)
    return sentiment, percentage

#calculates probabilities of given logits
def prob_op(logit):
    probabilities = F.softmax(logit, dim=1)
    return probabilities

#DRIVER FUNCTIONS 
def analyze_text(user_input):
    a = sentence_input(user_input) #takes sentence insert into df
    inputs = token_input(a) #takes df tokenizes; returns array of tensors aka inputs
    print(inputs)
    logits_list = logit_calc(inputs) #calculates logit value of every input and returns array of logits
    print(logits_list)
    p =[]
    for i in logits_list:
        prob = prob_op(i)  #probability of every logit value
        p = p + [prob]  #array of probabilities
        sentiment, percentage = final_results(prob)
    return sentiment, percentage

def analyze_excel_total(b):
    #a = xl_to_df(file)
    #b = one_col(a)
    question = b.columns[0]
    inputs = token_input_total(b)
    print(inputs)
    logits_list = logit_calc(inputs)
    print(logits_list)
    p =[]
    for i in logits_list:
        prob = prob_op(i)  #probability of every logit value
        p = p + [prob]  #array of probabilities
        sentiment, percentage = final_results(prob)
    return sentiment, percentage, question, b  #returns list of srrings, floats and string and original data

def analyze_excel_byrow(data):
    #a = xl_to_df(file) #file to dataframe
    #data = one_col(a)  #one column dataframe with all sentences
    question = data.columns[0]  #title of dataframe
    inputs = token_input(data) #for every row in dataframe it pre-processes, tokenizes
    print(inputs)
    logits_list = logit_calc(inputs)
    print(logits_list)
    p = np.empty((0,3))
    for i in logits_list:
        prob = prob_op(i)
        prob = np.array(prob)
        p = np.vstack((p,prob))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            p[i,j] = np.round(p[i,j],4)
    p[:,[2,0]] = p[:,[0,2]]
    p = np.multiply(p,100)
    p = pd.DataFrame(p)
    data = pd.DataFrame(data)
    data = pd.concat([data,p], axis=1)
    data.columns = ['data','positive score','neutral score','negative score']
    html = data.to_html(classes='my_table')
    print(data)
    return html, question, data  #returns html table with all questions and answers and string value

#for  multiple questions in excel sheet, make a function that converts sheet into multiple 1 column dataframes
#calculate sentiment for the dataframes (consolidated or one by one)
#return html table for every question

#for all columns in dataframe that's converted from excel file to pandas dataframe, perform analyze_excel_byrow and analyze_excel_total

