from doctest import Example
from typing import final
from venv import create
from flask import Flask, flash, render_template, request, jsonify, session, redirect, url_for, g
from flask.sessions import SessionInterface
from src.sentiment_analysis import analyze_excel_byrow, analyze_excel_total, analyze_text, xl_to_df
import pandas as pd
import numpy as np
import plotly as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from werkzeug.utils import secure_filename
import plotly.express as px
import lxml

app = Flask(__name__)
app.secret_key = "atul"
app.config['UPLOAD_FOLDER'] = './uploads/'  # Folder to store uploaded files

@app.route('/')
def home():
    session.clear()
    return render_template('index.html')

@app.route('/index.html')
def go_back():
    if 'file_result_total' or 'file_result_byrow' in session:
        session.pop('file_input',None)
        session.pop('file_result_byrow',None)
        session.pop('file_result_total',None)
        return redirect(url_for('home'))
    else:
        return redirect(url_for('home'))
    
@app.route('/dash',methods=['GET','POST']) # type: ignore
def dashboard():
    session.clear()
    session.pop('file_input',None)
    session.pop('file_result_total',None)
    session.pop('file_result_byrow',None)
    session.pop('text_input',None)
    session.pop('text_result',None)
    session.pop('heading',None)

    file = request.files['excelFiledash']
    if file.filename == '':
        flash('no file')
        return redirect(url_for('home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename('file.filename')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(file_path)
        g.df = pd.read_excel(file_path)
        dfstr = g.df.to_html(classes='my_table')
        option = request.form.get('operationMode')

        #if option == 'entire' or option == 'rows':
        consol_result, consol_ans, questions = calculate_sentiment_total(g.df)
        session['file_input'] = dfstr
        session['file_result_total'] = consol_result #{'filexyz...':[list of previously zipped lists]}
        session['heading'] = questions #{'heading':[list of strings]}
        #all_rows = all_rows.to_html(classes='my_table')
        individual_result, individual_ans, individual_questions = calculate_sentiment_byrow(g.df)
        session['file_result_byrow'] = individual_result

        unzipping_ans = [[i for i,j in consol_result],[j for i,j in consol_result]]
        list_of_dicts = unzipping_ans[1]

        unzipping_ans_byrow = [[i for i,j in individual_result],[j for i,j in individual_result]]
        list_of_htmls = unzipping_ans_byrow[1]
        list_of_dfs = [pd.read_html(a)[0] for a in list_of_htmls]

        #creating a grid
        cols = min(3, len(questions))  # Maximum 3 charts per row
        rows = (len(questions) + 2) // 3

        #pie chart
        consol_pie = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'domain'} for x in range(cols)] for y in range(rows)])
        color_map = {'positive': 'forestgreen','neutral': 'powderblue', 'negative': 'crimson'}
        for i, input_dict in enumerate(list_of_dicts):
            labels = list(input_dict.keys())
            data = list(input_dict.values())
            color_list = [color_map[label] for label in labels]
            #calculating place in grid
            row = i // cols + 1
            col = i % cols + 1

            consol_pie.add_trace(
                go.Pie(labels=labels, values=data, title=f'{i+1}', hole=0.4, marker=dict(colors=color_list)),row=row, col=col
            )
        
        consol_pie.update_layout(title_text='Consolidated Results', showlegend=True)
        consol_pie_json = json.dumps(consol_pie, cls=plt.utils.PlotlyJSONEncoder)

        #bar graph
        indi_bar = make_subplots(rows=len(questions), cols=1, subplot_titles=[f"{i+1}: {questions[i]}" for i in range(len(questions))], vertical_spacing=0.05)
        def wrap_labels(labels, width=60):
            return ['<br>'.join([label[i:i+width] for i in range(0, len(label), width)]) for label in labels]
        def calculate_row_height(labels, base_height=300, char_height=5):
            max_length = max(len(label) for label in labels)
            return base_height + (max_length * char_height)

        print(list_of_dfs)
        print("**********************************************************************************************************************************************")
        for index,df in enumerate(list_of_dfs):
            sentences = list(df.iloc[:,1])
            scores = df.iloc[:,2:]
            scores_list = scores.values.tolist()
            print(sentences)
            print("**********************************************************************************************************************************************")
            print(scores) #dataframe for all sentiment scores
            print("**********************************************************************************************************************************************")
            print(scores_list) #list of lists containing all sentiment scores for each
            formatted_labels = [label.replace('.', '.<br>') for label in sentences]
            wrapped_categories = wrap_labels(sentences)
            y_label_height = calculate_row_height(wrapped_categories)
            color_bar=['lightgreen','powderblue','crimson']
            for k in range(3):
                for xd,yd in zip(scores_list,wrapped_categories):
                    indi_bar.add_trace(go.Bar(x=[xd[k]],y=[yd],orientation="h", marker=dict(color=color_bar[k])),row=index+1,col=1)
        
        total_height = len(questions) * y_label_height
        indi_bar.update_layout(showlegend=False, barmode="stack", bargap=0.5, height=total_height, width=1200)
        indi_bar_json = json.dumps(indi_bar, cls=plt.utils.PlotlyJSONEncoder)

        return render_template('dashboard.html', consol_pie_json=consol_pie_json, questions=questions, indi_bar_json=indi_bar_json)
    else:
        return render_template('dashboard.html')

@app.route('/aftersubmit')
def data_viz():
    if 'file_result_total'and 'file_result_byrow' in session:
        input_data = session['file_input'] #html table
        questions = session['heading'] #list of strings
        consol_result = session['file_result_total'] #list(zip(questions,ans_list)) where ans_list=dict(zip(string array,int array))
        individual_result = session['file_result_byrow']#list(zip(questions,ans_list)) where ans_list = list[html tables]

        unzipping_ans = [[i for i,j in consol_result],[j for i,j in consol_result]]
        list_of_dicts = unzipping_ans[1]

        unzipping_ans_byrow = [[i for i,j in individual_result],[j for i,j in individual_result]]
        list_of_htmls = unzipping_ans_byrow[1]
        list_of_dfs = [pd.read_html(a)[0] for a in list_of_htmls]

        #creating a grid
        cols = min(3, len(questions))  # Maximum 3 charts per row
        rows = (len(questions) + 2) // 3

        #pie chart
        consol_pie = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'domain'} for x in range(cols)] for y in range(rows)])
        color_map = {'positive': 'forestgreen','neutral': 'powderblue', 'negative': 'crimson'}
        for i, input_dict in enumerate(list_of_dicts):
            labels = list(input_dict.keys())
            data = list(input_dict.values())
            color_list = [color_map[label] for label in labels]
            #calculating place in grid
            row = i // cols + 1
            col = i % cols + 1

            consol_pie.add_trace(
                go.Pie(labels=labels, values=data, title=f'{i+1}', hole=0.4, marker=dict(colors=color_list)),row=row, col=col
            )
        
        consol_pie.update_layout(title_text='Consolidated Results', showlegend=True)
        consol_pie_json = json.dumps(consol_pie, cls=plt.utils.PlotlyJSONEncoder)

        #bar graph
        indi_bar = make_subplots(rows=len(questions), cols=1, subplot_titles=[f"{i+1}: {questions[i]}" for i in range(len(questions))], vertical_spacing=0.05)
        def wrap_labels(labels, width=60):
            return ['<br>'.join([label[i:i+width] for i in range(0, len(label), width)]) for label in labels]
        def calculate_row_height(labels, base_height=300, char_height=5):
            max_length = max(len(label) for label in labels)
            return base_height + (max_length * char_height)

        print(list_of_dfs)
        print("**********************************************************************************************************************************************")
        for index,df in enumerate(list_of_dfs):
            sentences = list(df.iloc[:,1])
            scores = df.iloc[:,2:]
            scores_list = scores.values.tolist()
            print(sentences)
            print("**********************************************************************************************************************************************")
            print(scores) #dataframe for all sentiment scores
            print("**********************************************************************************************************************************************")
            print(scores_list) #list of lists containing all sentiment scores for each
            formatted_labels = [label.replace('.', '.<br>') for label in sentences]
            wrapped_categories = wrap_labels(sentences)
            y_label_height = calculate_row_height(wrapped_categories)
            color_bar=['lightgreen','powderblue','crimson']
            for k in range(3):
                for xd,yd in zip(scores_list,wrapped_categories):
                    indi_bar.add_trace(go.Bar(x=[xd[k]],y=[yd],orientation="h", marker=dict(color=color_bar[k])),row=index+1,col=1)
        
        total_height = len(questions) * y_label_height
        indi_bar.update_layout(showlegend=False, barmode="stack", bargap=0.5, height=total_height, width=1200)
        indi_bar_json = json.dumps(indi_bar, cls=plt.utils.PlotlyJSONEncoder)

        return render_template('dashboard.html',consol_pie_json=consol_pie_json,questions=questions,indi_bar_json=indi_bar_json)
    else:
        flash('no input')
        return redirect(url_for('home'))

@app.route('/submit/text',methods=['GET','POST'])
def submit_text():
    session.clear()
    if request.method == 'POST':
        session.pop('text_input',None)
        session.pop('text_result',None)
        session.pop('file_input',None)
        session.pop('file_result_total',None)
        session.pop('file_result_byrow',None)
        user_input = request.form['user_input']
        session['text_input'] = user_input
        sentiment, percentage = analyze_text(user_input)
        arr_text = list(zip(sentiment,percentage))
        session['text_result'] = arr_text
        return redirect(url_for('show_text',user_input=user_input,sentiment=sentiment, percentage=percentage,arr_text=session['text_result']))
    return render_template('index.html')

@app.route('/submit/text/answer')
def show_text():
    if 'text_result' in session:
        arr_text = session['text_result']
        user_input = session['text_input']
    return render_template('submit_text.html',arr_text=arr_text,user_input=user_input)

ALLOWED_EXTENSIONS = {'xls', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_sentiment_total(dataframe):
    answer_list_total = []
    questions = []
    for k in range(len(dataframe.columns)):
        col = dataframe.iloc[:,[k]]
        sentiments, scores, question, all_rows = analyze_excel_total(col)
        arr_file = dict(zip(sentiments, scores))
        answer_list_total.append(arr_file)
        questions.append(question)
    final_ans = list(zip(questions,answer_list_total))
    return final_ans, answer_list_total, questions

def calculate_sentiment_byrow(dataframe):
    answer_list_byrow = []
    questions = []
    for i in range(len(dataframe.columns)):
        ip = dataframe.iloc[:,[i]]
        html, question, data = analyze_excel_byrow(ip)
        answer_list_byrow.append(html)
        questions.append(question)
    answers = list(zip(questions,answer_list_byrow))
    return answers, answer_list_byrow, questions


@app.route('/submit/file',methods=['GET','POST'])
def submit_file():
    if request.method == 'POST':
        session.clear()

        session.pop('file_input',None)
        session.pop('file_result_total',None)
        session.pop('file_result_byrow',None)
        session.pop('text_input',None)
        session.pop('text_result',None)
        session.pop('heading',None)

        file = request.files['excelFile']
        if file.filename == '':
            flash('no file')
            return redirect(url_for('home'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename('file.filename')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(file_path)
            g.df = pd.read_excel(file_path)
            dfstr = g.df.to_html(classes='my_table')
            option = request.form.get('operationMode')

            #if option == 'entire' or option == 'rows':
            consol_result, consol_ans, consol_questions = calculate_sentiment_total(g.df)
            session['file_input'] = dfstr
            session['file_result_total'] = consol_result #{'filexyz...':[list of previously zipped lists]}
            session['heading'] = consol_questions #{'heading':[list of strings]}
            #all_rows = all_rows.to_html(classes='my_table')
            individual_results, individual_ans, individual_questions = calculate_sentiment_byrow(g.df)
            session['file_result_byrow'] = individual_results
            return redirect(url_for('show_file_total',file=file,final_ans=session['file_result_total'], questions=session['heading'], dfstr=session['file_input'],answers=session['file_result_byrow']))
    return render_template('index.html')

@app.route('/submit/file/answertotal')
def show_file_total():
    if 'file_result_total' in session:
        final_ans = session['file_result_total']
        df = session['file_input']
        question = session['heading']
    if 'file_result_byrow' in session:    
        answers = session['file_result_byrow']
    return render_template('submit_file.html',final_ans=final_ans,df=df,question=question,answers=answers)

@app.route('/submit/about.html')
@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/submit/documentation.html')
@app.route('/documentation.html')
def documentation():
    return render_template('documentation.html')

# main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=True)