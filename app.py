import flask
import pickle
import pandas as pd
import Rec_fx as rf   
from flask import Flask,render_template

# Use pickle to load in the pre-trained model and preprocessed parts
with open('rr_model.pkl', 'rb') as rr:
    model1 = pickle.load(rr)
with open('list1.pkl', 'rb') as ll:
    list1 = pickle.load(ll)
with open('my_df.pkl', 'rb') as dd:
    df = pickle.load(dd)
name = list(df["name"])
# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates',static_folder="static")
app.config['DEBUG'] = True
# Set up the main route



@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        # Extract the input
        i = flask.request.form['user']
        

        kn_ps,rc = rf.sample_train_recommendation(model1,list1[0],list1[2],i,10,'name',mapping=list1[3].mapping()[2],tag='category',
                              user_features = list1[4],item_features=list1[5])
       
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('index.html',kp=kn_ps,r = rc,i = i,rname = '',cat = '')   

@app.route('/details/<rname>/<cat>')
def details(rname,cat):
    r_name=rname
    c = cat
    address = df.iloc[(name.index(r_name))]["address"]
    stars = df.iloc[(name.index(r_name))]['stars']
    rev_c = df.iloc[(name.index(r_name))]['review_count']
    hours = df.iloc[(name.index(r_name))]['hours']
    print(r_name)
    return render_template('details1.html',info=r_name,address=address,stars=stars,rev_c=rev_c,hours=hours,c=c)

if __name__ == '__main__':
    app.run()
