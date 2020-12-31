import pickle
import os
#import sqlite3
#import numpy as np
import streamlit



curr_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(curr_dir,'pkl_objects','classifier.pkl'), 'rb'))
fit = pickle.load(open(os.path.join("pkl_objects","fit.pkl"), "rb"))
db = os.path.join(curr_dir, 'Sentiment.sqlite')

def welcome():
    return "Welcome all"

#def sqlite_entry():
        
    #conn = sqlite3.connect(path)
    #cursor = conn.cursor()
    #cursor.execute("INSERT INTO sentiment"\
	#	"(review, Sentiment, date) VALUES"\
	#			"(?,?, DATETIME('now'))", (document,pred))
    #conn.commit()
    #conn.close()


def classify(document):
    label = {0:"negative", 1:"Positive"}
    pred = clf.predict(fit.transform(document))
    print(clf.predict_proba(fit.transform(document))*100)

    if pred == 0:
        label = "negative 😞"
        #sqlite_entry()
        
    elif pred == 1:
       
        label =" positive 😁"
        #sqlite_entry()
    return ( label)

def probability(document):
    prob = (clf.predict_proba(fit.transform(document))*100)
    return prob
    #,"with probability of:", )


def main():
    streamlit.title("Movie reviews prediction")
    html_temp = """
    <div style = "background-color:black; padding: 13px">
    <h1 style = "background-color:white; text-align: center;">
    """

    streamlit.markdown(html_temp, unsafe_allow_html = True)

    Movie_Name = streamlit.text_input("Enter Movie name here:")
    
    Review = [streamlit.text_input("Enter review here:")]
    result = ""

    if streamlit.button("Predict"):

        result = classify(Review)
        print(result)

        streamlit.success("This review is {}".format(result))
        

if __name__ == "__main__":
    main()   