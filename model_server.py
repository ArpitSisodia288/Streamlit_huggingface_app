import pandas as pd
import joblib
import streamlit as st
import sys
from transformers import pipeline


# https://huggingface.co/models pre trained models of huggling face ( models)

#https://docs.streamlit.io/en/stable/api.html#display-interactive-widgets ( Streamlit Documentation)

st.title('Demo: Streamlit model serving and Visualization') #title
st.write('Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps - so let’s get started!')

# defining variables used as input



st.header('Task 1- please provide necessery inputs for question answering')
context = r"""GAIA team is in India. LB is a project.Lambda team is most efficient team in GAIA. Arpit is star of the team.
"""
question_context = st.text_area('Please type context of question', context)
question_user=st.text_area( 'please provide quesation', "Where can I find Ranjani?")

st.header('Task 2- please provide necessery inputs for text_generation')
user_text_input = st.text_input('sentence to start for text generation', 'Google platform')
number = st.number_input('Insert maximum length of paragraph', value= 30)

st.header('Task 3- please provide necessery inputs for sentiment- analysis')
user_text_input_sentiment = st.text_input('Please type context', 'Team Lambda is best R&D team in India.')

st.header('Task 4- please provide necessery inputs for text summarization')
# filename = st.text_input('Enter a file path:', '/Users/apple/Documents/Learning/sa.txt')
# aa= open(filename).read()
aa= context

## running the main code-

st.title( 'Please select any of the NLP tasks')

if st.button('Question-Answering'):
    # task 1 question answering
    nlp = pipeline("question-answering")
    result = nlp(question=question_user, context=question_context)
    st.write(result)
if st.button('Text generation'):      
    ## task 2- text generation
    generator = pipeline('text-generation', model='gpt2')
    op_text= generator(user_text_input, number, do_sample=False)
    st.write(op_text[0]['generated_text'])
    ##task 3- Sentiment Analysis
if st.button('Sentiment-Analysis'):
    nlp_sa = pipeline("sentiment-analysis")
    st.write(nlp_sa(user_text_input_sentiment))
    # tsk 4- text summarization
if st.button('Text summarization'):
    summarizer = pipeline("summarization")
    sum_op= (summarizer(aa, max_length=50, min_length=30, do_sample=False))
    st.write(sum_op)
    st.write('this feature is still in progress' )
