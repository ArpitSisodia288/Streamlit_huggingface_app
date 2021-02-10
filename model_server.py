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
context = r"""The lion (Panthera leo) is a species in the family Felidae and a member of the genus Panthera. It has a muscular, deep-chested body, short, rounded head, round ears, and a hairy tuft at the end of its tail. It is sexually dimorphic; adult male lions have a prominent mane. With a typical head-to-body length of 184–208 cm (72–82 in) they are larger than females at 160–184 cm (63–72 in). It is a social species, forming groups called prides. A lion pride consists of a few adult males, related females and cubs. Groups of female lions usually hunt together, preying mostly on large ungulates. The lion is an apex and keystone predator; although some lions scavenge when opportunities occur and have been known to hunt humans, the species typically does not.
"""
question_context = st.text_area('Please type context of question', context)
question_user=st.text_area( 'please provide quesation', "Where can I find lion?")

st.header('Task 2- please provide necessery inputs for text_generation')
user_text_input = st.text_input('sentence to start for text generation', 'Google platform')
number = st.number_input('Insert maximum length of paragraph', value= 30)

st.header('Task 3- please provide necessery inputs for sentiment- analysis')
user_text_input_sentiment = st.text_input('Please type context', 'India ia a vibrabnt country.')

st.header('Task 4- please provide necessery inputs for text summarization')
filename = st.text_input('Enter a file path:', '/Users/apple/Documents/Learning/sa.txt')
aa= open(filename).read()

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
