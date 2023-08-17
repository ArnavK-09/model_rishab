# imports
import pickle
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# set config
st.set_page_config(
    page_title="Talk to Rishab AI!",
    page_icon="🐵",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={"About": "https://github.com/ArnavK-09"},
)

# welcome
st.title("Talk to Rishab!", help="Currently In Development Version!")
st.caption(
    'Please be aware that while interacting with the AI model "Rishab," there is a possibility of receiving responses that may be incorrect or contain bugs. As with any AI system, it is important to exercise caution and critically evaluate the information provided. We continuously strive to improve "Rishab" to enhance its accuracy and reliability, but there might still be instances where its responses fall short.'
)
st.divider()

# model
DF = pd.read_csv("./data/dialogs.txt", sep="|")
MODEL = Pipeline( 
     [ 
         ("bow", CountVectorizer()), 
         ("tfidf", TfidfTransformer(sublinear_tf=True)), 
         ("classifier", RandomForestClassifier(n_estimators=100)), 
     ] 
 ) 
  
# data fit 
MODEL.fit(DF["question"], DF["answer"])
# get prompt
prompt = st.chat_input("Say something")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.balloons()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    av = "🫵" if message["role"] == "user" else "🐵"
    with st.chat_message(message["role"], avatar=av):
        st.write(message["content"])


# React to user input
if prompt:
    # Display user message in chat message container
    st.chat_message("user", avatar="🫵").write(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from model
    response = MODEL.predict([prompt])[0]

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🐵"):
        st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
