import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
st.set_page_config(page_title="Graph ML", page_icon=None, layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.title('Predictive Modeling on Graph-structured Data')
# G = nx.karate_club_graph()
# pos =nx.kamada_kawai_layout(G)
# fig, ax = plt.subplots()
# nx.draw_networkx(G, pos =pos, ax=ax)
# st.pyplot(fig)
st.write('<span style="font-size: 24px;">This app provides pretrained graph machine learning models for inference on graph structured data. </span>', unsafe_allow_html=True)




# st.pyplot(fig)

# Add a textbox and button for user interaction
# user_input = st.text_input("Enter a node to highlight", "0")







col1, col2, col3,col4,col5 = st.columns(5)
with col1:
    # st.write('Datasets')
    dataset = st.selectbox('Select dataset',['MUTAG','PTC','NCI1'])

with col2:
    # st.write('Models')
    model = st.selectbox('Select model',['NetLSD','DGSD',"GCN","GAT"])
with col3:
    eval = st.selectbox('Select evaluation',["10-folds","80-20"])
with col4:
    # st.write('Metric')
    metric = st.selectbox("Select metric",['accuracy'])
with col5:
    train = st.selectbox("Select training",['pre-trained','train'])


run_btn = st.button("run")
if run_btn:
    run_model(dataset,model, eval, metric,train)