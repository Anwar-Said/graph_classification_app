import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
import os
st.set_page_config(page_title="Graph ML", page_icon=None, layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.title('Benchmarking Graph Neural Networks')
# G = nx.karate_club_graph()
# pos =nx.kamada_kawai_layout(G)
# fig, ax = plt.subplots()
# nx.draw_networkx(G, pos =pos, ax=ax)
# st.pyplot(fig)
st.write('<span style="font-size: 24px;">This app provides pretrained graph machine learning models for inference on graph-structured data. </span>', unsafe_allow_html=True)




# st.pyplot(fig)

# Add a textbox and button for user interaction
# user_input = st.text_input("Enter a node to highlight", "0")







col1, col2, col3,col4,col5 = st.columns(5)
with col1:
    # st.write('Datasets')
    dataset = st.selectbox('Select dataset',['MUTAG','PTC'])

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



st.write('<span style="font-size: 24px;">classify graphs through descriptors. </span>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
folder_path = "test_samples/"
with c1:
    # st.write('Datasets')
    uploadg1 = st.file_uploader('Upload graph or select from the box to the right')
    # fig = plt.figure(figsize = (0.5,0.5))
    # if uploadg1 is not None:
    #     graph = pickle.loads(uploadg1.read())
        # nx.draw_networkx(graph, pos = nx.kamada_kawai_layout(graph),with_labels=False,node_size=0.5)
    # st.pyplot(fig)

with c2:
    files = [f.split(".")[0]+"_graph" for f in os.listdir(folder_path) if f.endswith(".pkl")]
    files.insert(0,"select graph ...")
    selected_file = st.selectbox("Select a file", files,index=0)

classify = st.button('Classify')
if classify:
    if selected_file=="select graph ..." and uploadg1 is None:
        st.write("please upload a graph or select from the box")
        pass
    if uploadg1 is None and selected_file!="select graph ...":
        file = selected_file.split("_")[0]+".pkl"
        st.write(file)
        graph = pickle.load(open(os.path.join(folder_path,file), "rb"))
        classify_graph(graph)
        # selected_option = st.selectbox('Select an option', files, index=None)
    elif uploadg1 is not None:
        try:
            graph = pickle.loads(uploadg1.read())
            if isinstance(graph,nx.Graph):
                classify_graph(graph)
                # uploadg1.clear()
                # selected_option = st.selectbox('Select an option', files, index=None)
            else:
                st.write("data format incompatible!")
        except:
            st.write("data format incompatible! please upload a networkx graph in pickle format")




    
    
