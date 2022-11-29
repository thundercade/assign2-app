#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: dupr0017@umn.edu
"""
#imports
import streamlit as st
import pandas as pd
st.write(pd.__version__)

import pickle as pkl
from tqdm import tqdm

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from string import punctuation
from collections import Counter
from heapq import nlargest

import os
# nlp = spacy.load("en_core_web_sm")

# import sentence_transformers
# from sentence_transformers import SentenceTransformer, util

with open("tokyo_corpus_embeddings.pkl" , "rb") as file_1, open("tokyo_df.pkl" , "rb") as file_2, open("tokyo_corpus.pkl" , "rb") as file_3:
    corpus_embeddings = pkl.load(file_1)
    df = pkl.load(file_2)
    corpus = pkl.load(file_3)



st.title("MABA 6490 -- Assignment 2 -- Hotel Search")
st.markdown("This app will recommend a hotel in Tokyo based on your input below")
st.markdown("This is v1")

text = st.text_input('Enter text:')

st.write("You searched for:", text)

st.write(corpus_embeddings.shape)
st.write(df.shape)

st.write("You made it to the end!")


#


#
# @st.cache(persist=True)
# def load_data():
#     df = pd.read_csv("https://datahub.io/machine-learning/iris/r/iris.csv")
#     return(df)
#
#
#
# def run():
#     st.subheader("Iris Data Loaded into a Pandas Dataframe.")
#
#     df = load_data()
#
#
#
#     disp_head = st.sidebar.radio('Select DataFrame Display Option:',('Head', 'All'),index=0)
#
#
#
#     #Multi-Select
#     #sel_plot_cols = st.sidebar.multiselect("Select Columns For Scatter Plot",df.columns.to_list()[0:4],df.columns.to_list()[0:2])
#
#     #Select Box
#     #x_plot = st.sidebar.selectbox("Select X-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=0)
#     #y_plot = st.sidebar.selectbox("Select Y-axis Column For Scatter Plot",df.columns.to_list()[0:4],index=1)
#
#
#     if disp_head=="Head":
#         st.dataframe(df.head())
#     else:
#         st.dataframe(df)
#     #st.table(df)
#     #st.write(df)
#
#
#     #Scatter Plot
#     fig = px.scatter(df, x=df["sepallength"], y=df["sepalwidth"], color="class",
#                  size='petallength', hover_data=['petalwidth'])
#
#     fig.update_layout({
#                 'plot_bgcolor': 'rgba(0, 0, 0, 0)'})
#
#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#
#     st.write("\n")
#     st.subheader("Scatter Plot")
#     st.plotly_chart(fig, use_container_width=True)
#
#
#     #Add images
#     #images = ["<image_url>"]
#     #st.image(images, width=600,use_container_width=True, caption=["Iris Flower"])
#
#
#
#
#
# if __name__ == '__main__':
#     run()
