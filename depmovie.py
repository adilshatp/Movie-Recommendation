# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:22:33 2022

@author: ACER
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import streamlit as st
import pickle

st.title("Movie Recommandation")

def user_input_parameters():
    title=st.text_input('enter movie name')
    data={'movie_name':title
          }
    features = pd.DataFrame(data,index = [0])
    return features

submit=st.button('Recommended Movies')


dd=user_input_parameters()
st.subheader('Movies') 
st.write(dd)


model = pickle.load(open('REcommendation_movie.pickle','rb'))
model(dd)

st.subheader("Result=")
st.write(model)