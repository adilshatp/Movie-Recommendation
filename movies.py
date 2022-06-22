# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:14:16 2022

@author: ACER
"""
import matplotlib.image as mp
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import streamlit as st
import pickle

st.title("Movie Recommandation")

image=mp.imread("movieee.jpg")
st.image(image)

df=pd.read_csv(r"C:\Users\ACER\deployment Recomndation_movie\movies.csv")

selected_feature = ['genres','keywords','tagline','cast','director']

for feature in selected_feature:
    df[feature] = df[feature].fillna('')

combi_feature = df['genres']+' '+df['keywords']+' '+df['tagline']+' '+df['cast']+' '+df['director']

vec= TfidfVectorizer()
feature_vec=vec.fit_transform(combi_feature)

similarity= cosine_similarity(feature_vec)

movie_name =st.text_input('Enter your movie name: ')
submit=st.button('Recommended Movies')

if submit is True:

    st.write("THE MOVIES ARE")
    
    list_of_all_movies = df["title"].tolist()
    
    close_match =difflib.get_close_matches(movie_name,list_of_all_movies)
    
    close_match_new = close_match[0]
    
    index_movie =df[ df.title == close_match_new]['index'].values[0]
    
    similarity_score = list(enumerate(similarity[index_movie]))
    
    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse=True)
    
    print('Movies for you : \n')
    
    i=1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df[df.index==index]['title'].values[0]
        if (i<11):
            st.write(title_from_index)
            i+=1
          
    
    
       
