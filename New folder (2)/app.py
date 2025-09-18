import streamlit as st
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd

df = pd.read_csv('mushrooms.csv')
st.write('Hello')
l=sk.linear_model.LinearRegression()
plt.figure()