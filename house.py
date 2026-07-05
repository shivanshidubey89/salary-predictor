import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #ml 
from sklearn.metrics import r2_score #for accuracy check


st.set_page_config(page_title=" 🏘️House Price Prediction",layout="centered")

st.write("Predict House Price Based on differents Parameters")

# data load

df=pd.read_csv("HousePrice.csv")

with st.expander("📜 View Dataset"):
    st.dataframe(df)

# input data and output
x=df[["Area","Bedrooms","Bathrooms"]]
y=df["Price"]


# device data for train test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# train the selected model

model=LinearRegression()
model.fit(x_train,y_train)

# model accurecy

y_pred = model.predict(x_test)

score=r2_score(y_test,y_pred)

with st.container(border=True):
    st.subheader("know  ur future House Price 🤔?")

    area=st.number_input("enter ur square foot :", min_value=500,max_value=1000,step=100)
    bedrooms=st.number_input("enter ur bedroom size:",min_value=1,max_value=10,step=1)
    bathrooms=st.number_input("enter ur Bathrooms size :",min_value=1,max_value=10,step=1)


    if st.button("Predict Price ☺️"):
        prediction = model.predict([[area, bedrooms, bathrooms]])
        


        st.success(f"Predicted House Price:₹{prediction[0]:,.2f}")
        st.info(f"Model Accurecy (R2 score) :{score:.4f}")

        st.subheader("📊 Area vs House Price")

        fig,ax=plt.subplots(figsize=(6,3))

        ax.scatter(df["Area"],df["Price"],color="blue",s=50,label="House Data")

        ax.plot(df["Area"],model.predict(x),color="red",linewidth=1,label="Regression Line")
        ax.set_title("Area vs House Price")
        ax.set_xlabel("Area based sq.ft")
        ax.set_ylabel("price")
        ax.legend()
        st.pyplot(fig)