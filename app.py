import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="salary Prediction",layout="centered")

st.title("salary Prediction")
st.write("this app predicts salary based on year of experience.")

df = pd.read_csv("salary.csv")


x= df[["experience"]]
y=df["salary"]

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)


model= LinearRegression()
model.fit(x_train,y_train)


pred = model.predict(x_test)

score=r2_score(y_test,pred)

with st.container(border=True):
    st.subheader("salary Prediction")
    year=st.number_input("enter the year of experience:",min_value=0.0,max_value=24.0,step=0.5)


    if st.button("predict salary"):
        Prediction=model.predict([[year]])


        st.success(f"Predicted salary:₹{Prediction[0]:2f}")
        st.info(f"model accuracy score:{score:.4f}")

        fig,ax=plt.subplots(figsize=(7,4))

        ax.scatter(x,y, color='blue',label='salary Data',s=80)
        ax.plot(x,model.predict(x),color='pink',label='Regression Line',linewidth=1)

        ax.set_title("employe vs salary")
        ax.set_ylabel("salary")
        ax.set_xlabel("year of experience")
        ax.legend()
        plt.grid(True)

        st.pyplot(fig)