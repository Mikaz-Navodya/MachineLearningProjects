import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay , PrecisionRecallDisplay
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from pathlib import Path
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    @st.cache_data(persist=True)
    def LoadDataSet():
        csv_path = Path.cwd() / "StreamlitWebapp-MushroomClassifier/mushrooms.csv"
        df = pd.read_csv(csv_path)
        LE = LabelEncoder()
        for col in df.columns:
            df[col]=LE.fit_transform(df[col])
        return df
    
    @st.cache_data(persist=True)
    def split(df):
        Y=df.type
        X=df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(
                model, 
                x_test, 
                y_test, 
                display_labels=classNames,
                cmap="Blues",
                ax=ax
            )
            st.pyplot(fig)


        if 'ROC Curve'in metrics_list:
            st.subheader('ROC curve')
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(
                model,
                x_test,
                y_test,
                ax=ax
            )
            st.pyplot(fig)

        if 'Precision Recall curve'in metrics_list:
            st.subheader('Precision Recall curve')      
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(
                model,
                x_test,
                y_test,
                ax=ax
            )
            st.pyplot(fig)
                
    df = LoadDataSet()
    x_train, x_test, y_train, y_test = split(df)
 #  x_train=x_train.to_numpy().reshape(-1, 1)
 #   y_train=y_train.to_numpy().reshape(-1, 1)
    print(x_train.shape)  # (?, ?)
    print(y_train.shape)  # (?)

    classNames = ['Edible','Poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression","Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C= st.sidebar.number_input("C (Regularization paramter)", 0.01, 10.0, step=0.01, key='C')
        kernel= st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma =st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect('What metrics to plot',('Confusion Matrix','ROC Curve','Precision Recall curve'))
        
        if st.sidebar.button('Classify',key='classify'):
            st.subheader('Support Vector Machine Results')
            model=SVC(C=C,kernel=kernel,gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_predict= model.predict(x_test)
            st.write('Accuracy = ', round(accuracy,2))
            st.write('Precision Score',round(precision_score(y_test,y_pred=y_predict,labels=classNames),2))
            st.write('Recall Score =',round(recall_score(y_test,y_predict,labels=classNames),2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Hyperparameters')
        C=st.sidebar.number_input("C (Regularization Parameter)",0.01,10.0,step=0.01,key='C_Lgr')
        max_iter=st.sidebar.slider("Max iterations", 100, 500,key='max_iter')
        metrics=st.sidebar.multiselect('What metrics to plot',('Confusion Matrix','ROC Curve','Precision Recall curve'))

        if st.sidebar.button('Classify',key='classify'):
            st.subheader('Logistic Regression')
            model=LogisticRegression(C=C,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write('Accuracy =',round(accuracy,2))
            st.write('Precision Score =', round(precision_score(y_test,y_pred=y_pred,labels=classNames),2))
            st.write('Recall Score =',round(recall_score(y_test,y_pred,labels=classNames),2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_est')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio ("Bootstrap samples when building trees", (True, False), key='bootstrap')
        metrics=st.sidebar.multiselect('What metrics to plot',('Confusion Matrix','ROC Curve','Precision Recall curve'))
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred= model.predict(x_test)
            st.write("Accuracy: ",round (accuracy,2))
            st.write("Precision",round( precision_score(y_test, y_pred, labels=classNames),2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, labels=classNames),2))
            plot_metrics(metrics)

    if st.sidebar.checkbox('Show raw data',False):
        st.subheader('Mushroom Data set')
        st.write(df)
if __name__ == '__main__':
    main()
