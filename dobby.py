# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:22:20 2020

@author: nikil reddy
"""

import streamlit as st

import numpy as np
import pandas as pd

import sklearn
import matplotlib.pyplot as pyplot 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
import altair as alt

import xgboost as xgb

def main():
    activities =  ["EDA &VIZ" , "Modelling"]
    choice = st.sidebar.selectbox("Select Activities",activities)
    if choice == 'EDA &VIZ':
        st.title('Play with ML')
        html_temp = """
        <div style="background-color:tomato;padding:12px">
        <h2 style="color:white;text-align:center;"> Play with ML App </h2>
        </div>
        """
        st.header('hey,tired of modelling and tuning ML Models,  wanna play with data & ML modles? Then upload a data here.. **_Dobby , a free elf_** is here for you ')
        st.subheader("Exploratory Data Analysis & Vizualization ")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        
        if data is not None:
            st.subheader('EDA')
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.write('shape:',df.shape)
            
            if st.checkbox("Show Columns"):
                all_columns = df.columns.to_list()
                st.write(all_columns)
                
            if st.checkbox("Null values"):
                st.write(df.isnull().sum())
                
            if st.checkbox("Information"):
                st.write(df.info())
                
            if st.checkbox("Summary"):
                 st.write(df.describe())
                 
            if st.checkbox("Show Selected Columns"):
                all_columns_names = df.columns.tolist()
                selected_columns = st.multiselect("Select Columns",all_columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
                
            if st.checkbox("Correlation Plot(Seaborn)"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
                
            st.subheader('Data Visualization')
            
            if st.checkbox("Show Value Counts"):
                column = st.selectbox("Select a Column to show value counts",all_columns)
                st.write(df[column].value_counts().plot(kind='bar'))
                
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","pie","line","hist","box","kde","altair_chart"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
            
            if st.button("Generate Plot"):
                st.success("Generating   {} plot  for {}".format(type_of_plot,selected_columns_names))
                
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)
                    
                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)
                    
                elif st.checkbox("Pie Plot"):
                    column_to_plot = st.selectbox("Select 1 Column",selected_columns_names)
                    pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie_plot)
                    st.pyplot()
                    
                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)
                    
                elif type_of_plot == 'altair_chart':
                    a = st.selectbox("Select X axis",all_columns)
                    b = st.selectbox("Select Y axis",all_columns)
                    c = st.selectbox("Select a column ",all_columns)
                    cust_data = pd.DataFram([a,b,c])
                    c = alt.Chart(cust_data).mark_circle().encode(x='a', y='b',size='c', color='c', tooltip=['a', 'b', 'c'])
                    st.altair_chart(c, use_container_width=True)
                    
                elif type_of_plot:
                    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()
                    
    if choice == 'Modelling':
        st.header('Training')
        st.subheader("Hello Iam Dobby. Dobby has no master - Dobby is a free elf. Due to SARS-CoV-2 lockdown I dont have much work to do , So Iam here to make your model.")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        
        if data is not None:
            st.subheader('EDA')
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.write('shape:',df.shape)
            
            st.header('Data Preprocessing')
            
            features = st.multiselect("Select feature columns",all_columns)
            X = df[features]
            st.dataframe(X)
            st.write(X.head())
            st.write(X.shape)
            labels = st.selectbox("Select label column",all_columns)
            y = df[labels]
            st.dataframe(y)
            st.write(y.head())
            st.write(X.shape)
            
            all_columns = X.columns.tolist()
            
            if st.checkbox("Handling missing values"):
                
                radioval = st.radio("choose type",('ffill','statistical')) 
                
                if radioval == 'ffill':
                    X=X.ffill(axis = 0)
                    
                elif radioval == 'statistical':
                    if st.checkbox("handle with mean"):
                        selected_columns = st.multiselect("Select Columns to handle with mean ",all_columns)
                        X[selected_columns] = X[selected_columns].fillna(X[selected_columns].mean(),inplace = True)
                        st.write('handled with mean')
                        
                    elif st.checkbox("handle with median"):
                        selected_columns = st.multiselect("Select Columns to handle with median",all_columns)
                        X[selected_columns] = X[selected_columns].fillna(X[selected_columns].median(),inplace = True)
                        st.write('handled with median')
                        
                    elif st.checkbox("handle with mode"):
                        selected_columns = st.multiselect("Select Columns to handle with mode",all_columns)
                        X[selected_columns] = X[selected_columns].fillna(X[selected_columns].mode()[0],inplace = True)
                        st.write('handled with mode')
                        
                        
            if st.checkbox("One hot encoding"):
                X = pd.get_dummies(X)
                
            st.write('Train - val split')
            number=st.number_input('test split size', min_value=0.00, max_value=1.00)
            from sklearn.cross_validation import train_test_split  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = number, random_state = 0)
            st.write(X_train.shape)
            st.write(X_test.shape)
            
            
            if st.checkbox("Feature Scaling"):
                radioval = st.radio("choose type of feature scaling",('Standardization','Normalization'))
                if radioval == 'Standardization':
                    from sklearn.preprocessing import StandardScaler
                    sc_X = StandardScaler()
                    X_train = sc_X.fit_transform(X_train)
                    X_test = sc_X.transform(X_test)
                    #sc_y = StandardScaler()
                    #y_train = sc_y.fit_transform(y_train)
                if radioval == 'Normalization':
                    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
                    X_train = min_max_scaler.fit_transform(X_train)
                    X_test = min_max_scaler.transform(X_test)
                    
            st.header("Training")
            models=['Logistic Regression','KNN','SVM','Random Forest']
            model = st.selectbox("Select  a model ",models)
            st.sidebar.markdown("Hyperparameter Tuning")
            
            if model == 'Logistic Regression':
                from sklearn.linear_model import LogisticRegression
                classifier = LogisticRegression(random_state = 0)
                classifier.fit(X_train, y_train)
                
            if model == 'KNN':
                n_neighbors = st.sidebar.slider('n_neighbors',min_value=1, max_value=5, step=1)
                p = st.sidebar.selectbox("P",[1,2,3,4])
                from sklearn.neighbors import KNeighborsClassifier
                classifier = KNeighborsClassifier(n_neighbors = n_neighbors, metric = 'minkowski', p = p)
                
            if model == 'SVM':
                from sklearn.svm import SVC 
                kernel_list = ['linear','poly','rbf','sigmoid']
                kernel = st.sidebar.selectbox("P",kernel_list)
                C = st.sidebar.slider('C',min_value=1, max_value=6, step=1)
                degree = st.sidebar.slider('Degree',min_value=1, max_value=10, step=1)
                classifier=SVC(kernel= kernel, C=C, random_state=0, degree=degree )
                classifier.fit(X_train,y_train)
                
            if model == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                criterion = st.sidebar.selectbox("criterion",["gini","entropy"])
                n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=500 , step=1) 
                max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                classifier = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion, max_depth = max_depth , random_state = 0)
                classifier.fit(X_train, y_train)
                
            y_pred = classifier.predict(X_test)
            from sklearn.metrics import accuracy_score
            acc=accuracy_score(y_test, y_pred)
            st.subheader('val_accuracy:',acc)
            
            
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)
            
            st.write(y_pred.value_counts().plot(kind='bar'))
            st.ballons()
            

if __name__ == '__main__':
    main()

