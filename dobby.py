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

import pickle
import base64

def main():
    activities =  ["EDA &VIZ" , "Modelling"]
    choice = st.sidebar.selectbox("Select Activities",activities)
    if st.sidebar.checkbox('About'):
        st.sidebar.markdown("""
                           app work in progress .This is a beta release.
                           
                           version: b-0.0.1
                           
                           initial release:27/6/2020
                           
                           helpful suggestions are welcome.
                           
                           contact: postme_@hotmail.com
                           """)
        
    if choice == 'EDA &VIZ':
        st.title('Play with ML')
        
        html_temp1 = """<img src="images/dobby1.jpeg" alt="It's dobby" width="120" height="150">"""
        st.markdown(html_temp1,unsafe_allow_html=True)
        st.write("can't see Dobby? I know because i do work from home , You will see me soon")
        html_temp = """
        <div style="background-color:coral;padding:12px">
        <h2 style="color:white;text-align:center;"> Play with ML App </h2>
        
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        st.markdown('hey,tired of modelling and tuning ML Models,  wanna play with data & ML modles? Then upload a dataset here.. **_Dobby , a free elf_** is here for you ')
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
                st.write(df[column].value_counts())
                st.write(df[column].value_counts().plot(kind='bar'))
                st.pyplot()
                
            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select Type of Plot",["area","bar","pie","line","hist","box","kde","altair_chart"])
            selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
            
            if st.button("Generate Plot"):
                st.success("Generating   {} plot  for {}".format(type_of_plot,selected_columns_names))
                
                if type_of_plot == 'area':
                    cust_data = df[selected_columns_names]
                    st.area_chart(cust_data)
                    st.pyplot()
                    
                elif type_of_plot == 'bar':
                    cust_data = df[selected_columns_names]
                    st.bar_chart(cust_data)
                    st.pyplot()
                    
                elif type_of_plot == "Pie Plot":
                    column_to_plot = st.selectbox("Select 1 Column",selected_columns_names)
                    pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
                    st.write(pie_plot)
                    st.pyplot()
                    
                elif type_of_plot == 'line':
                    cust_data = df[selected_columns_names]
                    st.line_chart(cust_data)
                    st.pyplot()
                    
                elif type_of_plot == 'altair_chart':
                    a = st.selectbox("Select X axis",all_columns)
                    b = st.selectbox("Select Y axis",all_columns)
                    c = st.selectbox("Select a column ",all_columns)
                    cust_data = pd.DataFrame([a,b,c])
                    c = alt.Chart(cust_data).mark_circle().encode(x='a', y='b',size='c', color='c', tooltip=['a', 'b', 'c'])
                    st.altair_chart(c, use_container_width=True)
                    st.pyplot()
                    
                elif type_of_plot:
                    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
                    st.write(cust_plot)
                    st.pyplot()
                    
    if choice == 'Modelling':
        html_temp = """
        <div style="background-color:coral;padding:12px">
        <h2 style="color:white;text-align:center;"> Play with ML App </h2>
        </div>
        """
        
        st.header('Training')
        st.markdown("**_Hello Iam Dobby. Dobby has no master - Dobby is a free elf_**. Due to SARS-CoV-2 lockdown I dont have much work to do , So Iam here to make your model.")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        
        if data is not None:
            st.subheader('EDA')
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.write('shape:',df.shape)
            
            st.header('Data Preprocessing')
            
            all_columns = df.columns.tolist()
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
                
                if radioval == 'fbfill':
                    if st.checkbox("fbfill"):
                        X=X.ffill(axis = 0)
                    elif st.checkbox("bfill"):
                        X=X.ffill(axis = 0)
                    st.markdown('**_missing values are fb filled_**')
                    
                    
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
                    st.markdown('**_missing values are filled statistically_**')
                        
                        
            if st.checkbox("One hot encoding"):
                if st.checkbox("encode features"):
                    X = pd.get_dummies(X)
                    st.write("features are one hot encoded")
                if st.checkbox("encode labels"):
                    y = pd.get_dummies(y)
                    st.write("labels are one hot encoded")
                    st.dataframe(y)
                
                
            st.write('Train - val split')
            number=st.number_input('test split size', min_value=0.05, max_value=1.00)
            from sklearn.model_selection import train_test_split  
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = number, random_state = 0)
            st.write(X_train.shape)
            st.write(X_test.shape)
            
            
            if st.checkbox("Feature Scaling"):
                radioval = st.radio("choose type of feature scaling",('none','Standardization','Normalization'))
                if radioval == 'none':
                    st.write("you skipped feature scaling")
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
            problem_types=['Regression','Classification']
            problem_type = st.selectbox("Select Problem Type ",problem_types)
            st.sidebar.markdown("Hyperparameter Tuning")
            
            
            if problem_type == 'Classification':
                models=['Logistic Regression','KNN','SVM','DecisionTree','Random Forest','XgBoostClassifier']
                model = st.selectbox("Select  a model ",models)
                if model == 'Logistic Regression':
                    from sklearn.linear_model import LogisticRegression
                    classifier = LogisticRegression(random_state = 0)
                
                
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
                
                if model == 'DecisionTree':
                    from sklearn.tree import DecisionTreeClassifier
                    criterion = st.sidebar.selectbox("criterion",["gini","entropy"])
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    min_samples_leaf = st.sidebar.slider('min_samples_leaf', min_value=1, max_value=10, step=1)
                    classifier = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth ,min_samples_leaf=min_samples_leaf,random_state = 0)
                
                
                if model == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    criterion = st.sidebar.selectbox("criterion",["gini","entropy"])
                    n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=500 , step=1) 
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    classifier = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion, max_depth = max_depth , random_state = 0)
                
                if model == 'XgBoostClassifier':
                    from xgboost import XGBClassifier
                    n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=2000)
                    reg_lambda = st.sidebar.number_input('reg_lambda', min_value=0.01, max_value=5.00 , step=0.02)
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    colsample_bytree = st.sidebar.number_input('colsample_bytree', min_value=0.50, max_value=1.00 , step=0.05)
                    classifier = XGBClassifier(n_estimators=n_estimators,reg_lambda=reg_lambda,max_depth=max_depth,colsample_bytree=colsample_bytree)
            
                if st.button("Train"):
                    with st.spinner('model is training...'):
                        classifier.fit(X_train, y_train)
                    st.success('Model trained!')
                 
                    y_pred = classifier.predict(X_test)
                    from sklearn.metrics import accuracy_score
                    acc=accuracy_score(y_test, y_pred)
                    st.write('val_accuracy:',acc)
                    from sklearn.metrics import confusion_matrix , classification_report
                    st.write(classification_report(y_test, y_pred))
                    cm = confusion_matrix(y_test, y_pred)
                    st.markdown("**_confusion matrix_**")
                    st.write(cm)
                    y_pred = pd.DataFrame(y_pred)
                    st.dataframe(y_pred)
                    st.write(y_pred[0].value_counts())
                    st.write(y_pred[0].value_counts().plot(kind='bar'))
                    st.pyplot()
                    st.balloons()
                
                
                def download_model(model):
                    output_model = pickle.dumps(model)
                    st.write("model saved as output_model ")
                    b64 = base64.b64encode(output_model).decode()
                    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
                if st.button("save & Download model"):
                    download_model(classifier)
                
            if problem_type == 'Regression':  
                models=['Linear Regression', 'SVR','DecisionTree','Random Forest','XgBoostRegression']
                model = st.selectbox("Select  a model ",models)
            
                if model == 'Linear Regression':
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    
                if model== 'SVR':
                    from sklearn.svm import SVR
                    kernel_list = ['linear','poly','rbf','sigmoid']
                    kernel = st.sidebar.selectbox("P",kernel_list)
                    degree = st.sidebar.slider('Degree',min_value=1, max_value=10, step=1)
                    regressor = SVR(kernel = kernel , degree = degree )
                    
                if model== 'DecisionTree':
                    from sklearn.tree import DecisionTreeRegressor
                    criterion = st.sidebar.selectbox("criterion",["mse","friedman_mse","mae"])
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    min_samples_leaf = st.sidebar.slider('min_samples_leaf', min_value=1, max_value=10, step=1)
                    regressor = DecisionTreeRegressor(criterion = criterion, max_depth = max_depth ,min_samples_leaf=min_samples_leaf,random_state = 0)
                    
                if model== 'Random Forest':
                    from sklearn.ensemble import RandomForestRegressor
                    n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=500 , step=1)
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    criterion = st.sidebar.selectbox("criterion",["mse","mae"])
                    regressor = RandomForestRegressor(n_estimators = n_estimators,criterion = criterion, max_depth = max_depth , random_state = 0)
                    
                if model == 'XgBoostRegression':
                    from xgboost import XGBRegressor
                    n_estimators = st.sidebar.number_input('n_estimators', min_value=1, max_value=2000)
                    reg_lambda = st.sidebar.number_input('reg_lambda', min_value=0.01, max_value=5.00 , step=0.02)
                    max_depth = st.sidebar.slider('max_depth', min_value=1, max_value=10, step=1)
                    learning_rate = st.sidebar.number_input('learning_rate', min_value=0.01, max_value=5.00 , step=0.02)
                    booster = st.sidebar.selectbox("booster",["gbtree","gblinear","dart"])
                    colsample_bytree = st.sidebar.number_input('colsample_bytree', min_value=0.50, max_value=1.00 , step=0.05)
                    regressor = XGBRegressor(n_estimators=n_estimators,learning_rate=learning_rate,booster=booster,reg_lambda=reg_lambda,max_depth=max_depth,colsample_bytree=colsample_bytree)
                    
                if st.button("Train"):
                    with st.spinner('model is training...'):
                        regressor.fit(X_train, y_train)
                    st.success('Model trained!')
                    
                    y_pred = regressor.predict(X_test)
                    from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
                    mae_tr = mean_absolute_error(y_train,regressor.predict(X_train))
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mse_tr = mean_squared_error(y_train,regressor.predict(X_train))
                    r2 = r2_score(y_test, y_pred)
                    r2_tr = r2_score(y_train,regressor.predict(X_train))
                    st.write('mean absolute error:')
                    st.write('train:',mae_tr,'val:', mae )
                    st.write('mean squared error:')
                    st.write('train:',mse_tr,'val:', mse)
                    st.write('r2:')
                    st.write('train:',r2_tr,'val:', r2)
                    y_pred = pd.DataFrame(y_pred)
                    st.dataframe(y_pred)
                    st.balloons()
                    
                def download_model(model):
                    output_model = pickle.dumps(model)
                    st.write("model saved as output_model ")
                    b64 = base64.b64encode(output_model).decode()
                    href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                if st.button("save & Download model"):
                    download_model(regressor)
                     

if __name__ == '__main__':
    main()

