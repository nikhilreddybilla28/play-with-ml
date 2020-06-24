# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 08:49:48 2020

@author: nikil reddy
"""

import streamlit as st

import numpy as np
import pandas as pd

import sklearn
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 

import xgboost


def main():
	"""ML App made  with Streamlit """    
    
	activities = ["EDA &VIZ" , "Modelling"]	
	choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'EDA &VIZ':
        st.title('Play with ML')
        st.markdown('hey, wanna play with data & ML modles? Then upload a data here..  ')
        st.title('   ')
        st.title('   ')
        st.subheader("Exploratory Data Analysis & Vizualization ")
        
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		if data is not None:
            st.markdown('EDA')
			df = pd.read_csv(data)
            st.dataframe(df.head())
            st.write('shape:',df.shape)
        
			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)
                
            if st.checkbox("Null values"):
                st.write(df1.isnull().sum())
                
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
                
            st.markdown('Data Visualization')
            if st.checkbox("Show Value Counts"):
                column = st.selectbox("Select a Column",all_columns)
                st.write(df[column].value_counts().plot(kind='bar'))
                
                all_columns_names = df.columns.tolist()
                lot = st.selectbox("Select Type of Plot",["area","bar","pie","line","hist","box","kde","altair_chart"])
                selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)
                
                if st.button("Generate Plot"):
				    st.success("Generating   {} plot  for {}".format(type_of_plot,selected_columns_names))

                    if type_of_plot == 'area':
                        cust_data = df[ccolumns_names]
					    st.area_chart(cust_data)
                        
				    elif type_of_plot == 'bar':
					    cust_data = df[selected_columns_names]
					    st.bar_chart(cust_data)
                
                    elif type_of_plot == 'pie':
					    cust_data = df[selected_columns_names]
                        for i in range(len(selected_columns_names))
    					    pie = cust_data[:,i].plot(kind='pie')
					        st.write(cust_plot)
					        st.pyplot()
                    
    
				    elif type_of_plot == 'line':
					    cust_data = df[selected_columns_names]
					    st.line_chart(cust_data)
                    
                    elif type_of_plot == 'altair_chart':
					    a = st.selectbox("Select X axis",all_columns)
                        b = st.selectbox("Select Y axis",all_columns)
                        c = st.selectbox("Select a column ",all_columns)
                        c = alt.Chart(cust_data).mark_circle().encode(x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
					    st.altair_chart(c, use_container_width=True)

				# Custom Plot 
				    elif type_of_plot:
					    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					    st.write(cust_plot)
					    st.pyplot()
                
choice == 'Modelling':   
    pass
            
if __name__ == '__main__':
	main()
           
            
            
            
            
            
            
            
            
            
            
            
            
            
