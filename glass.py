import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)



@st.cache()
def prediction(model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe):
    pred=model.predict([[RI,Na,Mg,Al,Si,K,Ca,Ba,Fe]])
    if pred[0]==1:
        return "building windows float processed"
    elif pred[0]==2:
        return "building windows non float processed"
    elif pred[0]==3:
        return "vehicle windows float processed"
    elif pred[0]==4:
        return "vehicle windows non float processed"
    elif pred[0]==5:
        return "containers"   
    elif pred[0]==6: 
        return "tableware"
    else:
        return "headlamp"
st.title('GLASS TYPE PREDICTION')
st.sidebar.header('DATA ANALYSIS')
if st.sidebar.checkbox('show data'):
    st.subheader('Dataset')
    st.dataframe(glass_df)
RI=st.sidebar.slider('RI',float(glass_df['RI'].min()),float(glass_df['RI'].max()))
Na=st.sidebar.slider('Na',float(glass_df['Na'].min()),float(glass_df['Na'].max()))
Mg=st.sidebar.slider('Mg',float(glass_df['Mg'].min()),float(glass_df['Mg'].max()))
Al=st.sidebar.slider('Al',float(glass_df['Al'].min()),float(glass_df['Al'].max()))
Si=st.sidebar.slider('Si',float(glass_df['Si'].min()),float(glass_df['Si'].max()))
K=st.sidebar.slider('K',float(glass_df['K'].min()),float(glass_df['K'].max()))
Ca=st.sidebar.slider('Ca',float(glass_df['Ca'].min()),float(glass_df['Ca'].max()))
Ba=st.sidebar.slider('Ba',float(glass_df['Ba'].min()),float(glass_df['Ba'].max()))
Fe=st.sidebar.slider('Fe',float(glass_df['Fe'].min()),float(glass_df['Fe'].max()))
select=st.sidebar.selectbox('CLASSIFIER',['RANDOM FOREST','LOGISTIC REG','SVC'])


if select=='RANDOM FOREST':
    xx=st.sidebar.number_input('Number of trees',1,1000)
    yy=st.sidebar.number_input('Max depth',1,1000)
    if st.sidebar.button('PREDICT'):
    
        rf_model = RandomForestClassifier(n_estimators=xx,max_depth=yy)
        
        rf_model.fit(X_train, y_train)
        score1 = rf_model.score(X_train, y_train)
        rf=prediction(rf_model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write('the predicted iris flower is:',rf,'-----The aaccuracy of this model is:',score1)
        plot_confusion_matrix(rf_model,X_test,y_test)
        st.pyplot()
elif select=='LOGISTIC REG':
    max_iter=st.sidebar.number_input('Iterations',1,1000)
    if st.sidebar.button('PREDICT'):
        lr_model = LogisticRegression(max_iter=max_iter)
        
        lr_model.fit(X_train, y_train)
        score2 = lr_model.score(X_train, y_train)
        lr=prediction(lr_model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write('the predicted iris flower is:',lr,'-----The aaccuracy of this model is:',score2)
        plot_confusion_matrix(lr_model,X_test,y_test)
        st.pyplot()
else:
    x=st.sidebar.radio('Kernal',['linear','rbf','poly'])
    y=st.sidebar.number_input('C',0,1000)
    z=st.sidebar.number_input('gamma',0,1000)
    if st.sidebar.button('PREDICT'):
        svc_model = SVC(kernel=x,C=y,gamma=z)

        svc_model.fit(X_train, y_train)
        score = svc_model.score(X_train, y_train)
        svc=prediction(svc_model,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe)
        st.write('the predicted iris flower is:',svc,'-----The aaccuracy of this model is:',score)
        plot_confusion_matrix(svc_model,X_test,y_test)
        st.pyplot()
st.sidebar.subheader('DATA VISUALISATION')
L=st.sidebar.multiselect('Select Chats',['Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot','scatter plot'])
if 'Correlation Heatmap' in L:
    plt.figure(figsize=(10,10))
    sns.heatmap(glass_df.corr(),anot=True)
    st.pyplot()
if 'Line Chart' in L:
    plt.figure(figsize=(15,5))
    st.line_chart(glass_df)
    st.pyplot()
if 'Area Chart' in L:
    plt.figure(figsize=(15,5))
    st.area_chart(glass_df)
    st.pyplot()

if 'Count Plot' in L:
    plt.figure(figsize=(10,10))
    sns.countplot(glass_df['GlassType'])
    st.pyplot()
if 'Pie Chart' in L:
    plt.figure(figsize=(10,10))
    plt.pie(glass_df['GlassType'].value_counts(),labels=glass_df['GlassType'].value_counts().index,autopct='%.2f%%')
    st.pyplot()
if 'Box Plot' in L:
    user=st.sidebar.selectbox('select a col',list(glass_df.columns))
    plt.figure(figsize=(15,5))
    sns.boxplot(glass_df['user'])
    st.pyplot()
if 'scatter plot' in L:
    user1=st.sidebar.multiselect('select columns',list(glass_df.columns))
    for i in user1:
        plt.figure(figsize=(10,10))
        plt.scatter(glass_df[i],glass_df['GlassType'])
        st.pyplot()