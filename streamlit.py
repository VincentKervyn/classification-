
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns#

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import scikitplot as skplt

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE

import pandas_profiling as pp

import streamlit as st
import streamlit.components.v1 as components
import time

#Loading the data
data = pd.read_csv('./data/BankChurners.csv')

####SIDE BAR####
# I want a side bar to select the step of the analysis
with st.sidebar:
    # st.set_page_config(layout="wide")
    st.set_page_config(layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 200px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 100px;
            margin-left: -20000px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title('Analysis step')
    st.sidebar.text('Please select the one\n of your interest\n and click expand')

##Buttons on side bar
    value_expand_select_explo = False
    if st.button(' Exploration Report'):
        value_expand_select_explo = True

    value_expand_select_explo_resume = False
    if st.button(' Exploration Resume'):
        value_expand_select_explo = True

    st.markdown("""---""")

    value_expand_select_model = False
    if st.button(' Model'):
        value_expand_select_model = True

    st.markdown("""---""")

    value_expand_select_result = False
    if st.button(' Results'):
        value_expand_select_result = True


####HEADER####
st.title('Churn prediction Dashboard')
st.header('The dashboard will present data analysis and model prediction')
st.subheader ('This project is based on ''Credit Card customers'' Kaggle dataset')
""" Source: https://www.kaggle.com/sakshigoyal7/credit-card-customers
"""
st.image('assets/dataset-cover.jpg')

"""An important financial institution is interested in analyzing its client database to increase the revenue generated from credit cardholders. They are concern about customers closing their bank accounts after accepting products from other institutions.

The churn rate is above 15% and increasing, the CEO urges the marketing team to start a marketing campaign for client retention."""

##### Exploration Report ####

expand_select_explo = st.expander(" Exploration report", expanded=value_expand_select_explo)
with expand_select_explo:
    """ We made a first data cleaning, removing obviously useless data.
    
    Under you will find the first data's report 
    (it could take some time to load)
    """


    #make good fellow people wait ===>TODO: implement loading bar and way to delay loading of the report after


    # Add a placeholder
   # latest_iteration = st.empty()
    #bar = st.progress(0)

    #for i in range(100):
        # Update the progress bar with each iteration.
     #   latest_iteration.text(f'Iteration {i + 1}')
      #  bar.progress(i + 1)
       # time.sleep(0.1)

    data2 = data.iloc[:, :-2]
    data2 = data2.iloc[:, 1:]
   # report = pp.ProfileReport(data2, title="Exploratory analysis").to_html()
    #components.html(report, height= 4000, width=600)
   # st.write(report.html, unsafe_allow_html=True)


#### Exploration resume####
expand_select_explo_resume = st.expander(" Exploration Resume", expanded=value_expand_select_explo_resume)
with expand_select_explo_resume:
    """ You will find here a resume of the data exploration
     
     LoremIpsum"""

    c_data = data2
    ### Plot
    fig1= make_subplots(rows=2, cols=1)

    tr1 = go.Box(x=data2['Customer_Age'], name='Age Box Plot', boxmean=True)
    tr2 = go.Histogram(x=data2['Customer_Age'], name='Age Histogram')

    fig1.add_trace(tr1, row=1, col=1)
    fig1.add_trace(tr2, row=2, col=1)

    fig1.update_xaxes(tick0=0.0, dtick=5)
    fig1.update_xaxes(range=[0, 75], row=1, col=1)
    fig1.update_xaxes(title_text="yaxis 2 title", range=[0, 75], row=2, col=1)

    fig1.update_xaxes(nticks=15)
    fig1.update_layout(height=700, width=1200, title_text="Distribution of Customer Ages")

    #### Plot 2 ==>TODO: explain et give more context
    """ LoremIpsum"""
    fig2 = px.histogram(c_data, x='Attrition_Flag', title='Proportion of churn (attrited) vs not churn customers',
                       color='Income_Category',
                       barmode='group',
                       category_orders={
                           'Income_Category': ['Less than 40K', '40K-', '60K-', '80K-', '120K+', 'Unknown']}
                       #                     text= val

                       )
    st.plotly_chart(fig1, use_container_width=True)

    st.plotly_chart(fig2, use_container_width=True)

    ###PLot TreeMap and table
    st.text('explain here why you look that ')
    Marital_st = ['Divorced', 'Divorced', 'Married', 'Married', 'Single', 'Single', 'Unknown', 'Unknown']
    Client_st = ['Existing', 'Attrited','Existing', 'Attrited','Existing', 'Attrited','Existing', 'Attrited']
    scores =    [627       ,  121      ,3978     ,709        ,3275      ,668        ,620      ,129]

    Attrit_count= pd.DataFrame(dict(Marital_st = Marital_st, Client_st =Client_st, scores= scores ))

    Attrit_count['all']= 'all'

    fig3 = px.treemap(Attrit_count,
        path = ['all','Client_st', "Marital_st", ],
        values = "scores"
    )

    fig3.update_traces(root_color="lightgrey", textinfo= 'label+value+percent parent+percent entry')
    fig3.update_layout(margin = dict(t=50, l=25, r=25, b=25)) title_text= 'Repartition marital status between churn'
    st.plotly_chart(fig3, use_container_width=True)


####Model####

expand_select_model = st.expander(" Model", expanded=value_expand_select_model)
with expand_select_model:
    """ To establish a prediction of the number of churn that coulb be expected, a Machine Learning Model is needed"""

#### Results ####
expand_select_result = st.expander(" Results", expanded=value_expand_select_result)
with expand_select_result:
    """ So what ? """