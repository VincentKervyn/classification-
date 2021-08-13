
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
from streamlit_pandas_profiling import st_profile_report
import streamlit as st
import streamlit.components.v1 as components
import time

#Loading the data
data = pd.read_csv('BankChurners.csv')

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

    value_expand_select_preprocess = False
    if st.button(' Preprocessing'):
        value_expand_select_result = True
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
    
    pr = data2.profile_report()
    st_profile_report(pr)
    
#     prof = pp.ProfileReport(data2, explorative=True, minimal=True)
#     output = prof.to_file(output_file="output_min.html", silent=False)
    
#     report = pp.ProfileReport(data2, title="Exploratory analysis").to_html()
#     components.html(report, height= 4000, width=600)
#     st.write(report.html, unsafe_allow_html=True)  


#### Exploration resume####
expand_select_explo_resume = st.expander(" Exploration Resume", expanded=value_expand_select_explo_resume)
with expand_select_explo_resume:
    """ You will find here a resume of the data exploration
     
     LoremIpsum""" 
#     ==>TODO: fill lorem Ipsum

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
    fig3.update_layout(margin = dict(t=50, l=25, r=25, b=25),title_text= 'Repartition marital status between churn')
    st.plotly_chart(fig3, use_container_width=True)



###Preprocessing####
expand_select_preprocess = st.expander(" Preprocessing", expanded=value_expand_select_model)
with expand_select_preprocess:
    """ LorempIpsum"""
##Plot
    data2.Attrition_Flag = data2.Attrition_Flag.replace({'Attrited Customer':1,'Existing Customer':0})
    data2.Gender = data2.Gender.replace({'F':1,'M':0})
    data2 = pd.concat([data2,pd.get_dummies(data2['Education_Level'], prefix = 'Education_')],axis=1)
    data2 = pd.concat([data2,pd.get_dummies(data2['Income_Category'], prefix = 'Income_')],axis=1)
    data2 = pd.concat([data2,pd.get_dummies(data2['Marital_Status'], prefix = 'Marital_')],axis=1)
    data2 = pd.concat([data2,pd.get_dummies(data2['Card_Category'])],axis=1)
    data2.drop(columns = ['Education_Level','Income_Category','Marital_Status','Card_Category'],inplace=True)

    fig4 = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion',  'Spearman Correaltion'))
    colorscale=     [[1.0              , "rgb(165,0,38)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.0               , "rgb(49,54,149)"]]

    s_val =data2.corr('pearson')
    mask = np.zeros_like(s_val, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan
    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig4.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=1,ygap=1,colorscale=colorscale),
        row=1, col=1
    )


    s_val = data2.corr('spearman')
    mask = np.zeros_like(s_val, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan

    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig4.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale),
        row=2, col=1
    )
    fig4.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=10,
            font_family="Rockwell"
        )
    )
    fig4.update_layout(height=1800, width=900, title_text="Correlation ")
    st.plotly_chart(fig4, use_container_width=True)


    st.header('Upsampling')
    """Loremp Ipsum sum sum """
    oversample = SMOTE()
    X, y = oversample.fit_resample(data2[data2.columns[1:]], data2[data2.columns[0]])
    usampled_df = X.assign(Churn = y)

    ohe_data =usampled_df[usampled_df.columns[15:-1]].copy()
    ohe_data.head()
    usampled_df = usampled_df.drop(columns=usampled_df.columns[15:-1])

    ## Plot post up sample


    fig5 = make_subplots(rows=2, cols=1,shared_xaxes=True,subplot_titles=('Perason Correaltion over sample',
                                                                         'Spearman Correaltion over sample'))
    colorscale=     [[1.0              , "rgb(165,0,38)"],
                    [0.8888888888888888, "rgb(215,48,39)"],
                    [0.7777777777777778, "rgb(244,109,67)"],
                    [0.6666666666666666, "rgb(253,174,97)"],
                    [0.5555555555555556, "rgb(254,224,144)"],
                    [0.4444444444444444, "rgb(224,243,248)"],
                    [0.3333333333333333, "rgb(171,217,233)"],
                    [0.2222222222222222, "rgb(116,173,209)"],
                    [0.1111111111111111, "rgb(69,117,180)"],
                    [0.0               , "rgb(49,54,149)"]]

    s_val =usampled_df.corr('pearson')
    mask = np.zeros_like(s_val, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan

    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig5.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,name='pearson',showscale=False,xgap=1,ygap=1,colorscale=colorscale),
        row=1, col=1
    )


    s_val = usampled_df.corr('spearman')
    mask = np.zeros_like(s_val, dtype = bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan

    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig5.add_trace(
        go.Heatmap(x=s_col,y=s_idx,z=s_val,xgap=1,ygap=1,colorscale=colorscale),
        row=2, col=1
    )
    fig5.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )


    fig5.update_layout(height=900, width=900, title_text="Upsmapled Correlations")
    st.plotly_chart(fig5, use_container_width=True)

    st.header('Principal Component Analysis')
    """" Lorem Ipsum"""

    N_COMPONENTS = 4

    pca_model = PCA(n_components=N_COMPONENTS)

    pc_matrix = pca_model.fit_transform(ohe_data)
    evr = pca_model.explained_variance_ratio_
    total_var = evr.sum() * 100
    cumsum_evr = np.cumsum(evr)

    trace1 = {
        "name": "individual explained variance",
        "type": "bar",
        'y': evr}
    trace2 = {
        "name": "cumulative explained variance",
        "type": "scatter",
        'y': cumsum_evr}
    data = [trace1, trace2]
    layout = {
        "xaxis": {"title": "Principal components"},
        "yaxis": {"title": "Explained variance ratio"},
    }
    fig6 = go.Figure(data=data, layout=layout)
    fig6.update_layout(title='Explained Variance Using {} Dimensions'.format(N_COMPONENTS))
    st.plotly_chart(fig6, use_container_width=True)

    ###Truc
    st.header( 'Correlation post upsampling' )
    usampled_df_with_pcs = pd.concat([usampled_df,
                                      pd.DataFrame(pc_matrix,
                                                   columns=['PC-{}'.format(i) for i in range(0, N_COMPONENTS)])],
                                     axis=1)

    fig7 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Perason Correaltion', 'Spearman Correaltion'))

    s_val = usampled_df_with_pcs.corr('pearson')

    mask = np.zeros_like(s_val, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan

    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig7.add_trace(
        go.Heatmap(x=s_col, y=s_idx, z=s_val, name='pearson', showscale=False, xgap=1, ygap=1, colorscale=colorscale),
        row=1, col=1
    )

    s_val = usampled_df_with_pcs.corr('spearman')

    mask = np.zeros_like(s_val, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    s_val[mask] = np.nan

    s_idx = s_val.index
    s_col = s_val.columns
    s_val = s_val.values
    fig7.add_trace(
        go.Heatmap(x=s_col, y=s_idx, z=s_val, xgap=1, ygap=1, colorscale=colorscale),
        row=2, col=1
    )
    fig7.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
    fig7.update_layout(height=700, width=900, title_text="Upsmapled Correlations With PC\'s")
    st.plotly_chart(fig7, use_container_width=True)


####Model####
expand_select_model = st.expander(" Model", expanded=value_expand_select_model)
with expand_select_model:
    """ To establish a prediction of the number of churn that coulb be expected, a Machine Learning Model is needed"""

    st.header('Model Selection ')
###SPLIT#
    X_features = ['Total_Trans_Ct', 'PC-3', 'PC-1', 'PC-0', 'PC-2', 'Total_Ct_Chng_Q4_Q1', 'Total_Relationship_Count']

    X = usampled_df_with_pcs[X_features]
    y = usampled_df_with_pcs['Churn']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=20)

    rf_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", RandomForestClassifier(random_state=20))])
    ada_pipe = Pipeline(
        steps=[('scale', StandardScaler()), ("RF", AdaBoostClassifier(random_state=20, learning_rate=0.7))])
    svm_pipe = Pipeline(steps=[('scale', StandardScaler()), ("RF", SVC(random_state=20, kernel='rbf'))])

    f1_cross_val_scores = cross_val_score(rf_pipe, train_x, train_y, cv=10, scoring='f1')
    ada_f1_cross_val_scores = cross_val_score(ada_pipe, train_x, train_y, cv=10, scoring='f1')
    svm_f1_cross_val_scores = cross_val_score(svm_pipe, train_x, train_y, cv=10, scoring='f1')

    fig8 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=('Random Forest Cross Val Scores',
                                                                           'Adaboost Cross Val Scores',
                                                                           'SVM Cross Val Scores'))

    fig8.add_trace(
        go.Scatter(x=list(range(0, len(f1_cross_val_scores))), y=f1_cross_val_scores, name='Random Forest'),
        row=1, col=1
    )
    fig8.add_trace(
        go.Scatter(x=list(range(0, len(ada_f1_cross_val_scores))), y=ada_f1_cross_val_scores, name='Adaboost'),
        row=2, col=1
    )
    fig8.add_trace(
        go.Scatter(x=list(range(0, len(svm_f1_cross_val_scores))), y=svm_f1_cross_val_scores, name='SVM'),
        row=3, col=1
    )

    fig8.update_layout(height=700, width=900, title_text="Different Model 10 Fold Cross Validation")
    fig8.update_yaxes(title_text="F1 Score")
    fig8.update_xaxes(title_text="Fold #")
    st.plotly_chart(fig8, use_container_width=True)

    ## Model Evaluation
    st.header('Model Evalutation ')
    rf_pipe.fit(train_x, train_y)
    rf_prediction = rf_pipe.predict(test_x)

    ada_pipe.fit(train_x, train_y)
    ada_prediction = ada_pipe.predict(test_x)

    svm_pipe.fit(train_x, train_y)
    svm_prediction = svm_pipe.predict(test_x)

    fig9 = go.Figure(data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Test Data<b>'],
                                               line_color='darkslategray',
                                               fill_color='whitesmoke',
                                               align=['center', 'center'],
                                               font=dict(color='black', size=18),
                                               height=40),

                                   cells=dict(values=[['<b>Random Forest<b>', '<b>AdaBoost<b>', '<b>SVM<b>'],
                                                      [np.round(f1(rf_prediction, test_y), 2),
                                                       np.round(f1(ada_prediction, test_y), 2),
                                                       np.round(f1(svm_prediction, test_y), 2)]]))
                          ])

    fig9.update_layout(title='Model Results On Test Data')
    st.plotly_chart(fig9, use_container_width=True)

    ## Model Evaluation On Original Data (Before Upsampling)
    st.header('Model Evaluation On Original Data (Before Upsampling)')
    ohe_data = c_data[c_data.columns[16:]].copy()
    pc_matrix = pca_model.fit_transform(ohe_data)
    original_df_with_pcs = pd.concat(
        [c_data, pd.DataFrame(pc_matrix, columns=['PC-{}'.format(i) for i in range(0, N_COMPONENTS)])], axis=1)

    unsampled_data_prediction_RF = rf_pipe.predict(original_df_with_pcs[X_features])
    unsampled_data_prediction_ADA = ada_pipe.predict(original_df_with_pcs[X_features])
    unsampled_data_prediction_SVM = svm_pipe.predict(original_df_with_pcs[X_features])

    fig10 = go.Figure(
        data=[go.Table(header=dict(values=['<b>Model<b>', '<b>F1 Score On Original Data (Before Upsampling)<b>'],
                                   line_color='darkslategray',
                                   fill_color='whitesmoke',
                                   align=['center', 'center'],
                                   font=dict(color='black', size=18),
                                   height=40),

                       cells=dict(values=[['<b>Random Forest<b>', '<b>AdaBoost<b>', '<b>SVM<b>'], [
                           np.round(f1(unsampled_data_prediction_RF, original_df_with_pcs['Attrition_Flag']), 2),
                           np.round(f1(unsampled_data_prediction_ADA, original_df_with_pcs['Attrition_Flag']), 2),
                           np.round(f1(unsampled_data_prediction_SVM, original_df_with_pcs['Attrition_Flag']), 2)]]))
              ])

    fig10.update_layout(title='Model Result On Original Data (Without Upsampling)')
    st.plotly_chart(fig10, use_container_width=True)

#### Results ####
expand_select_result = st.expander(" Results", expanded=value_expand_select_result)
with expand_select_result:
    """ So what ? """

    z = confusion_matrix(unsampled_data_prediction_RF, original_df_with_pcs['Attrition_Flag'])
    fig11 = ff.create_annotated_heatmap(z, x=['Not Churn', 'Churn'], y=['Predicted Not Churn', 'Predicted Churn'],
                                      colorscale='gnbu', xgap=3, ygap=3)
    fig11['data'][0]['showscale'] = True
    fig11.update_layout(title='Prediction On Original Data With Random Forest Model Confusion Matrix')
    st.plotly_chart(fig11, use_container_width=True)

    unsampled_data_prediction_RF = rf_pipe.predict_proba(original_df_with_pcs[X_features])
    skplt.metrics.plot_precision_recall(original_df_with_pcs['Attrition_Flag'], unsampled_data_prediction_RF)
    plt.legend(prop={'size': 20})
    st.image('assets/curve.JPG')


