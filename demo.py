import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lifetimes
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import *

st.title('Customer Churn Analysis Using Lifetime Value Theory')
@st.cache
def load_data(path):
    return(pd.read_csv(path))
cust_trans = load_data(r'C:\Users\Matt\Desktop\Streamlit\cust_trans.csv')
CBS = load_data(r'C:\Users\Matt\Desktop\Streamlit\CBS.csv')

# Show/Hide Data:
show_hide = st.sidebar.selectbox('Show/Hide Transaction Data',('Show Data Example', 'Show Number of Transaction Records'))
if show_hide == ('Show Data Example'):
    st.subheader("Showing raw data---->>>")	
    st.write(cust_trans[['CustID', 'TransDate']].head(20))
else: st.write('Number of Transaction Records: ' + str(len(cust_trans)))

# Select Model:
option = st.sidebar.selectbox('Select Model', ['BG/NBD'])
def choose_model(option):
    if option == ('BG/NBD'):
        model = BetaGeoFitter()
    return model

# Train Model:
def transform_cal_hold_to_freq_count(df):
    return pd.DataFrame(df['frequency_cal'].value_counts().sort_index()).reset_index().rename(columns = {'index':'freq','frequency_cal':'count'})

def transform_sim_to_freq_count(df):
    return pd.DataFrame(df['frequency'].value_counts().sort_index().reset_index().rename(columns = {'index': 'freq', 'frequency': 'count'}))

def binned_max_freq(df,max_x):
    return df.loc[0:max_x].append({'freq':str(max_x) + '+','count':df[df.iloc[:,0] >= max_x].iloc[:,1].sum()},ignore_index = True)

@st.cache(suppress_st_warning=True)
def train_model(option):
    if st.button('Train Model'):
        model = choose_model(option)
        bgnbd_model = model.fit(CBS['frequency_cal'], CBS['recency_cal'], CBS['T_cal'])
        simulated_bgnbd = model.generate_new_data(size = len(CBS))
        st.write('Model Training Complete!')   
    else: 
        bgnbd_model = None
        simulated_bgnbd = None
    return bgnbd_model, simulated_bgnbd

bgnbd_model, simulated_bgnbd = train_model(option)

def plot_calibration_fit(option):
    if st.button('Plot Model Fit'):
        if bgnbd_model == None:
            return st.write('Please Train Model First!')
        else: 
            actual_cal_freq = transform_cal_hold_to_freq_count(CBS)
            simulated_cal_freq_bgnbd = transform_sim_to_freq_count(simulated_bgnbd)
            actual_binned = binned_max_freq(actual_cal_freq, 20)
            simulated_binned_bgnbd = binned_max_freq(simulated_cal_freq_bgnbd, 20)
            cal_actual_sim_bgnbd = actual_binned.merge(simulated_binned_bgnbd, how = 'outer', on = 'freq')
            cal_actual_sim_bgnbd = cal_actual_sim_bgnbd.rename(columns ={'freq':'transactions' ,'count_x':'actual_counts','count_y':"simulated_counts"})
            cal_actual_sim_bgnbd['transactions'] = cal_actual_sim_bgnbd['transactions'].astype(str)
            bgnbd_actual_pred_cal_plot = cal_actual_sim_bgnbd.plot(kind = 'bar', figsize = (8,6))
            bgnbd_actual_pred_cal_plot.set_title('Count of Actual vs. Predicted Transactions in Calibration Period',fontweight = "bold", pad = 20)
            bgnbd_actual_pred_cal_plot.legend(['Actual','Simulated'])
            bgnbd_actual_pred_cal_plot.set_xticklabels(cal_actual_sim_bgnbd['transactions'])[0]
            bgnbd_actual_pred_cal_plot.set_xlabel('Number of Transactions in Calibration')
            bgnbd_actual_pred_cal_plot.set_ylabel('Number of Customers')
            return(st.pyplot())
    else:
        return st.write('')

calibration_plot = plot_calibration_fit(option)

#actual_cal_freq = transform_cal_hold_to_freq_count(CBS)
#simulated_cal_freq_bgnbd = transform_sim_to_freq_count(sim_cal[option])







        #actual_binned = binned_max_freq(actual_cal_freq, 20)
        #simulated_binned_bgnbd = binned_max_freq(simulated_cal_freq_bgnbd, 20)
        #cal_actual_sim_bgnbd = actual_binned.merge(simulated_binned_bgnbd, how = 'outer', on = 'freq')
        #cal_actual_sim_bgnbd = cal_actual_sim_bgnbd.rename(columns ={'freq':'transactions' ,'count_x':'actual_counts','count_y':"simulated_counts"})
        #cal_actual_sim_bgnbd['transactions'] = cal_actual_sim_bgnbd['transactions'].astype(str)
        #bgnbd_actual_pred_cal_plot = cal_actual_sim_bgnbd.plot(kind = 'bar', figsize = (8,6))
        #bgnbd_actual_pred_cal_plot.set_title('Count of Actual vs. Predicted Transactions in Calibration Period',fontweight = "bold", pad = 20)
        #bgnbd_actual_pred_cal_plot.legend(['Actual','Simulated'])
        #bgnbd_actual_pred_cal_plot.set_xticklabels(cal_actual_sim_bgnbd['transactions'])[0]
        #bgnbd_actual_pred_cal_plot.set_xlabel('Number of Transactions in Calibration')
        #bgnbd_actual_pred_cal_plot.set_ylabel('Number of Customers')
        #st.pyplot()