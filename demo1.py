import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
import lifetimes
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes import ParetoNBDFitter
from lifetimes.plotting import *

'''
# Customer Transaction Prediction

We will predict customer transactions using BG/NBD model. The basic assumption is that each customer has a probability of being "alive", which means this customer still remains
as a customer, or being "dead", which means this customer drops out and no longer is a customer. If the individual remains as a customer, the individual has a probability of making 
the transaction in the future.\n
The model thus is the integration of the following two processes: \n
Transaction Process:\n
    • While "alive", the customer number of transactions follows a Poisson distribution. \n
    • Heterogeneity in transaction rates across customers follows a Gamma distribution. \n
Latent Attrition Process: \n
    • After any transaction, a customer "dies" with a probability p. \n
    • Heterogeneity in death probabilities across customers follows a Beta distribution. \n

'''

'''
# Data Description
We will use customer transaction records from 01/01/2018 to 12/31/2018 as training data, predict customer transactions from 01/01/2019 to 6/10/2019 as testing data.
The individual leve transaction records look like the following:
'''

# Load Data:
cust_trans = pd.read_csv(r'cust_trans.csv')
CBS = pd.read_csv(r'CBS.csv')

# Show/Hide Data:
show_hide = st.selectbox('Show/Hide Transaction Data',('Show Data Example', 'Show Number of Transaction Records'))
if show_hide == ('Show Data Example'):
    st.markdown("**Showing raw data---->>>**")	
    st.write(cust_trans[['CustID', 'TransDate']].head(20))
else: 
    st.markdown("**Hiding raw data---->>>**")
    st.write('Number of Transaction Records: ' + str(len(cust_trans)))

# Simulate calibration period transactions:
@st.cache
def simulate_calibration():
    model = BetaGeoFitter()
    bgnbd_model = model.fit(CBS['frequency_cal'], CBS['recency_cal'], CBS['T_cal'])
    simulated_bgnbd = bgnbd_model.generate_new_data(size = len(CBS))
    return bgnbd_model, simulated_bgnbd

bgnbd_model, simulated_bgnbd = simulate_calibration()

'''
# Train Model & Visualize Fittings.
Click the button below to build a BG/NBD model on training data.
'''

# Train data button:
if st.button('Simulate Transactions on Training Data'): 
    st.write('Simulation Complete!')

# Function to generate data frame with actual frequency and predicted freqency at holdout period 
# from original individual-level calibration and holdout summary data (customer by sufficient statistics, or CBS):
def generate_actual_pred_holdout_bgnbd(df):
    df['predict_holdout_bgnbd'] = bgnbd_model.predict(df['duration_holdout'], df['frequency_cal'],df['recency_cal'],df['T_cal'])
    df = df.reset_index()
    prediction = df.loc[:,['CustID','frequency_holdout','predict_holdout_bgnbd']]
    prediction['predict_holdout_bgnbd'] = prediction['predict_holdout_bgnbd'].fillna(0)
    prediction['predict_round_bgnbd'] = prediction['predict_holdout_bgnbd'].astype(int)
    return prediction

# Functions to transform data:
def transform_cal_hold_to_freq_count(df):
    return pd.DataFrame(df['frequency_cal'].value_counts().sort_index()).reset_index().rename(columns = {'index':'freq','frequency_cal':'count'})
def transform_sim_to_freq_count(df):
    return pd.DataFrame(df['frequency'].value_counts().sort_index().reset_index().rename(columns = {'index': 'freq', 'frequency': 'count'}))
def binned_max_freq(df,max_x):
    return df.loc[0:max_x].append({'freq':str(max_x) + '+','count':df[df.iloc[:,0] >= max_x].iloc[:,1].sum()},ignore_index = True)

# Visualize Model Fit:
fit_viz = st.selectbox('Select Visualization', ('Simulation on Training Data', 'Simulation on Testing Data'))
def create_fit_viz(fit_viz):
    if fit_viz == 'Simulation on Training Data':
        '''
        This visualiztion shows the transaction distributions in training period based on model output.
        '''
        actual_cal_freq = transform_cal_hold_to_freq_count(CBS)
        simulated_cal_freq_bgnbd = transform_sim_to_freq_count(simulated_bgnbd)
        actual_binned = binned_max_freq(actual_cal_freq, 20)
        simulated_binned_bgnbd = binned_max_freq(simulated_cal_freq_bgnbd, 20)
        cal_actual_sim_bgnbd = actual_binned.merge(simulated_binned_bgnbd, how = 'outer', on = 'freq')
        cal_actual_sim_bgnbd = cal_actual_sim_bgnbd.rename(columns ={'freq':'transactions' ,'count_x':'actual_counts','count_y':"simulated_counts"})
        cal_actual_sim_bgnbd['transactions'] = cal_actual_sim_bgnbd['transactions'].astype(str)
        bgnbd_actual_pred_cal_plot = cal_actual_sim_bgnbd.plot(kind = 'bar', figsize = (8,6))
        bgnbd_actual_pred_cal_plot.set_title('Count of Actual vs. Predicted Transactions in Training Period',fontweight = "bold", pad = 20)
        bgnbd_actual_pred_cal_plot.legend(['Actual','Simulated'])
        bgnbd_actual_pred_cal_plot.set_xticklabels(cal_actual_sim_bgnbd['transactions'])[0]
        bgnbd_actual_pred_cal_plot.set_xlabel('Number of Transactions in Training Period')
        bgnbd_actual_pred_cal_plot.set_ylabel('Number of Customers')
        return st.pyplot()
    else:
        '''
        This visualiztion shows the transaction distributions in testing period based on model output.
        '''
        predict_holdout_bgnbd = generate_actual_pred_holdout_bgnbd(CBS)
        holdout_actual_bgnbd = pd.DataFrame(predict_holdout_bgnbd["frequency_holdout"].value_counts().sort_index()).reset_index()\
                                 .rename(columns = {'index':'freq','frequency_holdout':'count'})
        holdout_predict_bgnbd = pd.DataFrame(predict_holdout_bgnbd["predict_round_bgnbd"].value_counts().sort_index())\
                                  .reset_index().rename(columns = {'index':'freq','predict_round_bgnbd':'count'})
        hd_actual_bgnbd_binned = binned_max_freq(holdout_actual_bgnbd, 20)
        hd_predict_bgnbd_binned = binned_max_freq(holdout_predict_bgnbd, 20)
        hold_actual_pred_bgnbd = hd_actual_bgnbd_binned.merge(hd_predict_bgnbd_binned, how = 'outer', on = 'freq')
        hold_actual_pred_bgnbd = hold_actual_pred_bgnbd.rename(columns ={'freq':'transactions' ,'count_x':'actual_counts','count_y':"predicted_counts"})
        bgnbd_actual_pred_hold_plot = hold_actual_pred_bgnbd.plot(kind = 'bar', figsize = (8,6))
        bgnbd_actual_pred_hold_plot.set_title('Count of Actual vs. Predicted No. of Transactions in Testing Period',\
                                      fontweight = "bold", pad = 20)
        bgnbd_actual_pred_hold_plot.legend(['Actual','Predicted'])
        bgnbd_actual_pred_hold_plot.set_xticklabels(hold_actual_pred_bgnbd['transactions'])[0]
        bgnbd_actual_pred_hold_plot.set_xlabel('Number of Transactions in Testing Period')
        bgnbd_actual_pred_hold_plot.set_ylabel('Number of Customers')
        return st.pyplot()


viz = create_fit_viz(fit_viz)

# Plot individual custoemr dropout probability at the end of calibration:
# cust_id = st.text_input('Input Customer ID')

'# Individual Level Visualization'
indi_viz_type = st.selectbox('Select Visualization Type', ('History Alive Probability', 'Recency Frequency Matrix'))
def individual_plot(indi_viz_type):
    if indi_viz_type == 'History Alive Probability':
        '''
        This plot tells us the historical alive probability for a customer we have observed in training period. 
        The red dashed line is the date that the customer makes a transaction.
        The blue line shows the probability variations after each transaction.\n
        Enter a valid customer ID to generate the plot.
        '''
        cust_id = st.number_input('Enter Customer ID')
        if cust_id not in cust_trans['CustID'].unique():
            return st.write('Customer Not Found...Please Enter Valid Customer ID.')
        else:
            fig = plt.figure(figsize=(10,10))
            days_since_birth = int(CBS[CBS['CustID'] == cust_id]['T_cal'])
            sp_trans = cust_trans[cust_trans['TransDate'] <= '2018-12-31'].loc[cust_trans['CustID'] == cust_id]
            viz = plot_history_alive(bgnbd_model, days_since_birth, sp_trans, 'TransDate')
            return st.pyplot()
    else: 
        '''
        This heat map shows the distribution of number of transactions made by a customer in prediction period, given this customer's number of transactions have already made 
        (frequency), and the time most recent transaction is made (recency). \n
        We call this the "Recency-Frequency Matrix". \n
        This plot hslps us understand a customer's behavior in future periods based on the customer's previous purhcasing behavior.\n
        Enter parameters in the box below to generate the plot.
        '''
        future_period = st.number_input('Enter Prediction Period (Days)', step = 1.0)
        max_recency = st.number_input('Enter Number of Days Since First Purchase', step = 1.0)
        d = pd.DataFrame(np.zeros((int(max_recency) + 1, int(max_recency) + 1)))
        for i in range(1, int(max_recency) + 1):
            for j in range(i+1):
                d.at[j,i] = bgnbd_model.predict(future_period, j, i, max_recency)
        d = d.iloc[:,1:]
        fig, ax = plt.subplots(figsize=(14,10))
        exp_future_trans = sns.heatmap(d.T,cmap = 'YlOrBr', ax = ax)
        exp_future_trans.set_title('Transactions in '+ str(future_period) + ' Days '  + 'Given Recency at Day ' + str(max_recency),\
                                    fontweight = "bold", pad = 20)
        exp_future_trans.set_xlabel("Customer's Historical Frquency")
        exp_future_trans.set_ylabel("Customer's Recency")
        return st.pyplot()

try:
    individual_plot(indi_viz_type)
except ValueError:
    pass

# Generate List of Customers Likely to Dropout:
st.title('Find Customers Likely to Dropout')
'''
Using the model, we can find the customers who are likely to drop out at the end of training period. 
This can help the business identify potential customer loses and take action to retain these customers.
Enter the drop out proabilities below to find the potential customer loses.
'''
min = st.slider('Min.Dropout Proability', min_value = 0.0, max_value = 1.0, step = 0.05)
max = st.slider('Max. Dropout Proability', min_value = 0.0, max_value = 1.0, step = 0.05)
def prob_alive(min, max):
    df = CBS[['CustID', 'frequency_cal','recency_cal', 'T_cal']]
    if min < max:
        df['p_alive'] = bgnbd_model.conditional_probability_alive(df['frequency_cal'], df['recency_cal'], df['T_cal'])
        cust_list = df[(df['p_alive'] >= min)&(df['p_alive'] <= max)]['CustID']
        return st.write(cust_list)
    else:
        return st.write('Selection invalid. Minimum dropout probability must be smaller than maximum dropout probability.')

prob_alive(min, max)