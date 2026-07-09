
import pandas as pd 
import numpy as np
import streamlit as st
import gc

@st.cache_data(show_spinner=False)
def get_baseline_default_rate(df,target_col='bad_loan'):
    

    baseline_rate = df[target_col].mean()
    return baseline_rate

@st.cache_data(show_spinner=False)
def calculate_woe_iv(df, feature, target='bad_loan'):
    """
    Transforms a categorical variable into Weight of Evidence (WoE) scores 
    and computes the Total Information Value (IV) using vectorized operations.
    """
   
    subset = df[[feature, target]].copy()
    
    
    stats = subset.groupby(feature, observed=True)[target].agg([("Total", "count"), ("Bad", "sum")])
    
    stats['Good'] = stats['Total'] - stats['Bad']
      
    stats['Dist_Bad'] = stats['Bad'] / stats['Bad'].sum()
    stats['Dist_Good'] = stats['Good'] / stats['Good'].sum()
    
    stats['WoE'] = np.log((stats['Dist_Good'] + 0.001) / (stats['Dist_Bad'] + 0.001))
    stats['IV_Contribution'] = (stats['Dist_Good'] - stats['Dist_Bad']) * stats['WoE']
    
    woe_df = stats.sort_values(by='WoE', ascending=False).reset_index()
    total_iv = woe_df['IV_Contribution'].sum()
    
    del subset, stats
    gc.collect()
    
    return woe_df, total_iv


