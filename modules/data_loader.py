import pandas as pd 
import os
import streamlit as st


@st.cache_resource(show_spinner=False)

def load_data():
    """
    Loads the optimized parquet file from the data directory.
    Includes graceful error handling for missing files.
    """
    file_path = "https://huggingface.co/datasets/100rya-py/credit-risk-dashboard/resolve/main/loan_cleaned.parquet?download=true"
    drop_cols = [ #reducing noise
        'id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code', 
        'policy_code', 'pymnt_plan', 'application_type', 'initial_list_status',
        'funded_amnt', 'funded_amnt_inv', 'out_prncp', 'out_prncp_inv', 
        'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
        'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
        'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d'
    ]
    
    try:
        df = pd.read_parquet(file_path,engine='pyarrow')
        """
        Calculates the global baseline default rate of the portfolio.
        This acts as the 'Visual Anchor' for all our risk assessments.
        """
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

        #creating bad_loan column for statistical analysis
        bad_statuses = ['Charged Off',
                'Late (31-120 days)',
                'In Grace Period',
                'Late (16-30 days)',
                'Does not meet the credit policy. Status:Charged Off',
                'Default']
        df['bad_loan'] = df['loan_status'].apply(lambda x:1 if x in bad_statuses else 0)

        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype('category')
        
    
        float_cols = df.select_dtypes(include=['float64']).columns
        df[float_cols] = df[float_cols].astype('float32')
    
    
        int_cols = df.select_dtypes(include=['int64']).columns
        df[int_cols] = df[int_cols].astype('int32')
    
    
        return df
    except FileNotFoundError:
        st.error(f"Critical Error: The Dataset was not found at {file_path}. Please check your pipeline")
        st.stop()

