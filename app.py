import pandas as pd
import streamlit as st
from modules.data_loader import load_data
from modules.stats_engine import get_baseline_default_rate, calculate_woe_iv
import plotly.express as px
import plotly.graph_objects as go


#Page Configuration
st.set_page_config(page_title="Credit Risk Scorecard",initial_sidebar_state="expanded",layout="wide")

#data ingestio and state initialization
if 'engine_initialized' not in st.session_state:
    with st.spinner("Initializing Risk Engine........."):
        st.session_state.df = load_data()
        st.session_state.baseline_rate = get_baseline_default_rate(st.session_state.df)
        st.session_state.engine_initialized = True

df = st.session_state.df
baseline_rate = st.session_state.baseline_rate

#sidebar setup
with st.sidebar:
    st.title("Model Controls")
    st.markdown("Adjust these parameters below to test the risk engine")
    st.info("System Status: Online")
    st.caption(f"Rows Loaded in Memory: {len(df):,}")
    with st.expander("Category Slicers", expanded=False,width=325):
            with st.form("tab1_slicers"):
                s_col1, s_col2 = st.columns(2)
               
                grade_options = sorted(df['grade'].dropna().unique().tolist()) if 'grade' in df.columns else ['A', 'B', 'C', 'D']
                selected_grades = st.multiselect("Risk Grades", options=grade_options, default=grade_options)
                home_options = ["All"] + df['home_ownership'].dropna().unique().tolist() if 'home_ownership' in df.columns else ["All", "RENT", "OWN", "MORTGAGE"]
                selected_home = st.selectbox("Home Ownership", options=home_options)
                
                apply_filters = st.form_submit_button("Apply Filters", type="primary")
    if apply_filters:
            mask = pd.Series(True, index=df.index)
            if selected_grades:
                mask &= (df['grade'].isin(selected_grades))
            if selected_home != "All":
                mask &= (df['home_ownership'] == selected_home)
            display_df = df[mask]
    else:
        display_df = df

#main dashboard layout or architecture  

st.title("Credit Risk Scoring and Analysis Engine",text_alignment='center')
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Executive Overview"
tab1,tab2,tab3 = st.tabs(["Executive Overview","Statistical Analysis","Live Risk Calculator"],on_change='rerun',key='active_tab')

if st.session_state["active_tab"] == "Executive Overview": # to prevent the remaining tabs from loading until the first tab is selected
    with tab1:
        st.header("Executive Risk Overview")
        col1,col2,col3 = st.columns(3)
        st.markdown(
        """
        <style>
        /* Targeting the metric value */
        [data-testid="stMetricValue"] {
            text-align: center;
        }
        /* Targeting the metric label */
        [data-testid="stMetricLabel"] {
            text-align: center;
            /* Optional: Makes the label slightly more prominent */
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

        with col1:
            st.metric("Total Applications",value =f"{len(display_df):,}")
        with col2:
            total_exposure = display_df['loan_amnt'].sum() if 'loan_amnt' in display_df.columns else 0
            st.metric("Total Exposure $",value=f"{total_exposure:,}")
        
        with col3:
            bad_loans = display_df['bad_loan'].sum() if 'bad_loan' in display_df.columns else 0
            st.metric("Total Bad Loans",value = f"{bad_loans:,}")
        
    
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = baseline_rate * 100,
            number = {'suffix': "%", 'valueformat': ".2f"},
            title = {'text': "System Baseline Risk"},
            gauge = {
                'axis': {'range': [None, 30]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgreen"},
                    {'range': [10, 15], 'color': "gold"},
                    {'range': [15, 30], 'color': "crimson"}
                ],
            }
        ))
        gauge_fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=5), paper_bgcolor='rgba(0,0,0,0)', font={'color': "black"})
        st.plotly_chart(gauge_fig, use_container_width=True)    
        st.subheader("Portfolio Distribution: Good V/s Bad Loans")
        status_counts = display_df['bad_loan'].value_counts().reset_index()
        status_counts.columns = ['Loan_Status', 'Count']
        status_counts['Loan_Status'] = status_counts['Loan_Status'].map({0: 'Good Loans', 1: 'Bad Loans'})

        fig = px.bar(status_counts,x='Loan_Status',y='Count',color="Loan_Status",
                     color_discrete_map={'Good Loans': '#1E3A8A', 'Bad Loans': '#E11D48'},text_auto=True)

        fig.update_traces(marker_line_width=1.5,opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>")

        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',showlegend=False,
            transition_duration=500,hovermode="x unified")

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Time Series Trend: Default Rate Over Time")

        with st.spinner("Calculating Temporal Trend"):
            trend_df = display_df.groupby("issue_d").agg(Default_Rate = ("bad_loan","mean")).reset_index()
            trend_df.sort_values("issue_d", inplace=True)

            fig_trend = px.line(trend_df,x='issue_d',y='Default_Rate',markers=True,
                    line_shape='spline')

            fig_trend.update_traces(line_color='#1E3A8A', line_width=3,
                    marker=dict(size=8, color='#E11D48') 
                )
            fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Time Period",yaxis_title="Default Rate",hovermode="x unified",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
            st.plotly_chart(fig_trend,use_container_width=True)

elif st.session_state["active_tab"] == "Statistical Analysis":
    with tab2:
        st.header("Statiscal Intelligence (WoE and IV)") 
        exclude_cols =['bad_loan','loan_amnt','issue_d','year','loan_status']   
        feature_options = [col for col in display_df.select_dtypes(include=['object', 'category', 'string']).columns if col not in exclude_cols]
        if not feature_options:
            feature_options = ['grade', 'home_ownership', 'purpose', 'verification_status']
        selected_feature = st.selectbox("Select Risk Factor for Information Value Analysis",
                                        options=map(lambda x: x.replace("_"," ").title(),feature_options))

    with st.spinner(f"Computing Weight of Evidence for {selected_feature}...."):
            woe_df,iv_value = calculate_woe_iv(display_df,selected_feature.lower().replace(' ','_'))

            if iv_value<0.02: iv_strength = "Useless Predictor"
            elif iv_value<0.1: iv_strength = "Weak Predictor"
            elif iv_value<0.3: iv_strength = "Medium Predictor"
            elif iv_value<0.5: iv_strength = "Strong Predictor"
            else: iv_strength = "Suspiciously Strong Predictor/Too Good to be True"

            stat_col1,stat_col2 = st.columns([1,3])

            with stat_col1:
                st.metric("Information Value", f"{iv_value:.4f}")
                st.caption("Higher IV values signify stronger capability to segregate good loans from bad loans.")
                st.markdown(f"Predictive Strength: {iv_strength}")

            with stat_col2:
                st.subheader(f"Weight of Evidence (WoE) Trend: {selected_feature.replace('_',' ').title()}")

                fig2 = px.bar(woe_df,x=selected_feature.lower().replace(' ','_'),y='WoE',color='WoE',text_auto='.2f',color_continuous_scale=px.colors.diverging.RdYlBu)
                fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',xaxis_title=f"Categories of {selected_feature.title()}",
                yaxis_title="WoE Score"
                )
                st.plotly_chart(fig2, use_container_width=True)
elif st.session_state["active_tab"] == "Live Risk Calculator":
    with tab3:
        st.header("Live Interactive Risk Calculator")
        st.markdown("Enter applicant details to simulate a real-time credit decision. Calculation triggers only on demand.")
        
        available_grades = sorted(df['grade'].dropna().unique()) if 'grade' in df.columns else ['A', 'B', 'C', 'D']
        available_home = df['home_ownership'].dropna().unique() if 'home_ownership' in df.columns else ['RENT', 'OWN', 'MORTGAGE']

        with st.form("risk_engine_form", clear_on_submit=False):
            c1, c2 = st.columns(2)

            with c1:
                grade_input = st.selectbox("Assigned Grade", options=available_grades)
                loan_amnt_input = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, step=500)

            with c2:
                home_input = st.selectbox("Home Ownership", options=available_home)
                int_rate_input = st.slider("Interest Rate (%)", 5.0, 30.0, 10.0)

            submit_button = st.form_submit_button(label="Analyze Risk Profile 🚀", type="primary")

        
        if submit_button:
            
            with st.spinner("Processing through Risk Inference Engine..."):
                applicant_data = {
                    "grade": grade_input,
                    "loan_amnt": loan_amnt_input,
                    "home_ownership": home_input,
                    "int_rate": int_rate_input
                }

                risk_score = (loan_amnt_input / 50000) * 0.4 + (int_rate_input / 30) * 0.6
                risk_category = "High Risk" if risk_score > 0.5 else "Low Risk"

                st.markdown("---")
                res1, res2 = st.columns(2)
                res1.metric("Calculated Risk Score", f"{risk_score:.2f}")
                res2.metric("Decision", risk_category, delta_color="inverse")

                st.success("Analysis Complete.")


