import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import ValidationError

# Import the custom engine
from engine import ValuationEngine, DCFInput

# --- Configuration & Setup ---
st.set_page_config(
    page_title="QuantPro: DCF Valuation Engine",
    page_icon="fq",
    layout="wide"
)

# Initialize Engine
engine = ValuationEngine()

# --- Caching ---
@st.cache_data(ttl=3600) # Cache FX calls for 1 hour
def get_cached_fx(base, quote):
    return engine.get_real_time_fx(base, quote)

@st.cache_data
def run_simulation(input_model_dict, manual_fx):
    # Reconstruct Pydantic model inside cache (Streamlit hashes arguments)
    model = DCFInput(**input_model_dict)
    return engine.calculate_metrics(model, manual_fx_rate=manual_fx)

@st.cache_data
def run_sensitivity(input_model_dict, current_fx):
    model = DCFInput(**input_model_dict)
    return engine.sensitivity_analysis(model, current_fx)

# --- UI Styling ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #555; }
    .metric-value { font-size: 24px; font-weight: bold; color: #0f1116; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Inputs ---
st.sidebar.header("1. Global Parameters")
project_name = st.sidebar.text_input("Project Name", "Alpha Expansion")
reporting_ccy = st.sidebar.selectbox("Reporting Currency (Investor)", ["USD", "EUR", "GBP", "JPY", "CAD"])
project_ccy = st.sidebar.selectbox("Project Currency (Local)", ["EUR", "USD", "GBP", "JPY", "CAD"], index=0)

st.sidebar.header("2. Financial Assumptions")
initial_inv = st.sidebar.number_input("Initial Investment (Local Ccy)", min_value=1000.0, value=1000000.0, step=10000.0)
wacc_input = st.sidebar.slider("WACC (Cost of Capital %)", 1.0, 20.0, 8.5) / 100
tax_input = st.sidebar.slider("Corporate Tax Rate (%)", 0.0, 40.0, 21.0) / 100

st.sidebar.header("3. Projections")
proj_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_cf = st.sidebar.number_input("Year 1 Operating Cash Flow (Local Ccy)", value=250000.0)
growth_rate = st.sidebar.slider("Annual Growth Rate (%)", -10.0, 50.0, 5.0) / 100

# Generate Cash Flows based on simple growth for the demo
cash_flows = [base_cf * ((1 + growth_rate) ** i) for i in range(proj_years)]

# --- Main Dashboard ---
st.title(f"📊 DCF Valuation: {project_name}")

# 1. Validation & Data Prep
try:
    # Build Pydantic Model
    dcf_input = DCFInput(
        project_name=project_name,
        initial_investment=initial_inv,
        cash_flows=cash_flows,
        wacc=wacc_input,
        tax_rate=tax_input,
        reporting_currency=reporting_ccy,
        project_currency=project_ccy
    )

    # 2. Fetch Live Data
    live_fx = get_cached_fx(project_ccy, reporting_ccy)
    
    # Allow manual FX override
    use_manual_fx = st.checkbox(f"Override Live FX Rate ({live_fx:.4f})")
    if use_manual_fx:
        active_fx = st.number_input("Manual FX Rate", value=live_fx, format="%.4f")
    else:
        active_fx = live_fx

    # 3. Calculation Engine
    # Pass dict to cache function to avoid pickling issues with Pydantic in some envs
    results = run_simulation(dcf_input.model_dump(), active_fx)
    
    npv = results['npv']
    irr = results['irr']
    df_details = results['details_df']

    # --- Results Display ---
    
    # Top KPI Row
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Present Value ({reporting_ccy})</div>
            <div class="metric-value">{npv:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        irr_display = f"{irr:.2%}" if irr else "N/A"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid #28a745;">
            <div class="metric-label">Internal Rate of Return (IRR)</div>
            <div class="metric-value">{irr_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.metric("FX Rate Used", f"{active_fx:.4f}", help=f"1 {project_ccy} = {active_fx:.4f} {reporting_ccy}")
        
    with c4:
        st.metric("WACC / Tax", f"{wacc_input:.1%} / {tax_input:.1%}")

    st.divider()

    # Layout: Charts and Tables
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Cash Flow Trajectory")
        
        # Plotly Bar Chart: Annual Cash Flows
        fig_cf = px.bar(
            df_details, 
            x='Year', 
            y=['Op_CF_Project_Ccy', 'Post_Tax_CF_Reporting_Ccy'],
            barmode='group',
            labels={'value': 'Amount', 'variable': 'Cash Flow Type'},
            title="Pre-Tax (Local) vs Post-Tax (Reporting) Cash Flows"
        )
        fig_cf.update_layout(xaxis_title="Year", yaxis_title=f"Currency Amount")
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_right:
        st.subheader("Financial Schedule")
        # Formatting for display
        display_df = df_details[['Year', 'Op_CF_Project_Ccy', 'Post_Tax_CF_Reporting_Ccy', 'PV_Reporting_Ccy']].copy()
        display_df.columns = ['Year', f'Op CF ({project_ccy})', f'Post-Tax ({reporting_ccy})', f'PV ({reporting_ccy})']
        st.dataframe(display_df.style.format("{:,.0f}"), use_container_width=True, hide_index=True)

    # --- Sensitivity Analysis Section ---
    st.divider()
    st.subheader("🌍 FX Sensitivity Analysis")
    st.markdown("Impact of exchange rate volatility on Project NPV (±45% Movement).")

    # Run sensitivity
    sens_df = run_sensitivity(dcf_input.model_dump(), active_fx)

    # Visualization
    fig_sens = px.area(
        sens_df, 
        x="FX_Movement_Pct", 
        y="NPV",
        labels={"FX_Movement_Pct": "FX Movement (%)", "NPV": f"NPV ({reporting_ccy})"},
        title=f"NPV Sensitivity to {project_ccy}/{reporting_ccy} Rate"
    )
    
    # Add a vertical line for current state
    fig_sens.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Current Rate")
    
    # Dynamic coloring based on NPV > 0
    fig_sens.update_traces(line_color='#1f77b4', fill='tozeroy')
    st.plotly_chart(fig_sens, use_container_width=True)

except ValidationError as e:
    st.error(f"Input Validation Error: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
