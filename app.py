import streamlit as st
import pandas as pd
import plotly.express as px
from pydantic import ValidationError

# Import the custom engine
from engine import ValuationEngine, DCFInput

# --- Configuration & Setup ---
st.set_page_config(
    page_title="QuantPro: DCF Valuation Engine",
    page_icon="fq",
    layout="wide"
)

# --- Font Customization (Roboto) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .metric-label { font-size: 14px; color: #555; }
    .metric-value { font-size: 24px; font-weight: bold; color: #000; }
    </style>
""", unsafe_allow_html=True)

# Initialize Engine
engine = ValuationEngine()

# --- Helper Functions ---

def get_color_style(value):
    """Returns CSS color string based on value sign."""
    if value > 0:
        return "color: #28a745; border-left: 5px solid #28a745;" # Green
    elif value < 0:
        return "color: #dc3545; border-left: 5px solid #dc3545;" # Red
    else:
        return "color: #007bff; border-left: 5px solid #007bff;" # Blue (Breakeven)

# --- Bi-Directional Sync Callbacks ---
# These functions ensure that when one widget moves, the other updates to match.

if 'wacc' not in st.session_state: st.session_state.wacc = 0.085
if 'tax' not in st.session_state: st.session_state.tax = 0.21

def update_wacc_from_slider():
    st.session_state.wacc = st.session_state.wacc_slider
def update_wacc_from_input():
    st.session_state.wacc = st.session_state.wacc_input
    
def update_tax_from_slider():
    st.session_state.tax = st.session_state.tax_slider
def update_tax_from_input():
    st.session_state.tax = st.session_state.tax_input

# --- Caching ---
@st.cache_data(ttl=3600)
def get_cached_fx(base, quote):
    return engine.get_real_time_fx(base, quote)

@st.cache_data
def run_simulation(input_model_dict, manual_fx):
    model = DCFInput(**input_model_dict)
    return engine.calculate_metrics(model, manual_fx_rate=manual_fx)

@st.cache_data
def run_sensitivity(input_model_dict, current_fx):
    model = DCFInput(**input_model_dict)
    return engine.sensitivity_analysis(model, current_fx)

# --- Sidebar: Inputs ---
st.sidebar.header("1. Global Parameters")
project_name = st.sidebar.text_input("Project Name", "Alpha Expansion")
reporting_ccy = st.sidebar.selectbox("Reporting Currency (Investor)", ["USD", "EUR", "GBP", "JPY", "CAD"])
project_ccy = st.sidebar.selectbox("Project Currency (Local)", ["EUR", "USD", "GBP", "JPY", "CAD"], index=0)

st.sidebar.header("2. Financial Assumptions")
initial_inv = st.sidebar.number_input("Initial Investment (Local Ccy)", min_value=1000.0, value=1000000.0, step=10000.0)

# Synced WACC Input
st.sidebar.subheader("Cost of Capital (WACC)")
col_w1, col_w2 = st.sidebar.columns([1, 2])
with col_w1:
    # Text Input linked to main 'wacc' state
    st.number_input("WACC", 0.0, 0.5, key="wacc_input", on_change=update_wacc_from_input, format="%.3f", label_visibility="collapsed", value=st.session_state.wacc)
with col_w2:
    # Slider linked to main 'wacc' state
    st.slider("", 0.0, 0.20, key="wacc_slider", on_change=update_wacc_from_slider, format="%.3f", label_visibility="collapsed", value=st.session_state.wacc)

# Synced Tax Input
st.sidebar.subheader("Tax Rate")
col_t1, col_t2 = st.sidebar.columns([1, 2])
with col_t1:
    st.number_input("Tax", 0.0, 0.5, key="tax_input", on_change=update_tax_from_input, format="%.3f", label_visibility="collapsed", value=st.session_state.tax)
with col_t2:
    st.slider("", 0.0, 0.50, key="tax_slider", on_change=update_tax_from_slider, format="%.3f", label_visibility="collapsed", value=st.session_state.tax)

# --- 2. Cash Flow Projection Method ---
st.sidebar.header("3. Projections")
proj_years = st.sidebar.number_input("Projection Years", 3, 15, 5)

method = st.sidebar.radio("Projection Method", ["Growth Rate (Auto)", "Manual Entry"])

cash_flows = []
if method == "Growth Rate (Auto)":
    base_cf = st.sidebar.number_input("Year 1 Operating Cash Flow", value=250000.0)
    growth_rate = st.sidebar.slider("Annual Growth Rate (%)", -10.0, 50.0, 5.0) / 100
    cash_flows = [base_cf * ((1 + growth_rate) ** i) for i in range(proj_years)]
else:
    st.sidebar.info("Enter Cash Flows in the Main Dashboard")
    # Initialize default data for the editor
    default_data = pd.DataFrame({
        "Year": range(1, proj_years + 1),
        "Operating Cash Flow": [250000.0] * proj_years
    })

# --- Main Dashboard ---
st.title(f"📊 DCF Valuation: {project_name}")

# Handle Manual Entry in Main Area
if method == "Manual Entry":
    st.subheader("📝 Manual Cash Flow Entry")
    st.caption("Please edit the 'Operating Cash Flow' column below. The 'Year' column is fixed.")
    
    # QoL Change: Disable "Year" column to prevent confusion
    edited_df = st.data_editor(
        default_data, 
        column_config={
            "Operating Cash Flow": st.column_config.NumberColumn(format="%.2f", required=True),
            "Year": st.column_config.NumberColumn(disabled=True) # Makes this column read-only
        },
        disabled=["Year"], # Extra layer of safety
        hide_index=True,
        use_container_width=True,
        num_rows="fixed" # Prevents adding/deleting rows
    )
    cash_flows = edited_df["Operating Cash Flow"].tolist()

# 1. Validation & Data Prep
try:
    dcf_input = DCFInput(
        project_name=project_name,
        initial_investment=initial_inv,
        cash_flows=cash_flows,
        wacc=st.session_state.wacc, 
        tax_rate=st.session_state.tax, 
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
    results = run_simulation(dcf_input.model_dump(), active_fx)
    
    npv = results['npv']
    irr = results['irr']
    df_details = results['details_df']

    # --- Results Display ---
    
    # Prepare styles and strings explicitly BEFORE the HTML block
    npv_style = get_color_style(npv)
    irr_val = irr if irr else 0 
    irr_style = get_color_style(irr_val)
    irr_display = f"{irr:.2%}" if irr else "N/A"
    
    # Robust Formatting for FX and Rates
    fx_display_str = f"{active_fx:.4f}"
    rates_display_str = f"{st.session_state.wacc:.1%} / {st.session_state.tax:.1%}"

    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; {npv_style}">
            <div class="metric-label">Net Present Value ({reporting_ccy})</div>
            <div class="metric-value">{npv:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; {irr_style}">
            <div class="metric-label">Internal Rate of Return (IRR)</div>
            <div class="metric-value">{irr_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; border-left: 5px solid #6c757d;">
            <div class="metric-label">FX Rate Used</div>
            <div class="metric-value">{fx_display_str}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card" style="background-color: #f8f9fa; border-left: 5px solid #6c757d;">
            <div class="metric-label">WACC / Tax</div>
            <div class="metric-value">{rates_display_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Layout: Charts and Tables
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Cash Flow Trajectory")
        fig_cf = px.bar(
            df_details, 
            x='Year', 
            y=['Op_CF_Project_Ccy', 'Post_Tax_CF_Reporting_Ccy'],
            barmode='group',
            labels={'value': 'Amount', 'variable': 'Cash Flow Type'},
            title="Pre-Tax (Local) vs Post-Tax (Reporting) Cash Flows"
        )
        fig_cf.update_layout(font=dict(family="Roboto"))
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_right:
        st.subheader("Financial Schedule")
        display_df = df_details[['Year', 'Op_CF_Project_Ccy', 'Post_Tax_CF_Reporting_Ccy', 'PV_Reporting_Ccy']].copy()
        display_df.columns = ['Year', f'Op CF ({project_ccy})', f'Post-Tax ({reporting_ccy})', f'PV ({reporting_ccy})']
        st.dataframe(display_df.style.format("{:,.0f}"), use_container_width=True, hide_index=True)

    # --- Sensitivity Analysis Section ---
    st.divider()
    st.subheader("🌍 FX Sensitivity Analysis")
    
    sens_df = run_sensitivity(dcf_input.model_dump(), active_fx)

    fig_sens = px.area(
        sens_df, 
        x="FX_Movement_Pct", 
        y="NPV",
        labels={"FX_Movement_Pct": "FX Movement (%)", "NPV": f"NPV ({reporting_ccy})"},
        title=f"NPV Sensitivity to {project_ccy}/{reporting_ccy} Rate"
    )
    
    fig_sens.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Current Rate")
    fig_sens.update_traces(line_color='#007bff', fill='tozeroy')
    fig_sens.update_layout(font=dict(family="Roboto"))
    
    st.plotly_chart(fig_sens, use_container_width=True)

except ValidationError as e:
    st.error(f"Input Validation Error: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
