import numpy as np
import numpy_financial as npf
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional, Dict

# --- Data Models (Pydantic) ---

class DCFInput(BaseModel):
    """
    Strict validation schema for DCF inputs.
    """
    project_name: str
    initial_investment: float = Field(..., gt=0, description="Initial outlay (positive value)")
    cash_flows: List[float] = Field(..., min_length=1, description="Projected Operating Cash Flows")
    wacc: float = Field(..., gt=0, le=1.0, description="Weighted Average Cost of Capital (0 to 1)")
    tax_rate: float = Field(..., ge=0, le=1.0, description="Corporate Tax Rate (0 to 1)")
    reporting_currency: str = Field(..., min_length=3, max_length=3)
    project_currency: str = Field(..., min_length=3, max_length=3)

    @field_validator('cash_flows')
    def check_non_empty(cls, v):
        if not v:
            raise ValueError("Cash flows list cannot be empty")
        return v

# --- Core Logic ---

class ValuationEngine:
    """
    A modular engine for Discounted Cash Flow (DCF) valuation with 
    multi-currency and sensitivity capabilities.
    """

    def get_real_time_fx(self, base_currency: str, quote_currency: str) -> float:
        """
        Fetches real-time FX rate using yfinance. 
        Returns 1.0 if currencies are identical.
        Fallback logic included for stability.
        """
        base = base_currency.upper()
        quote = quote_currency.upper()

        if base == quote:
            return 1.0

        # Construct ticker symbol (Standard convention: BASEQUOTE=X)
        ticker_symbol = f"{base}{quote}=X"
        
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Fetch generic '1d' history to get the latest close
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            else:
                # Try inverse pair if direct pair fails
                ticker_inv = yf.Ticker(f"{quote}{base}=X")
                hist_inv = ticker_inv.history(period="1d")
                if not hist_inv.empty:
                    return 1.0 / float(hist_inv['Close'].iloc[-1])
                
            return 1.0 # Default fallback if API fails
        except Exception as e:
            print(f"FX Fetch Error: {e}")
            return 1.0

    def calculate_metrics(self, data: DCFInput, manual_fx_rate: Optional[float] = None) -> Dict:
        """
        Performs the core DCF calculation: Tax impact, Discounting, NPV, IRR.
        Returns a dictionary containing scalar results and the details DataFrame.
        """
        
        # 1. Determine FX Rate
        fx_rate = manual_fx_rate if manual_fx_rate else self.get_real_time_fx(data.project_currency, data.reporting_currency)

        # 2. Convert Inputs to Reporting Currency
        # Note: Initial investment is usually an outflow (-), but user enters (+) in model
        # We treat it as negative for NPV/IRR calc internally
        inv_reporting = data.initial_investment * fx_rate
        
        # 3. Process Annual Cash Flows
        years = range(1, len(data.cash_flows) + 1)
        
        # Create DataFrame for transparent calculation
        df = pd.DataFrame({
            'Year': years,
            'Op_CF_Project_Ccy': data.cash_flows
        })

        # Apply Tax (Tax is applied to Operating CFs)
        # Post-Tax CF = CF * (1 - Tax Rate)
        df['Post_Tax_CF_Project_Ccy'] = df['Op_CF_Project_Ccy'] * (1 - data.tax_rate)
        
        # Convert to Reporting Currency
        df['Post_Tax_CF_Reporting_Ccy'] = df['Post_Tax_CF_Project_Ccy'] * fx_rate
        
        # Discount Factors
        df['Discount_Factor'] = 1 / ((1 + data.wacc) ** df['Year'])
        
        # Present Value
        df['PV_Reporting_Ccy'] = df['Post_Tax_CF_Reporting_Ccy'] * df['Discount_Factor']

        # 4. Calculate NPV
        # NPV = Sum(PV of inflows) - Initial Investment
        sum_pv = df['PV_Reporting_Ccy'].sum()
        npv = sum_pv - inv_reporting

        # 5. Calculate IRR
        # Construct full cash flow stream: [-Investment, CF1, CF2, ...]
        cf_stream = [-inv_reporting] + df['Post_Tax_CF_Reporting_Ccy'].tolist()
        
        try:
            irr = npf.irr(cf_stream)
            if np.isnan(irr):
                irr = None 
        except:
            irr = None

        return {
            "npv": npv,
            "irr": irr,
            "fx_rate_used": fx_rate,
            "initial_investment_reporting": inv_reporting,
            "details_df": df
        }

    def sensitivity_analysis(self, data: DCFInput, current_fx: float) -> pd.DataFrame:
        """
        Analyzes NPV sensitivity to FX rate movements (+/- 45%).
        """
        shifts = np.linspace(-0.45, 0.45, 20) # 20 data points from -45% to +45%
        results = []

        for shift in shifts:
            simulated_fx = current_fx * (1 + shift)
            # Recalculate metrics with the simulated FX rate
            metrics = self.calculate_metrics(data, manual_fx_rate=simulated_fx)
            
            results.append({
                "FX_Movement_Pct": shift * 100,
                "Simulated_FX_Rate": simulated_fx,
                "NPV": metrics['npv']
            })
            
        return pd.DataFrame(results)

# Unit Test block (runs if file is executed directly)
if __name__ == "__main__":
    test_data = DCFInput(
        project_name="Test Project",
        initial_investment=1000000,
        cash_flows=[200000, 300000, 400000, 500000],
        wacc=0.10,
        tax_rate=0.25,
        reporting_currency="USD",
        project_currency="EUR"
    )
    engine = ValuationEngine()
    print("Fetching FX and Calculating...")
    res = engine.calculate_metrics(test_data)
    print(f"NPV: {res['npv']:,.2f}")
    print(f"IRR: {res['irr']:.2%}")
