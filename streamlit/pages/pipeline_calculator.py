import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import io
import warnings
warnings.filterwarnings('ignore')


def get_data_path(relative_path):
    """
    Resolves the correct file path for both local and Streamlit Cloud environments.
    Tries the given path first, then checks one level up.
    """
    if os.path.exists(relative_path):
        return relative_path

    parent_path = os.path.join("..", relative_path)
    if os.path.exists(parent_path):
        return parent_path

    return relative_path  # Let it fail naturally if not found


# Set page config
st.set_page_config(
    page_title="Pipeline Calculator",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧮 Pipeline Calculator")
st.markdown("""
This interactive calculator predicts pipeline requirements based on:
- **Target market share goals** and territory-specific factors
- **Predicted sales** from machine learning models (98.9% accuracy)
- **Forecasted win rates** and recurring business patterns
- **Dynamic market conditions** and seasonal adjustments
""")

def format_table(df):
    """Format table without using pandas Styler to avoid errors"""
    return df

def get_user_provided_industry_data(territory, segment, year, month, default_value=1000):
    """
    Get user-provided industry data for future periods where market data is unavailable.
    This function can be called from other parts of the application.
    
    Parameters:
    -----------
    territory : str
        Territory code (e.g., 'A04')
    segment : str  
        Business segment ('Commercial' or 'Residential')
    year : int
        Year for the data
    month : int
        Month for the data
    default_value : int
        Default industry quantity if no user input available
        
    Returns:
    --------
    int
        Industry quantity value
    """
    # Default values based on segment if not provided by user
    default_values = {
        'Residential': 1200 if year >= 2025 else 1000,
        'Commercial': 800 if year >= 2025 else 600
    }
    
    # Check if we have stored user inputs in session state
    session_key = f"industry_data_{territory}_{segment}_{year}_{month}"
    if hasattr(st.session_state, 'user_industry_data') and session_key in st.session_state.user_industry_data:
        return st.session_state.user_industry_data[session_key]
    
    # Return default based on segment and year
    return default_values.get(segment, default_value)

# Load required data with enhanced validation and fallback options
@st.cache_data
def load_data():
    """
    Load pipeline data with multiple fallback options and validation
    """
    try:
        # Try to load the full pipeline results first
        df = pd.read_excel(get_data_path("results/full_pipeline_results.xlsx"))
        #df = pd.read_excel("../results/full_pipeline_results.xlsx")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        st.success("✅ Full pipeline results loaded successfully")
        
    except FileNotFoundError:
        try:
            # Fallback to test pipeline results
            df = pd.read_excel("../results/test_pipeline_results.xlsx")
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            
            st.warning("⚠️ Using test pipeline results (full results not available)")
            
        except FileNotFoundError:
            st.error("❌ No pipeline results found. Please run the pipeline calculation first.")
            return None
    
    except Exception as e:
        st.error(f"Error loading pipeline data: {str(e)}")
        return None
    
    # Enhanced data validation and cleaning
    try:
        # Ensure required columns exist
        required_columns = ['Territory', 'Segment', 'Date', 'Industry_Qty', 
                          'Predicted_Sales', 'Predicted_Win_Rate', 'Predicted_Residual',
                          'Actual_Residual', 'Adjusted_Pipeline_Need']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Missing columns: {missing_columns}. Some features may not work correctly.")
            
            # Add missing columns with reasonable defaults
            for col in missing_columns:
                if 'Industry_Qty' in col:
                    df[col] = 1000
                elif 'Win_Rate' in col:
                    df[col] = 0.4
                elif 'Sales' in col or 'Residual' in col or 'Pipeline' in col:
                    df[col] = 0
                else:
                    df[col] = 0
        
        # Data cleaning and validation
        if 'Industry_Qty' in df.columns:
            # Handle missing and invalid industry quantities
            df['Industry_Qty'] = df['Industry_Qty'].fillna(1000)
            df['Industry_Qty'] = df['Industry_Qty'].clip(lower=1)
            
            if df['Industry_Qty'].min() < 1:
                st.warning("Some industry quantities were less than 1. Values have been adjusted.")
        
        # Validate numeric columns
        numeric_columns = ['Predicted_Sales', 'Predicted_Win_Rate', 'Predicted_Residual', 
                          'Actual_Residual', 'Adjusted_Pipeline_Need']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Ensure win rates are within valid range
        if 'Predicted_Win_Rate' in df.columns:
            df['Predicted_Win_Rate'] = df['Predicted_Win_Rate'].clip(0.01, 0.99)
        
        return df
        
    except Exception as e:
        st.error(f"Error during data validation: {str(e)}")
        return None

# Load data
data = load_data()

if data is not None:
    # Data overview for user
    st.sidebar.markdown("### 📊 Data Overview")
    st.sidebar.write(f"**Records**: {len(data):,}")
    st.sidebar.write(f"**Date Range**: {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}")
    st.sidebar.write(f"**Territories**: {data['Territory'].nunique()}")
    
    # Determine data availability by year
    years = sorted(data['Year'].unique())
    year_completeness = {}
    
    for year in years:
        year_data = data[data['Year'] == year]
        months_available = len(year_data['Month'].unique()) if 'Month' in year_data.columns else 12
        has_industry_data = not year_data['Industry_Qty'].isnull().all()
        
        year_completeness[year] = {
            'months': months_available,
            'has_industry': has_industry_data,
            'is_complete': months_available >= 12 and has_industry_data
        }
    
    # Year selection with data completeness indicators
    year_options = []
    for year in years + [max(years) + 1, max(years) + 2]:  # Add future years
        if year in year_completeness:
            status = year_completeness[year]
            if status['is_complete']:
                year_options.append(f"{year} ✅ Complete")
            elif status['has_industry']:
                year_options.append(f"{year} ⚠️ Partial ({status['months']} months)")
            else:
                year_options.append(f"{year} ❌ Limited")
        else:
            year_options.append(f"{year} 🔮 Future")
    
    selected_year_option = st.sidebar.selectbox(
        "Select Year",
        options=year_options,
        index=len(years)-1 if years else 0  # Default to most recent year
    )
    
    # Extract year from selection
    selected_year = int(selected_year_option.split()[0])
    
    # Check if selected year has industry data
    if selected_year in year_completeness:
        has_industry_data = year_completeness[selected_year]['has_industry']
        months_available = year_completeness[selected_year]['months']
    else:
        has_industry_data = False
        months_available = 0
    
    # Handle data filtering and estimation mode
    if selected_year in years and has_industry_data:
        filtered_data = data[data['Year'] == selected_year]
        use_estimated_market = False
        st.sidebar.success(f"Using actual {selected_year} data")
    else:
        # Use most recent year as baseline for estimation
        baseline_year = max(years)
        filtered_data = data[data['Year'] == baseline_year]
        use_estimated_market = True
        
        if selected_year > max(years):
            st.sidebar.info(f"🔮 Future year selected. Using {baseline_year} as baseline.")
        else:
            st.sidebar.warning(f"⚠️ Limited data for {selected_year}. Using {baseline_year} as baseline.")

    # Get unique territories and segments
    territories = sorted(filtered_data['Territory'].unique())
    segments = sorted(filtered_data['Segment'].unique())

    # Input form
    st.sidebar.subheader("🎯 Pipeline Parameters")

    territory = st.sidebar.selectbox(
        "Select Territory",
        territories,
        index=0,
        help="Choose the territory for pipeline calculation"
    )

    segment = st.sidebar.selectbox(
        "Select Segment",
        segments,
        index=0,
        help="Choose the business segment (Commercial or Residential)"
    )

    # Filter data for the selected territory and segment
    territory_data = filtered_data[
        (filtered_data['Territory'] == territory) &
        (filtered_data['Segment'] == segment)
    ]

    if territory_data.empty:
        st.error(f"❌ No data available for Territory {territory}, Segment {segment}")
        st.info("Please try a different territory/segment combination or check data availability.")
        st.stop()
    
    # Calculate baseline metrics from available data
    try:
        summed_industry_qty = max(1, int(territory_data['Industry_Qty'].sum()))
        summed_predicted_sales = float(territory_data['Predicted_Sales'].sum())
        summed_predicted_residual = float(territory_data['Predicted_Residual'].sum())
        avg_predicted_win_rate = float(territory_data['Predicted_Win_Rate'].mean())
        summed_actual_residual = float(territory_data['Actual_Residual'].sum())
        
        # Get most recent record for baseline comparison
        baseline = territory_data.iloc[-1] if not territory_data.empty else None
        
    except Exception as e:
        st.error(f"Error calculating baseline metrics: {str(e)}")
        st.stop()
    
    # Market Parameters Section
    st.sidebar.subheader("🏪 Market Parameters")
    
    # Industry quantity input with enhanced UX for future years
    if use_estimated_market:
        # For future years, allow full range of industry quantity inputs
        min_industry_qty = 1
        max_industry_qty = summed_industry_qty * 3  # Allow up to triple the baseline
        industry_qty_default = summed_industry_qty
        
        st.sidebar.info(f"💡 Estimated market size based on {max(years)} baseline")
    else:
        # For historical years, maintain some constraints
        industry_qty_default = summed_industry_qty
        min_industry_qty = max(1, int(industry_qty_default * 0.8))  # 20% below baseline
        max_industry_qty = int(industry_qty_default * 1.5)  # 50% above baseline

    industry_qty = st.sidebar.number_input(
        "Industry Quantity (Market Size)",
        min_value=min_industry_qty,
        max_value=max_industry_qty,
        value=industry_qty_default,
        step=50,
        help=f"Total annual market size in units. Baseline: {industry_qty_default:,.0f}"
    )

    target_market_share = st.sidebar.slider(
        "Target Market Share (%)",
        min_value=5,
        max_value=60,
        value=30,
        step=1,
        help="Desired annual market share percentage (company target: 30%)"
    )

    st.sidebar.subheader("🔧 Model Predictions")

    use_model_predictions = st.sidebar.checkbox(
        "Use ML Model Predictions",
        value=True,
        help="Uncheck to manually override model predictions"
    )

    if use_model_predictions:
        # Use aggregated annual values for predictions
        predicted_sales = summed_predicted_sales
        predicted_win_rate = avg_predicted_win_rate
        predicted_residual = summed_predicted_residual
        actual_residual = summed_actual_residual
        
        # Display model confidence for transparency
        st.sidebar.success("🤖 Using AI predictions")
        st.sidebar.caption("Models: Sales (R²=98.9%), Win Rate (R²=98.4%), Residual (R²=58.9%)")
    else:
        st.sidebar.warning("⚠️ Manual override mode")
        
        predicted_sales = st.sidebar.number_input(
            "Predicted Sales (Units)",
            min_value=0,
            value=int(summed_predicted_sales),
            step=10,
            help="Override ML model sales prediction"
        )

        predicted_win_rate = st.sidebar.slider(
            "Predicted Win Rate",
            min_value=0.05,
            max_value=0.95,
            value=float(avg_predicted_win_rate),
            step=0.01,
            format="%.2f",
            help="Override ML model win rate prediction"
        )

        predicted_residual = st.sidebar.number_input(
            "Predicted Residual Business (Units)",
            min_value=0,
            value=int(summed_predicted_residual),
            step=10,
            help="Override ML model residual business prediction"
        )
        
        actual_residual = st.sidebar.number_input(
            "Actual Residual Business (Units)",
            min_value=0,
            value=int(summed_actual_residual),
            step=10,
            help="Historical residual business for comparison"
        )

    # Calculate pipeline metrics with enhanced validation
    try:
        current_market_share = (predicted_sales / industry_qty * 100) if industry_qty > 0 else 0
        target_volume = industry_qty * target_market_share / 100
        sales_gap = target_volume - predicted_sales
        
        # Ensure new business pipeline doesn't go negative
        new_business_pipeline = max(0, target_volume - predicted_residual)
        
        # Handle division by zero for win rate
        base_pipeline_need = (new_business_pipeline / predicted_win_rate) if predicted_win_rate > 0 else 0

        # Gap factor calculation with safety checks
        if target_volume > 0:
            raw_gap_factor = 1 + (sales_gap / target_volume)
            sales_gap_factor = max(0.5, min(3.0, raw_gap_factor))  # Limit to reasonable range
        else:
            sales_gap_factor = 1.0

        adjusted_pipeline_need = base_pipeline_need * sales_gap_factor
        
    except Exception as e:
        st.error(f"Error in pipeline calculations: {str(e)}")
        st.stop()

    # Dynamic summary text with enhanced business logic
    residual_comparison = ""
    if actual_residual > predicted_residual * 1.2:
        residual_comparison = f"We have **strong automatic repeat business** ({actual_residual:,.0f} actual residual),"
    elif actual_residual > predicted_residual * 0.8:
        residual_comparison = f"We have **consistent repeat business** ({actual_residual:,.0f} actual residual),"
    else:
        residual_comparison = f"While we had some repeat business ({actual_residual:,.0f} actual residual),"

    residual_prediction = ""
    if predicted_residual > actual_residual * 1.2:
        residual_prediction = f"we expect **significant growth** to around {predicted_residual:,.0f} (predicted residual) sales from repeat customers,"
    elif predicted_residual > actual_residual * 0.8:
        residual_prediction = f"we predict **maintaining** around {predicted_residual:,.0f} (predicted residual) sales from repeat customers,"
    else:
        residual_prediction = f"we forecast a **decrease** to around {predicted_residual:,.0f} (predicted residual) sales from repeat customers,"

    urgency_description = ""
    urgency_percent = (sales_gap_factor-1)*100
    if urgency_percent > 50:
        urgency_description = f"we added a **strong {urgency_percent:.0f}% urgency boost** to your outreach"
    elif urgency_percent > 20:
        urgency_description = f"we added a **moderate {urgency_percent:.0f}% urgency boost** to your outreach"
    elif urgency_percent > 0:
        urgency_description = f"we added a **slight {urgency_percent:.0f}% urgency adjustment** to your outreach"
    else:
        urgency_description = "**no urgency adjustment** was needed"

    summary_text = f"""
    ### 📊 Executive Summary for Territory **{territory} - {segment}**
    
    To achieve **{target_market_share}%** market share, we need **{target_volume:,.0f}** total sales.
    
    **Current Situation:**
    - {residual_comparison} and {residual_prediction}
    - This means we need **{new_business_pipeline:,.0f}** new sales to hit our target
    - With our **{predicted_win_rate*100:.1f}%** win rate, we need **{base_pipeline_need:,.0f}** qualified opportunities
    
    **Action Required:**
    Since our model predicts only **{predicted_sales:,.0f}** sales, {urgency_description}.
    
    ### 🎯 **Final Pipeline Target: {adjusted_pipeline_need:,.0f} opportunities**
    """
    
    if use_estimated_market:
        summary_text += "\n\n📝 *Note: Using estimated market parameters for future year planning.*"
    
    st.markdown("---")
    st.markdown(summary_text)

    
    # Define whether the territory has met its market share
    territory_met_market_share = current_market_share >= target_market_share


    # Display results in enhanced tabs
    tab1, tab2, tab3 = st.tabs(["📊 Pipeline Results", "🔍 Calculation Details", "📈 Historical Context"])

    with tab1:
        # Enhanced key metrics display
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            # Market share with color coding
            market_share_delta = current_market_share - target_market_share
            market_share_color = "normal" if market_share_delta >= 0 else "inverse"
            
            st.metric(
                "Current Market Share",
                f"{current_market_share:.1f}%",
                delta=f"{market_share_delta:+.1f}%",
                #delta_color=market_share_color,
                delta_color="normal",
                help=f"Current position vs {target_market_share}% target"
            )
            
            st.metric(
                "Sales Gap",
                f"{sales_gap:+,.0f} units",
                help="Gap between target volume and predicted sales"
            )

        with col2:
            st.metric(
                "Base Pipeline Need",
                f"{base_pipeline_need:,.0f} units",
                help="New business pipeline ÷ win rate"
            )
            
            if territory_met_market_share:
                urgency_color = "inverse"
            else:
                urgency_color = "inverse" if sales_gap_factor > 1.2 else "normal"
            st.metric(
                "Urgency Factor",
                f"{sales_gap_factor:.2f}x",
                delta=f"{(sales_gap_factor - 1)*100:+.0f}%",
                delta_color=urgency_color,
                help="Adjustment based on sales gap severity"
            )

        with col3:
            pipeline_delta = adjusted_pipeline_need - base_pipeline_need
            st.metric(
                "Final Pipeline Target",
                f"{adjusted_pipeline_need:,.0f} units",
                delta=f"{pipeline_delta:+,.0f}",
                help="Annual pipeline target after urgency adjustment"
            )
            
            st.metric(
                "Recurring Business",
                f"{predicted_residual:,.0f} units",
                delta=f"{predicted_residual - actual_residual:+,.0f}",
                help="Estimated vs historical recurring business"
            )

        # Enhanced Pipeline Waterfall Visualization
        st.subheader("💧 Pipeline Calculation Flow")

        # Create waterfall data with proper logic
        waterfall_values = [
            target_volume,  # Start with target
            -predicted_residual,  # Subtract recurring business
            base_pipeline_need - new_business_pipeline,  # Win rate impact
            adjusted_pipeline_need - base_pipeline_need  # Urgency adjustment
        ]

        waterfall_labels = [
            f"Target Volume\n({target_market_share}% of {industry_qty:,.0f})",
            f"Recurring Business\n(-{predicted_residual:,.0f})",
            f"Win Rate Impact\n(÷{predicted_win_rate:.1%})",
            f"Urgency Adjustment\n(×{sales_gap_factor:.2f})"
        ]

        # Create enhanced waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Pipeline Calculation",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=waterfall_labels + ["Final Pipeline Need"],
            y=waterfall_values + [adjusted_pipeline_need],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#EF553B", "line": {"color": "rgb(0,0,0)", "width": 2}}},
            increasing={"marker": {"color": "#00CC96", "line": {"color": "rgb(0,0,0)", "width": 2}}},
            totals={"marker": {"color": "#636EFA", "line": {"color": "rgb(0,0,0)", "width": 2}}}
        ))

        fig.update_layout(
            title="Annual Pipeline Calculation Breakdown",
            showlegend=False,
            height=500,
            yaxis_title="Units",
            xaxis_title="Calculation Steps"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Performance vs Target Comparison
        st.subheader("🎯 Performance vs Targets")

        # Create comparison chart
        comp_fig = go.Figure()

        comp_fig.add_trace(go.Bar(
            name="Current/Predicted",
            x=["Sales Volume", "Market Share", "Win Rate"],
            y=[predicted_sales, current_market_share, predicted_win_rate*100],
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            text=[f"{predicted_sales:,.0f}", f"{current_market_share:.1f}%", f"{predicted_win_rate:.1%}"],
            textposition="auto"
        ))

        comp_fig.add_trace(go.Bar(
            name="Targets",
            x=["Sales Volume", "Market Share", "Win Rate"],
            y=[target_volume, target_market_share, 0.45*100],  # Assume 45% target win rate
            marker_color=["#d62728", "#ff7f0e", "#2ca02c"],
            opacity=0.7,
            text=[f"{target_volume:,.0f}", f"{target_market_share}%", "45%"],
            textposition="auto"
        ))

        comp_fig.update_layout(
            title="Current Performance vs Business Targets",
            barmode="group",
            height=400,
            yaxis_title="Value",
            showlegend=True
        )
        
        st.plotly_chart(comp_fig, use_container_width=True)

    with tab2:
        # Enhanced calculation details
        st.subheader("🔍 Data Validation & Sources")
        
        if use_estimated_market:
            st.warning(f"🔮 **Future Year Analysis**: Using {max(years)} baseline for {selected_year} projections")
            
        # Data source summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Summary**")
            st.write(f"• **Territory**: {territory}")
            st.write(f"• **Segment**: {segment}")
            st.write(f"• **Analysis Year**: {selected_year}")
            st.write(f"• **Data Points**: {len(territory_data)} months")
            st.write(f"• **Industry Qty**: {industry_qty:,.0f} units")
            
        with col2:
            st.markdown("**🎯 Model Inputs**")
            st.write(f"• **Predicted Sales**: {predicted_sales:,.0f}")
            st.write(f"• **Win Rate**: {predicted_win_rate:.1%}")
            st.write(f"• **Residual Business**: {predicted_residual:,.0f}")
            st.write(f"• **Target Share**: {target_market_share}%")

        # Show detailed monthly data for historical years
        if not use_estimated_market and not territory_data.empty:
            st.subheader("📅 Monthly Performance Data")
            
            monthly_display = territory_data[['Date', 'Industry_Qty', 'Actual_Sales', 
                                           'Predicted_Sales', 'Predicted_Residual', 
                                           'Predicted_Win_Rate']].sort_values('Date')
            
            # Format the display
            monthly_display['Date'] = monthly_display['Date'].dt.strftime('%Y-%m')
            monthly_display.columns = ['Month', 'Industry Qty', 'Actual Sales', 
                                     'Predicted Sales', 'Predicted Residual', 'Win Rate']
            
            st.dataframe(
                monthly_display.style.format({
                    'Industry Qty': '{:,.0f}',
                    'Actual Sales': '{:,.0f}',
                    'Predicted Sales': '{:,.1f}',
                    'Predicted Residual': '{:,.1f}',
                    'Win Rate': '{:.1%}'
                }),
                height=300
            )

        # Detailed calculation steps
        st.subheader("🧮 Step-by-Step Calculation")

        calculation_steps = pd.DataFrame({
            "Step": [
                "1. Market Analysis",
                "2. Current Position", 
                "3. Target Setting",
                "4. Gap Analysis",
                "5. Recurring Business",
                "6. New Business Need",
                "7. Win Rate Conversion",
                "8. Urgency Adjustment",
                "9. Final Pipeline"
            ],
            "Component": [
                "Industry Quantity",
                "Predicted Sales",
                "Target Volume", 
                "Sales Gap",
                "Predicted Residual",
                "New Business Pipeline",
                "Base Pipeline Need",
                "Urgency Factor",
                "Adjusted Pipeline Need"
            ],
            "Value": [
                f"{industry_qty:,.0f} units",
                f"{predicted_sales:,.0f} units",
                f"{target_volume:,.0f} units",
                f"{sales_gap:+,.0f} units",
                f"{predicted_residual:,.0f} units",
                f"{new_business_pipeline:,.0f} units",
                f"{base_pipeline_need:,.0f} units",
                f"{sales_gap_factor:.2f}x",
                f"{adjusted_pipeline_need:,.0f} units"
            ],
            "Formula/Logic": [
                "Market research data" if not use_estimated_market else "Estimated from baseline",
                "ML model prediction (R²=98.9%)",
                f"Industry Qty × {target_market_share}% = {industry_qty:,.0f} × {target_market_share/100:.0%}",
                f"Target - Predicted = {target_volume:,.0f} - {predicted_sales:,.0f}",
                "ML model prediction (R²=58.9%)",
                f"Target - Residual = {target_volume:,.0f} - {predicted_residual:,.0f}",
                f"New Business ÷ Win Rate = {new_business_pipeline:,.0f} ÷ {predicted_win_rate:.1%}",
                f"1 + (Gap ÷ Target) = 1 + ({sales_gap:,.0f} ÷ {target_volume:,.0f}) = {sales_gap_factor:.2f}",
                f"Base × Urgency = {base_pipeline_need:,.0f} × {sales_gap_factor:.2f}"
            ]
        })

        st.dataframe(calculation_steps, use_container_width=True)

        # Methodology explanation
        st.subheader("📚 Calculation Methodology")

        st.markdown("""
        **🎯 Pipeline Calculation Logic:**
        
        1. **Market Sizing**: Determine total addressable market (Industry Quantity)
        2. **Target Setting**: Calculate required sales volume based on market share goals
        3. **Gap Analysis**: Compare target vs predicted sales to identify shortfall
        4. **Business Segmentation**: Separate new business needs from recurring revenue
        5. **Conversion Planning**: Apply win rates to determine required pipeline volume
        6. **Urgency Adjustment**: Increase pipeline based on gap severity
        
        **🤖 Model Accuracy:**
        - **Sales Predictions**: 98.9% accuracy (R²)
        - **Win Rate Forecasts**: 98.4% accuracy (R²)
        - **Residual Business**: 58.9% accuracy (R²)
        
        """)

    with tab3:
        # Historical context and trends
        st.subheader("📈 Historical Performance Context")
        
        if not use_estimated_market and len(territory_data) > 1:
            # Create historical trend charts
            fig_trends = go.Figure()
            
            # Sales trend
            fig_trends.add_trace(go.Scatter(
                x=territory_data['Date'],
                y=territory_data['Actual_Sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='blue', width=2)
            ))
            
            fig_trends.add_trace(go.Scatter(
                x=territory_data['Date'],
                y=territory_data['Predicted_Sales'],
                mode='lines+markers',
                name='Predicted Sales',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig_trends.update_layout(
                title=f"Sales Performance Trend - {territory} {segment}",
                xaxis_title="Date",
                yaxis_title="Sales Volume",
                height=400
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Performance summary
            st.subheader("📊 Historical Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Avg Monthly Sales",
                    f"{territory_data['Actual_Sales'].mean():,.0f}",
                    help="Average monthly sales performance"
                )
                
            with col2:
                st.metric(
                    "Sales Volatility",
                    f"{territory_data['Actual_Sales'].std():,.0f}",
                    help="Standard deviation of monthly sales"
                )
                
            with col3:
                win_rate_trend = territory_data['Predicted_Win_Rate'].iloc[-1] - territory_data['Predicted_Win_Rate'].iloc[0]
                st.metric(
                    "Win Rate Trend",
                    f"{territory_data['Predicted_Win_Rate'].iloc[-1]:.1%}",
                    delta=f"{win_rate_trend:+.1%}",
                    help="Latest win rate vs period start"
                )

        else:
            st.info("📝 Historical trend analysis available for years with complete monthly data")
            
        # Benchmark comparison
        if baseline is not None:
            st.subheader("🔍 Benchmark Comparison")
            
            comparison = pd.DataFrame({
                "Metric": ["Market Share", "Sales Volume", "Win Rate", "Pipeline Need"],
                "Latest Month": [
                    f"{baseline['Predicted_Market_Share']:.1f}%",
                    f"{baseline['Predicted_Sales']:,.0f}",
                    f"{baseline['Predicted_Win_Rate']:.1%}",
                    f"{baseline['Adjusted_Pipeline_Need']:,.0f}"
                ],
                "Annual Calculation": [
                    f"{current_market_share:.1f}%",
                    f"{predicted_sales:,.0f}",
                    f"{predicted_win_rate:.1%}",
                    f"{adjusted_pipeline_need:,.0f}"
                ],
                "Difference": [
                    f"{current_market_share - baseline['Predicted_Market_Share']:+.1f}%",
                    f"{predicted_sales - baseline['Predicted_Sales']:+,.0f}",
                    f"{predicted_win_rate - baseline['Predicted_Win_Rate']:+.1%}",
                    f"{adjusted_pipeline_need - baseline['Adjusted_Pipeline_Need']:+,.0f}"
                ]
            })

            st.dataframe(comparison, use_container_width=True)
            
            st.caption("💡 Compares annual aggregated calculations with latest monthly predictions")

    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Results")

    # Create comprehensive Excel export
    buffer = io.BytesIO()

    try:
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Original territory data
            if not territory_data.empty:
                territory_data.to_excel(writer, sheet_name='Monthly Data', index=False)

            # Calculation results
            results_df = pd.DataFrame({
                'Territory': [territory],
                'Segment': [segment],
                'Analysis_Year': [selected_year],
                'Industry_Qty': [industry_qty],
                'Target_Market_Share_Pct': [target_market_share],
                'Predicted_Sales': [predicted_sales],
                'Predicted_Win_Rate': [predicted_win_rate],
                'Predicted_Residual': [predicted_residual],
                'Actual_Residual': [actual_residual],
                'Current_Market_Share_Pct': [current_market_share],
                'Target_Volume': [target_volume],
                'Sales_Gap': [sales_gap],
                'New_Business_Pipeline': [new_business_pipeline],
                'Base_Pipeline_Need': [base_pipeline_need],
                'Sales_Gap_Factor': [sales_gap_factor],
                'Adjusted_Pipeline_Need': [adjusted_pipeline_need],
                'Data_Source': ['Estimated' if use_estimated_market else 'Actual'],
                'Model_Used': ['ML Predictions' if use_model_predictions else 'Manual Override'],
                'Export_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })

            results_df.to_excel(writer, sheet_name='Pipeline Results', index=False)
            calculation_steps.to_excel(writer, sheet_name='Calculation Steps', index=False)

        buffer.seek(0)

        # Download button
        st.sidebar.download_button(
            label="📊 Download Full Analysis",
            data=buffer,
            file_name=f"pipeline_analysis_{territory}_{segment}_{selected_year}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.ms-excel"
        )
        
    except Exception as e:
        st.sidebar.error(f"Export error: {str(e)}")

else:
    # Enhanced error state
    st.error("❌ Unable to load pipeline data")
    st.markdown("""
    ### 🔧 Troubleshooting Steps:
    
    1. **Check Data Files**: Ensure pipeline results exist in `../results/`
    2. **Run Pipeline Calculation**: Execute the main analysis notebook
    3. **Verify File Permissions**: Check that files are accessible
    4. **Contact Support**: If issues persist, contact the analytics team
    
    **Expected Files:**
    - `../results/full_pipeline_results.xlsx` (preferred)
    - `../results/test_pipeline_results.xlsx` (fallback)
    """)

# App footer
st.sidebar.markdown("---")
st.sidebar.image("https://www.seekpng.com/png/detail/209-2091306_rheem-logo.png", use_container_width=True)
st.sidebar.markdown("© 2025 Rheem Water Heater Division")
st.sidebar.markdown("*Advanced Analytics & Machine Learning*")