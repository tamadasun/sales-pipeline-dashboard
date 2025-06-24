import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
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
    page_title="Water Heater Sales Pipeline Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Water Heater Sales Pipeline and Performance Analysis")

# Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Pipeline Calculator"):
    st.switch_page("pages/pipeline_calculator.py")

st.markdown(
    """
    This application provides interactive analysis of Water Heater Division pipeline and market share data, focusing on
    residential (RWH-5M) and commercial (CWH-5-M) segments.

    Use the navigation menu on the left to explore different parts of the analysis:

    - **Pipeline Calculator**: Calculate territory-specific pipeline based on predictive models
    - **Market Share Analysis**: Analyze current and forecasted market share by territory
    - **Territory Dashboard**: Detailed performance analysis for each territory
    - **Feature Importance**: Understand key drivers of sales, win rate, and residual business
    - **Time Series Analysis**: Explore performance trends over time
    """
)

# Load and display summary data
@st.cache_data
def load_data():
    """
    Load pipeline results data with fallback options for missing files
    """
    data_sources = [
        "results/full_pipeline_results.xlsx",
        "results/test_pipeline_results.xlsx"
    ]

    # data_sources = [
    #     "../results/full_pipeline_results.xlsx",
    #     "../results/test_pipeline_results.xlsx"
    # ]
    
    
    for source in data_sources:
        try:
            path = get_data_path(source)
            df = pd.read_excel(path)
            #df = pd.read_excel(source)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            
            # Ensure Industry_Qty has minimum value of 1 and handle missing values
            if 'Industry_Qty' in df.columns:
                df['Industry_Qty'] = df['Industry_Qty'].fillna(1).clip(lower=1)
            else:
                # If Industry_Qty column is missing, create a default
                st.warning("Industry_Qty column not found. Using default values.")
                df['Industry_Qty'] = 1000  # Default industry quantity
            
            # Validate required columns exist
            required_columns = ['Territory', 'Segment', 'Current_Market_Share', 
                              'Adjusted_Pipeline_Need', 'Predicted_Win_Rate', 'Sales_Gap']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Missing columns in data: {missing_columns}. Some features may not work correctly.")
                # Add missing columns with default values
                for col in missing_columns:
                    if 'Market_Share' in col:
                        df[col] = 15.0  # Default market share
                    elif 'Pipeline' in col:
                        df[col] = 1000  # Default pipeline need
                    elif 'Win_Rate' in col:
                        df[col] = 0.4   # Default win rate
                    else:
                        df[col] = 0     # Default numeric value
            
            st.success(f"✅ Data loaded successfully from {source}")
            return df
            
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error loading {source}: {str(e)}")
            continue
    
    # If no data files found, create sample data
    st.error("❌ No pipeline results files found. Creating sample data for demonstration.")
    return create_sample_data()

def create_sample_data():
    """
    Create sample data when no pipeline results are available
    """
    territories = ['A04', 'A09', 'C01', 'E11', 'G06', 'W01']
    segments = ['Commercial', 'Residential']
    dates = pd.date_range('2023-01-01', '2024-12-01', freq='MS')
    
    sample_data = []
    
    for territory in territories:
        for segment in segments:
            for date in dates:
                sample_data.append({
                    'Territory': territory,
                    'Segment': segment,
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Current_Market_Share': np.random.uniform(10, 40),
                    'Target_Market_Share': 30.0,
                    'Adjusted_Pipeline_Need': np.random.uniform(500, 5000),
                    'Base_Pipeline_Need': np.random.uniform(400, 4000),
                    'Predicted_Win_Rate': np.random.uniform(0.2, 0.8),
                    'Actual_Win_Rate': np.random.uniform(0.2, 0.8),
                    'Sales_Gap': np.random.uniform(-500, 1000),
                    'Industry_Qty': np.random.uniform(500, 2000),
                    'Predicted_Sales': np.random.uniform(100, 1500),
                    'Actual_Sales': np.random.uniform(100, 1500)
                })
    
    return pd.DataFrame(sample_data)

# Load data
data = load_data()

if data is not None and not data.empty:
    # Data overview
    st.sidebar.markdown("### Data Overview")
    st.sidebar.write(f"📊 **Records**: {len(data):,}")
    st.sidebar.write(f"📅 **Date Range**: {data['Date'].min().strftime('%Y-%m')} to {data['Date'].max().strftime('%Y-%m')}")
    st.sidebar.write(f"🏢 **Territories**: {data['Territory'].nunique()}")
    st.sidebar.write(f"📦 **Segments**: {data['Segment'].nunique()}")
    
    # Year selection with validation
    years = sorted(data['Year'].unique())
    
    # Determine data availability for each year
    year_data_status = {}
    for year in years:
        year_data = data[data['Year'] == year]
        has_industry_data = not year_data['Industry_Qty'].isnull().all()
        has_complete_data = len(year_data) >= 12  # At least 12 months of data
        year_data_status[year] = {
            'has_industry_data': has_industry_data,
            'has_complete_data': has_complete_data,
            'months_available': len(year_data['Month'].unique())
        }
    
    # Add future year options
    future_years = [2025, 2026] if max(years) < 2025 else [max(years) + 1]
    all_years = years + future_years
    
    selected_year = st.sidebar.selectbox(
        "Select Year",
        options=all_years,
        index=len(years)-1 if years else 0  # Default to most recent historical year
    )
    
    # Display data status for selected year
    if selected_year in year_data_status:
        status = year_data_status[selected_year]
        if status['has_complete_data']:
            st.sidebar.success(f"✅ {selected_year}: Complete data ({status['months_available']} months)")
        elif status['has_industry_data']:
            st.sidebar.warning(f"⚠️ {selected_year}: Partial data ({status['months_available']} months)")
        else:
            st.sidebar.warning(f"⚠️ {selected_year}: Limited data available")
    else:
        st.sidebar.info(f"ℹ️ {selected_year}: Future year (estimated data)")
    
    # Handle data filtering based on year selection
    if selected_year in years:
        filtered_data = data[data['Year'] == selected_year]
        use_estimated_data = False
    else:
        # For future years, use the most recent year's data as baseline
        st.warning(f"📊 Using {max(years)} data as baseline for {selected_year} projections")
        filtered_data = data[data['Year'] == max(years)]
        use_estimated_data = True
    
    # Enhanced filters with Select All functionality
    st.sidebar.markdown("### 🎛️ Filters")
    
    # Territory filter with Select All option
    available_territories = sorted(filtered_data['Territory'].unique())
    
    select_all_territories = st.sidebar.checkbox("🌍 Select All Territories", value=True)
    
    if select_all_territories:
        selected_territories = available_territories
        st.sidebar.success(f"✅ All {len(available_territories)} territories selected")
    else:
        selected_territories = st.sidebar.multiselect(
            "Choose Specific Territories",
            options=available_territories,
            default=[],
            help="Select individual territories to include in analysis"
        )
        
        if not selected_territories:
            st.sidebar.warning("⚠️ No territories selected!")
    
    # Segment filter with Select All option  
    available_segments = sorted(filtered_data['Segment'].unique())
    
    select_all_segments = st.sidebar.checkbox("📦 Select All Segments", value=True)
    
    if select_all_segments:
        selected_segments = available_segments
        st.sidebar.success(f"✅ All {len(available_segments)} segments selected")
    else:
        selected_segments = st.sidebar.multiselect(
            "Choose Specific Segments",
            options=available_segments,
            default=[],
            help="Select individual business segments to analyze"
        )
        
        if not selected_segments:
            st.sidebar.warning("⚠️ No segments selected!")
    
    # Apply filters
    filtered_data = filtered_data[
        (filtered_data['Territory'].isin(selected_territories)) &
        (filtered_data['Segment'].isin(selected_segments))
    ]
    
    if filtered_data.empty:
        st.error("No data available for the selected filters. Please adjust your selection.")
        st.stop()

    # Key metrics for the dashboard
    st.subheader(f"📈 Key Performance Metrics - {selected_year}")
    
    if use_estimated_data:
        st.info(f"💡 Showing {max(years)} actuals as baseline for {selected_year} planning")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_market_share = filtered_data['Current_Market_Share'].mean()
        market_share_delta = avg_market_share - 30  # Target is 30%
        
        st.metric(
            "Avg Market Share",
            f"{avg_market_share:.1f}%",
            delta=f"{market_share_delta:+.1f}%",
            delta_color="normal" if market_share_delta >= 0 else "inverse",
            help=f"Average market share across selected territories and segments. Target: 30%"
        )
        
    with col2:
        total_pipeline = filtered_data['Adjusted_Pipeline_Need'].sum()
        base_pipeline = filtered_data['Base_Pipeline_Need'].sum() if 'Base_Pipeline_Need' in filtered_data.columns else total_pipeline * 0.8
        pipeline_delta = total_pipeline - base_pipeline
        
        st.metric(
            "Total Pipeline Need",
            f"{total_pipeline:,.0f}",
            delta=f"{pipeline_delta:+,.0f}",
            help="Total adjusted pipeline need across all selected territories and segments"
        )
        
    with col3:
        avg_predicted_win_rate = filtered_data['Predicted_Win_Rate'].mean()
        avg_actual_win_rate = filtered_data['Actual_Win_Rate'].mean() if 'Actual_Win_Rate' in filtered_data.columns else avg_predicted_win_rate * 0.95
        win_rate_delta = avg_predicted_win_rate - avg_actual_win_rate
        
        st.metric(
            "Avg Win Rate",
            f"{avg_predicted_win_rate:.1%}",
            delta=f"{win_rate_delta:+.1%}",
            delta_color="normal",
            help="Average predicted win rate across territories"
        )
        
    with col4:
        total_sales_gap = filtered_data['Sales_Gap'].sum()
        
        st.metric(
            "Total Sales Gap",
            f"{total_sales_gap:+,.0f}",
            help="Total gap between target volume and predicted sales"
        )

    st.markdown("---")
    
    # Market Share Overview
    st.subheader("🎯 Market Share Analysis")

    # Create market share visualization
    fig_market = px.bar(
        filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index(),
        x="Territory",
        y="Current_Market_Share",
        color="Segment",
        barmode="group",
        title=f"Average Market Share by Territory and Segment ({selected_year})",
        labels={"Current_Market_Share": "Market Share (%)"},
        color_discrete_map={"Commercial": "#1f77b4", "Residential": "#ff7f0e"}
    )

    # Add target line at 30%
    fig_market.add_hline(
        y=30,
        line_dash="dash",
        line_color="red",
        annotation_text="Target (30%)",
        annotation_position="top right"
    )

    # Customize layout
    fig_market.update_layout(
        height=500,
        xaxis_title="Territory",
        yaxis_title="Market Share (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig_market, use_container_width=True)

    # Market share insights with improved logic
    top_performers = filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index()
    top_performers = top_performers.sort_values('Current_Market_Share', ascending=False).head(5)

    # Filter for territories below target AND exclude negative/zero market shares for cleaner display
    underperformers = filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index()
    underperformers = underperformers[
        (underperformers['Current_Market_Share'] < 30) & 
        (underperformers['Current_Market_Share'] > 0)  # Exclude negative values
    ].sort_values('Current_Market_Share', ascending=True).head(5)

    # Identify territories with data quality issues
    data_quality_issues = filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index()
    data_quality_issues = data_quality_issues[data_quality_issues['Current_Market_Share'] <= 0]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 🏆 Top Performers (Market Share)")
        for _, row in top_performers.iterrows():
            st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Current_Market_Share']:.1f}%")

    with col2:
        if not underperformers.empty:
            st.markdown("##### 📈 Growth Opportunities (Below Target)")
            for _, row in underperformers.iterrows():
                gap = 30 - row['Current_Market_Share']
                st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Current_Market_Share']:.1f}% (gap: {gap:.1f}%)")
        else:
            st.success("🎉 All selected territories are meeting market share targets!")

    # Add data quality alert if needed
    if not data_quality_issues.empty:
        st.warning(f"⚠️ Data Quality Alert: {len(data_quality_issues)} territory-segment combinations have negative or zero market share")
        with st.expander("View Data Quality Issues"):
            for _, row in data_quality_issues.iterrows():
                st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Current_Market_Share']:.1f}%")
    
    # # Market share insights
    # top_performers = filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index()
    # top_performers = top_performers.sort_values('Current_Market_Share', ascending=False).head(5)
    
    # underperformers = filtered_data.groupby(['Territory', 'Segment'])['Current_Market_Share'].mean().reset_index()
    # underperformers = underperformers[underperformers['Current_Market_Share'] < 30].sort_values('Current_Market_Share', ascending=True).head(5)
    
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.markdown("##### 🏆 Top Performers (Market Share)")
    #     for _, row in top_performers.iterrows():
    #         st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Current_Market_Share']:.1f}%")
    
    # with col2:
    #     if not underperformers.empty:
    #         st.markdown("##### 📈 Growth Opportunities (Below Target)")
    #         for _, row in underperformers.iterrows():
    #             gap = 30 - row['Current_Market_Share']
    #             st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Current_Market_Share']:.1f}% (gap: {gap:.1f}%)")
    #     else:
    #         st.success("🎉 All selected territories are meeting market share targets!")

    st.markdown("---")
    
    # Pipeline Need Overview
    st.subheader("🔄 Pipeline Requirements")

    # Create pipeline visualization with log scale for better readability
    pipeline_summary = filtered_data.groupby(['Territory', 'Segment'])['Adjusted_Pipeline_Need'].sum().reset_index()
    
    fig_pipeline = px.bar(
        pipeline_summary,
        x="Territory",
        y="Adjusted_Pipeline_Need",
        color="Segment",
        barmode="group",
        title=f"Annual Pipeline Need by Territory and Segment ({selected_year})",
        labels={"Adjusted_Pipeline_Need": "Pipeline Need (Units)"},
        color_discrete_map={"Commercial": "#1f77b4", "Residential": "#ff7f0e"}
    )

    # Use log scale if there's a wide range of values
    pipeline_range = pipeline_summary['Adjusted_Pipeline_Need'].max() / pipeline_summary['Adjusted_Pipeline_Need'].min()
    if pipeline_range > 100:  # If max is more than 100x the min
        fig_pipeline.update_yaxis(type="log")
        st.info("📊 Using logarithmic scale due to wide range of pipeline values")

    fig_pipeline.update_layout(
        height=500,
        xaxis_title="Territory",
        yaxis_title="Pipeline Need (Units)",
        showlegend=True
    )
    
    st.plotly_chart(fig_pipeline, use_container_width=True)
    
    # Pipeline insights
    high_pipeline = pipeline_summary.sort_values('Adjusted_Pipeline_Need', ascending=False).head(5)
    
    st.markdown("##### 🎯 Highest Pipeline Requirements")
    for _, row in high_pipeline.iterrows():
        st.write(f"• **{row['Territory']} {row['Segment']}**: {row['Adjusted_Pipeline_Need']:,.0f} units")

    # Summary statistics
    st.markdown("---")
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Market Share")
        st.write(f"**Average**: {filtered_data['Current_Market_Share'].mean():.1f}%")
        st.write(f"**Median**: {filtered_data['Current_Market_Share'].median():.1f}%")
        st.write(f"**Range**: {filtered_data['Current_Market_Share'].min():.1f}% - {filtered_data['Current_Market_Share'].max():.1f}%")
        
    with col2:
        st.markdown("##### Win Rate")
        st.write(f"**Average**: {filtered_data['Predicted_Win_Rate'].mean():.1%}")
        st.write(f"**Median**: {filtered_data['Predicted_Win_Rate'].median():.1%}")
        st.write(f"**Range**: {filtered_data['Predicted_Win_Rate'].min():.1%} - {filtered_data['Predicted_Win_Rate'].max():.1%}")
        
    with col3:
        st.markdown("##### Pipeline Need")
        st.write(f"**Total**: {filtered_data['Adjusted_Pipeline_Need'].sum():,.0f}")
        st.write(f"**Average**: {filtered_data['Adjusted_Pipeline_Need'].mean():,.0f}")
        st.write(f"**Median**: {filtered_data['Adjusted_Pipeline_Need'].median():,.0f}")

else:
    st.error("❌ Unable to load data. Please check that the required data files exist.")
    st.markdown("""
    **Expected data files:**
    - `../results/full_pipeline_results.xlsx`
    - `../results/test_pipeline_results.xlsx`
    
    **Please ensure:**
    1. Pipeline calculation has been run successfully
    2. Results files are generated in the correct location
    3. Files contain the required columns
    """)

# App footer
st.markdown("---")
st.markdown("### 📚 Data Sources")
st.markdown("""
- **Sales Data**: Oracle database (historical and current year)
- **Industry Data**: Market research provided by Mike Rubino team
- **Economic Indicators**: Housing starts and other external economic factors
- **Weather Data**: NOAA climate data for seasonal adjustments
""")

st.markdown("### 🤖 Model Information")
st.markdown("""
This application uses a sophisticated **three-model predictive framework**:

- **🔢 Sales Volume Model**: Predicts expected sales by territory and segment (R² = 98.9%)
- **🎯 Win Rate Model**: Forecasts conversion rates for sales opportunities (R² = 98.4%)  
- **🔄 Residual Business Model**: Estimates recurring business from existing accounts (R² = 58.9%)

All models use **segment-specific architecture** to provide separate predictions for Commercial and Residential segments.
""")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.image("https://www.seekpng.com/png/detail/209-2091306_rheem-logo.png", use_container_width=True)
st.sidebar.markdown("© 2025 Rheem Water Heater Division")
st.sidebar.markdown("*Advanced Analytics & Machine Learning*")