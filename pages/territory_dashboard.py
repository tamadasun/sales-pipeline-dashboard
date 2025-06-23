import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Territory Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🗺️ Territory Dashboard")

# Create a clean, professional redirect page
st.markdown("""
<div style="
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem;
    border-radius: 10px;
    text-align: center;
    margin: 2rem 0;
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
">
    <h2 style="margin-bottom: 1rem; color: white;">🚀 Enhanced Territory Dashboard Available</h2>
    <p style="font-size: 1.2rem; margin-bottom: 2rem; color: white;">
        Our comprehensive territory analysis with interactive maps, geographic insights, 
        and advanced visualizations will be available in Power BI for enhanced performance and functionality in the future.
    </p>
</div>
""", unsafe_allow_html=True)

# Feature highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🗺️ Interactive Maps
    - **Geographic Territory Visualization**
    - **Performance Heat Maps**
    - **State, City & County Drill-downs**
    - **Custom Territory Boundaries**
    """)

with col2:
    st.markdown("""
    ### 📊 Advanced Analytics
    - **Real-time Performance Metrics**
    - **Territory Comparison Tools**
    - **Market Share Analysis**
    - **Pipeline Optimization Views**
    """)

with col3:
    st.markdown("""
    ### 🔄 Live Data Integration
    - **Automatic Data Refresh**
    - **Cross-functional Dashboards**
    - **Executive Reporting**
    - **Mobile Accessibility**
    """)

# Call-to-action section
st.markdown("---")
st.markdown("### 🎯 Access Territory Dashboard")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        text-align: center;
    ">
        <h4>Ready to explore your territories?</h4>
        <p>The link below will allow access to the full Territory Dashboard in Power BI in the future</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Power BI link button (replace with actual URL when completed)
    #power_bi_url = "https://app.powerbi.com/groups/me/dashboards/your-dashboard-id"
    power_bi_url = "https://app.powerbi.com/home?experience=power-bi"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="{power_bi_url}" target="_blank" style="
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 15px 30px;
            font-size: 18px;
            font-weight: bold;
            text-decoration: none;
            border-radius: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            display: inline-block;
        ">
            🚀 Open Territory Dashboard in Power BI
        </a>
    </div>
    """, unsafe_allow_html=True)

# Benefits section
st.markdown("---")
st.markdown("### ✨ Why Power BI for Territory Analysis?")

benefits_col1, benefits_col2 = st.columns(2)

with benefits_col1:
    st.markdown("""
    **🎯 Enhanced Performance**
    - Faster load times for large datasets
    - Real-time data processing
    - Optimized for geographic visualizations
    
    **🔧 Advanced Features**
    - Custom territory mapping
    - Drill-through capabilities
    - Advanced filtering options
    """)

with benefits_col2:
    st.markdown("""
    **👥 Collaboration**
    - Share dashboards across teams
    - Comment and annotation features
    - Automated report distribution
    
    **📱 Accessibility**
    - Mobile-optimized interface
    - Offline access capabilities
    - Cross-platform compatibility
    """)

# Quick stats (if data available)
try:
    # Try to load basic stats for display
    data_sources = [
        "../results/full_pipeline_results.xlsx",
        "../results/test_pipeline_results.xlsx"
    ]
    
    for source in data_sources:
        try:
            df = pd.read_excel(source)
            st.markdown("---")
            st.markdown("### 📈 Current Data Overview")
            
            #col1, col2, col3, col4 = st.columns(4)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Territories", df['Territory'].nunique())
            #with col2:
                #st.metric("Business Segments", df['Segment'].nunique() if 'Segment' in df.columns else 2)
            with col2:
                st.metric("Data Records", f"{len(df):,}")
            with col3:
                date_range = f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}" if 'Date' in df.columns else "2023-2024"
                st.metric("Date Range", date_range)
            break
            
        except:
            continue
            
except:
    pass

# Support section
st.markdown("---")
st.markdown("### 🆘 Need Help?")

support_col1, support_col2 = st.columns(2)

with support_col1:
    st.markdown("""
    **📧 Contact Information**
    - **IT and Analytics Support**: anthony.amadasun@rheem.com
    """)

with support_col2:
    st.markdown("""
    **📚 Resources**
    - [Power BI User Guide](https://learn.microsoft.com/en-us/power-bi/)
    - [Territory Analysis Documentation](#)
    - [Training Videos](#)
    """)

# Navigation helper
st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.markdown("""
**Available Dashboards:**
- 🏠 **Home**: Overview and key metrics
- 🧮 **Pipeline Calculator**: Territory calculations
- 🗺️ **Territory Dashboard**: Geographic analysis (Power BI)
- 📈 **Time Series Analysis**: Forecasting and trends
- 🔍 **Feature Importance**: Model insights
""")

if st.sidebar.button("🧮 Go to Pipeline Calculator"):
    st.switch_page("pages/pipeline_calculator.py")

if st.sidebar.button("📈 Go to Time Series Analysis"):
    st.switch_page("pages/time_series_analysis.py")

# App footer
st.sidebar.markdown("---")
st.sidebar.image("https://www.seekpng.com/png/detail/209-2091306_rheem-logo.png", use_container_width=True)
st.sidebar.markdown("© 2025 Rheem Water Heater Division")
st.sidebar.markdown("*Advanced Analytics & Machine Learning*")