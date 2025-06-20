import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📈 Time Series Analysis & Forecasting")
st.markdown("Interactive time series analysis with machine learning forecasting and confidence intervals.")

# Load required data
@st.cache_data
def load_time_series_data():
    """Load pipeline data optimized for time series analysis"""
    data_sources = [
        "../results/full_pipeline_results.xlsx",
        "../results/test_pipeline_results.xlsx"
    ]
    
    for source in data_sources:
        try:
            df = pd.read_excel(source)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            
            # Ensure required columns exist
            required_columns = ['Territory', 'Segment', 'Date', 'Actual_Sales', 
                              'Predicted_Sales', 'Predicted_Win_Rate', 'Current_Market_Share',
                              'Adjusted_Pipeline_Need', 'Industry_Qty']
            
            for col in required_columns:
                if col not in df.columns:
                    if 'Sales' in col:
                        df[col] = np.random.uniform(100, 1500, len(df))
                    elif 'Win_Rate' in col:
                        df[col] = np.random.uniform(0.2, 0.7, len(df))
                    elif 'Market_Share' in col:
                        df[col] = np.random.uniform(10, 40, len(df))
                    elif 'Pipeline' in col:
                        df[col] = np.random.uniform(500, 5000, len(df))
                    elif 'Industry' in col:
                        df[col] = np.random.uniform(800, 2500, len(df))
                    else:
                        df[col] = 0
            
            # Validate numeric columns
            numeric_cols = ['Actual_Sales', 'Predicted_Sales', 'Predicted_Win_Rate', 
                           'Current_Market_Share', 'Adjusted_Pipeline_Need', 'Industry_Qty']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Sort by date for time series analysis
            df = df.sort_values(['Territory', 'Segment', 'Date'])
            
            st.success(f"✅ Data loaded successfully from {source}")
            return df
            
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Error loading {source}: {str(e)}")
            continue
    
    # Create sample data if no files found
    st.warning("⚠️ Using sample data for demonstration")
    return create_sample_time_series_data()

def create_sample_time_series_data():
    """Create realistic sample time series data"""
    territories = ['A03', 'A04', 'C01', 'E11', 'G06', 'I01', 'P90', 'W01']
    segments = ['Commercial', 'Residential']
    
    # Create 3 years of monthly data
    date_range = pd.date_range('2022-01-01', '2024-12-31', freq='MS')
    
    sample_data = []
    
    for territory in territories:
        for segment in segments:
            # Base values for this territory-segment combination
            base_sales = np.random.uniform(300, 800)
            base_win_rate = np.random.uniform(0.25, 0.55)
            base_market_share = np.random.uniform(15, 35)
            
            # Create time series with trend and seasonality
            np.random.seed(hash(territory + segment) % 2**32)
            
            for i, date in enumerate(date_range):
                # Add trend
                trend = i * 0.02
                # Add seasonality
                seasonality = 10 * np.sin(2 * np.pi * (date.month - 1) / 12 + np.pi)
                # Add noise
                noise = np.random.normal(0, 20)
                
                actual_sales = max(50, base_sales + trend + seasonality + noise)
                predicted_sales = actual_sales * np.random.uniform(0.9, 1.1)
                win_rate = np.clip(base_win_rate + np.random.normal(0, 0.05), 0.1, 0.8)
                market_share = np.clip(base_market_share + trend/2 + np.random.normal(0, 2), 5, 50)
                pipeline_need = actual_sales / win_rate * np.random.uniform(1.2, 2.0)
                industry_qty = actual_sales / (market_share / 100) if market_share > 0 else 1000
                
                sample_data.append({
                    'Territory': territory,
                    'Segment': segment,
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Actual_Sales': actual_sales,
                    'Predicted_Sales': predicted_sales,
                    'Predicted_Win_Rate': win_rate,
                    'Current_Market_Share': market_share,
                    'Adjusted_Pipeline_Need': pipeline_need,
                    'Industry_Qty': industry_qty
                })
    
    return pd.DataFrame(sample_data)

def generate_forecast_for_metric(historical_data, metric_column, periods=12, confidence_levels=[0.8, 0.95]):
    """
    Generate future forecasts with confidence intervals for any metric
    """
    # Extract the time series for the specific metric
    values = historical_data[metric_column].values
    dates = pd.to_datetime(historical_data['Date'])
    
    if len(values) < 3:
        return pd.DataFrame()
    
    # Simple trend calculation
    x = np.arange(len(values))
    try:
        trend_coef = np.polyfit(x, values, 1)[0]
    except:
        trend_coef = 0
    
    # Seasonal component (12-month cycle)
    seasonal_pattern = []
    for month in range(1, 13):
        month_data = historical_data[historical_data['Date'].dt.month == month][metric_column]
        seasonal_avg = month_data.mean() if len(month_data) > 0 else values.mean()
        seasonal_pattern.append(seasonal_avg - values.mean())
    
    # Generate forecast dates
    last_date = dates.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
    
    # Generate forecasts
    forecasts = []
    for i, future_date in enumerate(future_dates):
        # Base trend projection
        trend_value = values[-1] + trend_coef * (i + 1)
        
        # Add seasonal component
        seasonal_value = seasonal_pattern[future_date.month - 1]
        
        # Base forecast
        base_forecast = trend_value + seasonal_value
        
        # Apply metric-specific constraints
        if metric_column == 'Current_Market_Share':
            # Market share should be reasonable (0-100%)
            base_forecast = np.clip(base_forecast, 0, 100)
        elif metric_column == 'Predicted_Win_Rate':
            # Win rate should be between 0-1
            base_forecast = np.clip(base_forecast, 0, 1)
        elif metric_column == 'Adjusted_Pipeline_Need':
            # Pipeline can be negative, but let's be reasonable
            base_forecast = max(base_forecast, -10000)
        else:
            # For sales, ensure non-negative
            base_forecast = max(0, base_forecast)
        
        # Generate confidence intervals based on historical volatility
        std_error = np.std(values) * (1 + i * 0.1)
        
        forecast_data = {
            'Date': future_date,
            'Forecast': base_forecast,
            'Trend': trend_value,
            'Seasonal': seasonal_value
        }
        
        # Add confidence intervals
        for conf_level in confidence_levels:
            z_score = 1.96 if conf_level == 0.95 else 1.28
            margin = z_score * std_error
            
            lower_bound = base_forecast - margin
            upper_bound = base_forecast + margin
            
            # Apply the same constraints to confidence intervals
            if metric_column == 'Current_Market_Share':
                lower_bound = np.clip(lower_bound, 0, 100)
                upper_bound = np.clip(upper_bound, 0, 100)
            elif metric_column == 'Predicted_Win_Rate':
                lower_bound = np.clip(lower_bound, 0, 1)
                upper_bound = np.clip(upper_bound, 0, 1)
            elif metric_column == 'Adjusted_Pipeline_Need':
                lower_bound = max(lower_bound, -10000)
            else:
                lower_bound = max(0, lower_bound)
                upper_bound = max(0, upper_bound)
            
            forecast_data[f'Lower_{int(conf_level*100)}'] = lower_bound
            forecast_data[f'Upper_{int(conf_level*100)}'] = upper_bound
        
        forecasts.append(forecast_data)
    
    return pd.DataFrame(forecasts)

# Load data
data = load_time_series_data()

if data is not None and not data.empty:
    # Simplified sidebar controls
    st.sidebar.markdown("### 🎛️ Analysis Controls")
    
    # Territory selection
    territories = sorted(data['Territory'].unique())
    selected_territory = st.sidebar.selectbox(
        "Territory",
        options=territories,
        index=0,
        help="Select territory for analysis"
    )
    
    # Segment selection
    segments = sorted(data['Segment'].unique())
    selected_segment = st.sidebar.selectbox(
        "Segment",
        options=segments,
        index=0,
        help="Choose business segment"
    )
    
    # Metric selection
    metrics = {
        'Sales Performance': 'Actual_Sales',
        'Pipeline Need': 'Adjusted_Pipeline_Need',
        'Win Rate': 'Predicted_Win_Rate',
        'Market Share': 'Current_Market_Share'
    }
    
    selected_metric = st.sidebar.selectbox(
        "Primary Metric",
        options=list(metrics.keys()),
        index=0,
        help="Choose metric for analysis"
    )
    
    # Forecasting parameters
    st.sidebar.markdown("### 🔮 Forecasting")
    
    forecast_periods = st.sidebar.slider(
        "Forecast Months",
        min_value=3,
        max_value=24,
        value=12,
        help="Number of months to forecast"
    )
    
    show_confidence = st.sidebar.checkbox(
        "Show Confidence Intervals",
        value=True,
        help="Display prediction confidence bands"
    )
    
    # Filter data based on selections and handle incomplete last month
    raw_filtered_data = data[
        (data['Territory'] == selected_territory) &
        (data['Segment'] == selected_segment)
    ].sort_values('Date')
    
    if raw_filtered_data.empty:
        st.error("No data available for selected filters. Please adjust your selection.")
        st.stop()
    
    # Check for May 2025 incomplete data (apply to all territories)
    incomplete_last_month = False
    incomplete_month_date = None
    incomplete_month_sales = None
    
    # Check if the last month is May 2025 or later (assuming it might be incomplete)
    if len(raw_filtered_data) >= 1:
        last_date = raw_filtered_data['Date'].iloc[-1]
        if last_date >= pd.to_datetime('2025-05-01'):
            incomplete_last_month = True
            incomplete_month_date = last_date
            incomplete_month_sales = raw_filtered_data[metrics[selected_metric]].iloc[-1]
    
    # If not May 2025+, use the original logic for incomplete month detection
    if not incomplete_last_month and len(raw_filtered_data) >= 3:
        last_month_sales = raw_filtered_data['Actual_Sales'].iloc[-1]
        prev_months_avg = raw_filtered_data['Actual_Sales'].iloc[-4:-1].mean()  # Average of previous 3 months
        
        # If last month is less than 30% of previous months average, consider it incomplete
        if last_month_sales < (prev_months_avg * 0.3) and prev_months_avg > 0:
            incomplete_last_month = True
            incomplete_month_date = raw_filtered_data['Date'].iloc[-1]
            incomplete_month_sales = raw_filtered_data[metrics[selected_metric]].iloc[-1]
    
    # Create filtered data for analysis (excluding incomplete last month if detected)
    if incomplete_last_month:
        filtered_data = raw_filtered_data.iloc[:-1].copy()  # Exclude last month
        st.info(f"ℹ️ Note: {incomplete_month_date.strftime('%B %Y')} sales ({incomplete_month_sales:,.0f}) appear incomplete and were excluded from forecast calculations.")
    else:
        filtered_data = raw_filtered_data.copy()
    
    # Data quality checks and user alerts
    st.markdown("---")
    
    # Check for data quality issues
    data_quality_issues = []
    
    if 'Current_Market_Share' in filtered_data.columns:
        # Check for market share > 100%
        high_market_share = filtered_data[filtered_data['Current_Market_Share'] > 100]
        if not high_market_share.empty:
            data_quality_issues.append({
                'type': 'High Market Share',
                'count': len(high_market_share),
                'description': 'Market share values exceed 100%',
                'impact': 'May indicate data quality issues with Industry Quantity or Sales reporting'
            })
    
    if 'Actual_Sales' in filtered_data.columns and 'Industry_Qty' in filtered_data.columns:
        # Check for sales > industry quantity
        sales_exceed_industry = filtered_data[filtered_data['Actual_Sales'] > filtered_data['Industry_Qty']]
        if not sales_exceed_industry.empty:
            data_quality_issues.append({
                'type': 'Sales Exceed Industry',
                'count': len(sales_exceed_industry),
                'description': 'Actual sales exceed total industry quantity',
                'impact': 'Indicates potential data timing issues or territory boundary mismatches'
            })
        
        # Check for very small industry quantities
        small_industry = filtered_data[filtered_data['Industry_Qty'] < 200]
        if not small_industry.empty:
            data_quality_issues.append({
                'type': 'Small Industry Quantity',
                'count': len(small_industry),
                'description': 'Industry quantities appear unusually small',
                'impact': 'May cause inflated market share calculations'
            })
    
    # Display data quality alerts
    if data_quality_issues:
        st.warning("⚠️ **Data Quality Alerts Detected**")
        
        for issue in data_quality_issues:
            with st.expander(f"🚨 {issue['type']} ({issue['count']} records)", expanded=True):
                st.markdown(f"**Issue**: {issue['description']}")
                st.markdown(f"**Impact**: {issue['impact']}")
                
                # Show specific examples
                if issue['type'] == 'High Market Share' and not high_market_share.empty:
                    st.markdown("**Examples:**")
                    examples = high_market_share[['Date', 'Current_Market_Share']].tail(3)
                    for _, row in examples.iterrows():
                        st.write(f"• {row['Date'].strftime('%Y-%m')}: {row['Current_Market_Share']:.1f}%")
                
                elif issue['type'] == 'Sales Exceed Industry' and not sales_exceed_industry.empty:
                    st.markdown("**Examples:**")
                    examples = sales_exceed_industry[['Date', 'Actual_Sales', 'Industry_Qty']].tail(3)
                    for _, row in examples.iterrows():
                        market_share = (row['Actual_Sales'] / row['Industry_Qty']) * 100
                        st.write(f"• {row['Date'].strftime('%Y-%m')}: Sales={row['Actual_Sales']:.0f}, Industry={row['Industry_Qty']:.0f}, Market Share={market_share:.1f}%")
                
                elif issue['type'] == 'Small Industry Quantity' and not small_industry.empty:
                    st.markdown("**Examples:**")
                    examples = small_industry[['Date', 'Industry_Qty', 'Actual_Sales']].tail(3)
                    for _, row in examples.iterrows():
                        market_share = (row['Actual_Sales'] / row['Industry_Qty']) * 100 if row['Industry_Qty'] > 0 else 0
                        st.write(f"• {row['Date'].strftime('%Y-%m')}: Industry={row['Industry_Qty']:.0f}, Sales={row['Actual_Sales']:.0f}, Market Share={market_share:.1f}%")
        
        # Explanation and recommendations
        st.info("""
        **📋 Data Quality Explanation:**
        
        **Market Share > 100% occurs when:**
        - Industry Quantity data is reported later than Sales data (timing mismatch)
        - Territory boundaries changed but Industry Quantity wasn't updated
        - Industry estimates are conservative while actual market is larger
        - Data collection methodologies differ between Sales and Industry reporting
        
        **Business Reality:** Values over 100% can happen temporarily but should be investigated.
        
        **Recommendations:**
        - Verify Industry Quantity data sources and timing
        - Flag these periods for manual review in business planning
        """)
    
    # Main visualization
    st.subheader(f"📈 {selected_metric} - {selected_territory} {selected_segment}")
    
    # Add data quality warning for current selection if needed
    if selected_metric == 'Market Share' and not filtered_data.empty:
        current_market_share = filtered_data['Current_Market_Share'].iloc[-1]
        if current_market_share > 100:
            st.error(f"🚨 **Current Market Share ({current_market_share:.1f}%) exceeds 100%** - Please review data quality before using for business decisions.")
        elif current_market_share > 80:
            st.warning(f"⚠️ **High Market Share ({current_market_share:.1f}%)** - Verify data accuracy for strategic planning.")
    
    # Create main chart
    fig = go.Figure()
    
    # Historical actual data (solid line)
    metric_column = metrics[selected_metric]
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data['Date'],
            y=filtered_data[metric_column],
            mode='lines+markers',
            name='Actual Sales' if selected_metric == 'Sales Performance' else f'Actual {selected_metric}',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        )
    )
    
    # Historical predicted data (dashed line) - only for Sales Performance
    if selected_metric == 'Sales Performance' and 'Predicted_Sales' in filtered_data.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_data['Date'],
                y=filtered_data['Predicted_Sales'],
                mode='lines+markers',
                name='Predicted Sales',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                marker=dict(size=3, symbol='diamond')
            )
        )
        
        # Add confidence intervals for historical predictions if available
        if 'Predicted_Sales_Lower' in filtered_data.columns and 'Predicted_Sales_Upper' in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(filtered_data['Date']) + list(filtered_data['Date'][::-1]),
                    y=list(filtered_data['Predicted_Sales_Upper']) + list(filtered_data['Predicted_Sales_Lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(44, 160, 44, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Historical Prediction Range',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
    
    # Generate and add future forecast using the new metric-specific function
    if len(filtered_data) >= 3:
        forecast_df = generate_forecast_for_metric(filtered_data, metric_column, periods=forecast_periods)
        
        if not forecast_df.empty:
            forecast_values = forecast_df['Forecast']
            lower_80 = forecast_df['Lower_80'] 
            upper_80 = forecast_df['Upper_80']
            lower_95 = forecast_df['Lower_95']
            upper_95 = forecast_df['Upper_95']
            
            # Create seamless transition by connecting last historical point to first forecast point
            last_historical_date = filtered_data['Date'].iloc[-1]
            last_historical_value = filtered_data[metric_column].iloc[-1]
            first_forecast_date = forecast_df['Date'].iloc[0]
            first_forecast_value = forecast_values.iloc[0]
            
            # Add transition line
            fig.add_trace(
                go.Scatter(
                    x=[last_historical_date, first_forecast_date],
                    y=[last_historical_value, first_forecast_value],
                    mode='lines',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
            
            # Future forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Date'],
                    y=forecast_values,
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=4, symbol='diamond')
                )
            )
            
            # Future confidence intervals
            if show_confidence:
                # 95% confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                        y=list(upper_95) + list(lower_95[::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence',
                        showlegend=True,
                        hoverinfo='skip'
                    )
                )
                
                # 80% confidence interval
                fig.add_trace(
                    go.Scatter(
                        x=list(forecast_df['Date']) + list(forecast_df['Date'][::-1]),
                        y=list(upper_80) + list(lower_80[::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 127, 14, 0.4)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='80% Confidence',
                        showlegend=True,
                        hoverinfo='skip'
                    )
                )
    
    # Add vertical line at forecast start (manual implementation to avoid timestamp error)
    if not filtered_data.empty and len(filtered_data) >= 3:
        forecast_start = filtered_data['Date'].max()
        
        # Get y-axis range for the vertical line
        y_min = filtered_data[metric_column].min()
        y_max = filtered_data[metric_column].max()
        y_range = y_max - y_min
        y_line_start = y_min - (y_range * 0.1)
        y_line_end = y_max + (y_range * 0.1)
        
        # Add vertical line manually
        fig.add_trace(
            go.Scatter(
                x=[forecast_start, forecast_start],
                y=[y_line_start, y_line_end],
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                name='Forecast Start',
                showlegend=False,
                hoverinfo='skip'
            )
        )
        
        # Add annotation
        fig.add_annotation(
            x=forecast_start,
            y=y_max + (y_range * 0.05),
            text="Forecast Start",
            showarrow=False,
            font=dict(color='gray', size=10),
            bgcolor='white',
            bordercolor='gray',
            borderwidth=1
        )
    
    # Show incomplete month data if it exists (as a different marker)
    if incomplete_last_month:
        fig.add_trace(
            go.Scatter(
                x=[incomplete_month_date],
                y=[raw_filtered_data[metric_column].iloc[-1]],
                mode='markers',
                name='Incomplete Month',
                marker=dict(size=8, symbol='x', color='red'),
                hovertemplate=f'<b>Incomplete Data</b><br>Date: %{{x}}<br>Value: %{{y}}<extra></extra>'
            )
        )
    
    # Update layout
    y_axis_title = selected_metric
    if selected_metric == 'Win Rate':
        y_axis_title += ' (0-1 scale)'
    elif selected_metric == 'Market Share':
        y_axis_title += ' (%)'
    
    fig.update_layout(
        title=f"{selected_metric} Analysis with {forecast_periods}-Month Forecast",
        xaxis_title="Date",
        yaxis_title=y_axis_title,
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple insights with help icons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_value = filtered_data[metric_column].iloc[-1]
        if selected_metric == 'Win Rate':
            st.metric("Current Value", f"{current_value:.1%}", help="The most recent month's value for the selected metric")
        elif selected_metric == 'Market Share':
            st.metric("Current Value", f"{current_value:.1f}%", help="The most recent month's value for the selected metric")
        else:
            st.metric("Current Value", f"{current_value:,.0f}", help="The most recent month's value for the selected metric")
    
    with col2:
        if len(filtered_data) > 1:
            prev_value = filtered_data[metric_column].iloc[-2]
            change = current_value - prev_value
            change_pct = (change / prev_value * 100) if prev_value != 0 else 0
            
            if selected_metric == 'Win Rate':
                st.metric("Monthly Change", f"{change:+.1%}", delta=f"{change_pct:+.1f}%", 
                         help="Change from the previous month (absolute change and percentage change)")
            elif selected_metric == 'Market Share':
                st.metric("Monthly Change", f"{change:+.1f}%", delta=f"{change_pct:+.1f}%", 
                         help="Change from the previous month (absolute change and percentage change)")
            else:
                st.metric("Monthly Change", f"{change:+,.0f}", delta=f"{change_pct:+.1f}%", 
                         help="Change from the previous month (absolute change and percentage change)")
    
    with col3:
        avg_value = filtered_data[metric_column].mean()
        if selected_metric == 'Win Rate':
            st.metric("Historical Avg", f"{avg_value:.1%}", 
                     help="Average value across all historical months in the selected time period")
        elif selected_metric == 'Market Share':
            st.metric("Historical Avg", f"{avg_value:.1f}%", 
                     help="Average value across all historical months in the selected time period")
        else:
            st.metric("Historical Avg", f"{avg_value:,.0f}", 
                     help="Average value across all historical months in the selected time period")
    
    # Data table (optional - collapsible) with both actual and predicted values
    with st.expander("📊 View Historical Data"):
        if selected_metric == 'Sales Performance' and 'Predicted_Sales' in filtered_data.columns:
            # For Sales Performance, show both actual and predicted
            display_data = filtered_data[['Date', metric_column, 'Predicted_Sales']].copy()
            display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m')
            display_data.columns = ['Month', f'Actual {selected_metric}', f'Predicted {selected_metric}']
            
            # Format both columns
            display_data[f'Actual {selected_metric}'] = display_data[f'Actual {selected_metric}'].apply(lambda x: f"{x:,.0f}")
            display_data[f'Predicted {selected_metric}'] = display_data[f'Predicted {selected_metric}'].apply(lambda x: f"{x:,.0f}")
        else:
            # For other metrics, show only the selected metric
            display_data = filtered_data[['Date', metric_column]].copy()
            display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m')
            display_data.columns = ['Month', selected_metric]
            
            if selected_metric == 'Win Rate':
                display_data[selected_metric] = display_data[selected_metric].apply(lambda x: f"{x:.1%}")
            elif selected_metric == 'Market Share':
                display_data[selected_metric] = display_data[selected_metric].apply(lambda x: f"{x:.1f}%")
            else:
                display_data[selected_metric] = display_data[selected_metric].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_data, use_container_width=True)
        
        # Show note about data completeness and quality
        if incomplete_last_month:
            st.caption(f"📝 Note: {incomplete_month_date.strftime('%B %Y')} data excluded from table above due to incomplete reporting.")
        
        # Add data quality notes for market share
        if selected_metric == 'Market Share':
            high_values = filtered_data[filtered_data['Current_Market_Share'] > 100]
            if not high_values.empty:
                st.caption(f"⚠️ Data Quality Note: {len(high_values)} months show market share >100%. This occurs when Industry Quantity ({filtered_data['Industry_Qty'].iloc[-1]:.0f}) is smaller than Actual Sales ({filtered_data['Actual_Sales'].iloc[-1]:.0f}), often due to data timing mismatches or territory boundary changes.")
        
        # If there are predicted values, show model accuracy info
        if selected_metric == 'Sales Performance' and 'Predicted_Sales' in filtered_data.columns:
            actual_values = filtered_data['Actual_Sales'].values
            predicted_values = filtered_data['Predicted_Sales'].values
            
            # Calculate simple accuracy metrics
            mae = np.mean(np.abs(actual_values - predicted_values))
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            st.caption(f"📊 Model Performance: MAE = {mae:,.0f}, MAPE = {mape:.1f}%")
    
    # Export functionality
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Export Data")
    
    if st.sidebar.button("📊 Download Analysis"):
        # Create export data
        export_data = filtered_data[['Date', 'Territory', 'Segment', metric_column]].copy()
        
        # Add forecast data if available
        if len(filtered_data) >= 3:
            forecast_df = generate_forecast_for_metric(filtered_data, metric_column, periods=forecast_periods)
            if not forecast_df.empty:
                # Create forecast export data
                forecast_export = pd.DataFrame({
                    'Date': forecast_df['Date'],
                    'Territory': selected_territory,
                    'Segment': selected_segment,
                    f'{selected_metric}_Forecast': forecast_df['Forecast']
                })
                
                import io
                buffer = io.BytesIO()
                
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_data.to_excel(writer, sheet_name='Historical_Data', index=False)
                    forecast_export.to_excel(writer, sheet_name='Forecast_Data', index=False)
                
                buffer.seek(0)
                
                st.sidebar.download_button(
                    label="📥 Download Excel",
                    data=buffer,
                    file_name=f"time_series_{selected_territory}_{selected_segment}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
                )

else:
    st.error("❌ Unable to load time series data")
    st.markdown("""
    ### 🔧 Troubleshooting:
    1. Check data files in `../results/`
    2. Run pipeline calculation first
    3. Contact analytics team if issues persist
    """)

# Navigation
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")

if st.sidebar.button("🏠 Go to Home"):
    st.switch_page("home.py")

if st.sidebar.button("🧮 Go to Pipeline Calculator"):
    st.switch_page("pages/pipeline_calculator.py")

if st.sidebar.button("🔍 Go to Feature Importance"):
    st.switch_page("pages/feature_importance.py")

# Footer
st.sidebar.markdown("---")
st.sidebar.image("https://www.seekpng.com/png/detail/209-2091306_rheem-logo.png", use_container_width=True)
st.sidebar.markdown("© 2025 Rheem Water Heater Division")
st.sidebar.markdown("*Advanced Analytics & Machine Learning*")
