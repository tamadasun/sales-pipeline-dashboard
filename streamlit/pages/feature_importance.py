import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from datetime import datetime
from sklearn.base import BaseEstimator, RegressorMixin
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import sklearn


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
    page_title="Feature Importance Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 Feature Importance Analysis")
st.markdown("Understand the key drivers behind our machine learning predictions for sales, win rates, and residual business.")

# Define SegmentSpecificModel class to prevent loading errors
class SegmentSpecificModel(BaseEstimator, RegressorMixin):
    def __init__(self, commercial_model, residential_model):
        self.commercial_model = commercial_model
        self.residential_model = residential_model
    
    def predict(self, X):
        return np.zeros(len(X))  # Simplified for compatibility
    
    def get_params(self, deep=True):
        return {"commercial_model": self.commercial_model, "residential_model": self.residential_model}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def load_data_dictionaries():
    """Load data dictionaries for feature insights"""
    if 'dictionaries' not in st.session_state:
        dictionaries = {}

        dict_files = {
            'feature_engineered': get_data_path("data/feature_engineered_data_dictionary.xlsx"),
            'main': get_data_path("data/data_dictionary.xlsx")
        }

        
        #dict_files = {
            #'feature_engineered': '../data/feature_engineered_data_dictionary.xlsx',
            #'main': '../data/data_dictionary.xlsx'
        #}

        
        
        for dict_name, file_path in dict_files.items():
            try:
                df = pd.read_excel(file_path)
                dictionaries[dict_name] = df
                st.sidebar.success(f"✅ {dict_name} dictionary loaded")
            except:
                try:
                    # Try markdown version
                    md_path = get_data_path(file_path.replace('.xlsx', '.md'))
                    #md_path = file_path.replace('.xlsx', '.md')
                    with open(md_path, 'r', encoding='utf-8') as f:
                        dictionaries[dict_name] = f.read()
                    st.sidebar.success(f"✅ {dict_name} dictionary loaded (MD)")
                except:
                    st.sidebar.warning(f"⚠️ {dict_name} dictionary not found")
        
        st.session_state.dictionaries = dictionaries
    
    return st.session_state.dictionaries

def get_comprehensive_feature_insight(feature_name, model_name, importance_value, data_dictionaries):
    """Generate comprehensive business insights using data dictionaries"""
    
    # First, try to find exact match in data dictionaries
    dict_definition = ""
    
    for dict_name, dict_data in data_dictionaries.items():
        if isinstance(dict_data, pd.DataFrame):
            if 'Column' in dict_data.columns and 'Description' in dict_data.columns:
                # Exact match
                exact_match = dict_data[dict_data['Column'] == feature_name]
                if not exact_match.empty:
                    dict_definition = exact_match['Description'].iloc[0]
                    break
                
                # Partial match
                partial_matches = dict_data[dict_data['Column'].str.contains(feature_name.replace('_', ''), case=False, na=False)]
                if not partial_matches.empty:
                    dict_definition = partial_matches['Description'].iloc[0]
                    break
    
    # Enhanced feature insights with business context
    enhanced_insights = {
        # Territory & Geographic
        'Territory_Upper': "Geographic territory codes (A03, A04, etc.) represent distinct sales regions with unique market characteristics, competition levels, customer demographics, and regulatory environments.",
        'Primary_State': "The state with highest sales volume in each territory. State-level factors include building codes, climate patterns, economic conditions, and competitive landscape that significantly impact water heater demand.",
        'Primary_City': "The city with highest sales volume in each territory. Urban vs suburban market dynamics, population density, local economic conditions, and infrastructure development drive performance variations.",
        'Primary_County': "The county with highest sales volume in each territory. County-level economic indicators, housing market conditions, and local regulations influence demand patterns and sales opportunities.",
        
        # Product & Pricing
        'Top_Product': "The highest-selling product by quantity in each territory-segment combination. Product preferences vary by region based on climate, building types, and customer preferences.",
        'Product_Concentration': "Sales dependence on top products (0-1 scale). Higher values indicate market specialization, while lower values suggest diversified product strategy. Regional preferences drive concentration patterns.",
        'Product_Count': "Number of unique products sold in each territory-segment-month. Higher diversity indicates market maturity and ability to serve varied customer needs.",
        'Unit_Selling_Price': "List price per unit before discounts. Pricing strategies vary by territory based on local competition, economic conditions, and customer segments served.",
        'Price_Tier': "Product classification (Budget/Standard/Premium/Luxury) based on pricing. Different territories may focus on different tiers based on market positioning and customer base.",
        'Discount_Rate': "Average discount percentage applied to orders. Heavy discounting may indicate competitive pressure, promotional strategies, or price-sensitive markets.",
        
        # Customer & Market
        'Market_Share': "Company's percentage of total industry sales in each territory-segment. Higher market share indicates competitive strength and customer preference.",
        'Industry_Qty': "Total market size (all competitors) in units. Larger markets provide more growth opportunity but may have more intense competition.",
        'Customer_Count': "Number of active customers in territory-segment. Broader customer base reduces concentration risk and indicates market penetration depth.",
        'Top_Customer': "Highest-volume customer in each territory-segment. Key account relationships are critical for revenue stability and market intelligence.",
        
        # Performance Metrics
        'Win_Rate': "Percentage of quotes that convert to sales. Higher win rates indicate better product-market fit, competitive positioning, and sales effectiveness.",
        'Revenue_per_Unit': "Average revenue generated per unit sold. Higher values indicate premium positioning or effective value-based selling strategies.",
        'Number_of_Bookings': "Count of confirmed orders. Strong booking performance indicates robust demand and effective sales processes.",
        
        # Time & Seasonality
        'Month': "Monthly seasonal patterns significantly impact water heater demand. Winter months typically see higher demand due to equipment failures and replacement needs.",
        'Quarter': "Quarterly business cycles affect planning, budgeting, and performance evaluation. Q4 often shows year-end buying patterns.",
        'Year': "Annual trends reveal long-term market evolution, economic cycles, and business growth patterns that inform strategic planning.",
        
        # Economic Indicators
        'housing_starts': "New construction activity is a leading indicator for water heater demand, particularly for new home installations and commercial projects.",
        'permits': "Building permits provide early signals of construction activity and future water heater demand 3-6 months ahead.",
        'CPI': "Consumer Price Index reflects inflation trends that affect customer purchasing power and pricing strategies.",
        'heating': "Heating degree days measure cold weather intensity. Colder periods increase water heater demand due to higher usage and equipment stress.",
        'cooling': "Cooling degree days indicate hot weather patterns that may affect energy efficiency preferences and replacement timing.",
        
        # Residual Business
        'Total_Residual': "Predictable recurring business from loyal customers provides revenue stability and cash flow predictability. Critical for financial planning.",
        'Monthly_Residual': "Average monthly recurring orders from repeat customers. Indicates customer loyalty and business relationship strength.",
        'Residual_Concentration': "Distribution of recurring business across customer base. High concentration indicates dependency on key accounts.",
        'Customer_Consistency': "How regularly customers place orders (0-1 scale). Higher consistency enables better demand forecasting and inventory planning."
    }
    
    # Find best matching insight
    business_context = dict_definition if dict_definition else "This feature significantly impacts business performance across all territories and segments."
    
    # Try enhanced insights for additional context
    for key, value in enhanced_insights.items():
        if key.lower() in feature_name.lower():
            business_context = value
            break
    
    # Model-specific interpretation
    model_context = {
        'Sales': "This feature drives sales volume predictions across all territories and segments. It helps forecast total units sold and revenue potential.",
        'Win Rate': "This feature influences conversion rate predictions across all territories and segments. It affects how many quotes convert to actual sales.",
        'Residual': "This feature impacts recurring business predictions across all territories and segments. It affects the stability and predictability of repeat customer orders."
    }
    
    # Determine impact level and recommendations
    if importance_value > 0.1:
        impact_level = "🔥 **CRITICAL DRIVER**"
        priority = "**Immediate Action Required**: This is a primary driver of business performance."
        monitoring = "Monitor daily/weekly for changes."
    elif importance_value > 0.05:
        impact_level = "⚡ **HIGH IMPACT**"
        priority = "**Strategic Priority**: Include in key planning initiatives and resource allocation decisions."
        monitoring = "Monitor monthly for trends."
    else:
        impact_level = "📊 **MODERATE INFLUENCE**"
        priority = "**Supporting Factor**: Consider in comprehensive analysis and operational planning."
        monitoring = "Monitor quarterly for longer-term trends."
    
    # Territory/Segment scope explanation
    scope_explanation = f"""
**Scope**: This feature importance represents the aggregate impact across all territories, segments, and time periods in our analysis. 
The model identifies this as a key driver of {model_name.lower()} performance when considering:
- All geographic territories (A03, A04, C01, E11, G06, I01, P90, W01, etc.)
- Both Commercial and Residential segments
- Historical patterns from 2022-2024
- Seasonal and temporal variations

**Business Application**: Use this insight to understand what factors most strongly influence {model_name.lower()} across your entire business operation.
"""
    
    return f"""
**📖 Feature Definition**: {business_context}

**🎯 Model Context**: {model_context.get(model_name, '')}

**📊 Business Impact**: {impact_level} (Importance Score: {importance_value:.4f})

{scope_explanation}
"""

def load_individual_model(model_path, model_name):
    """Load individual model with error handling"""
    
    # Inject SegmentSpecificModel into multiple locations
    current_module = sys.modules[__name__]
    current_module.SegmentSpecificModel = SegmentSpecificModel
    
    # Also try to inject into __main__ if it exists
    if '__main__' in sys.modules:
        sys.modules['__main__'].SegmentSpecificModel = SegmentSpecificModel
    
    # Add to globals for pickle
    globals()['SegmentSpecificModel'] = SegmentSpecificModel
    
    try:
        st.write(f"🔄 Attempting to load {model_name} from {model_path}")
        
        # Strategy 1: Direct joblib load
        model_data = joblib.load(model_path)
        #check if its a dictionary or raw model
        if isinstance(model_data, dict):
            st.sidebar.success(f"✅ {model_name} loaded successfully (dict format)")
            return model_data
        else:
            #its a raw model - wrap it in expected structure
            st.sidebar.success(f"✅ {model_name} loaded successfully (raw model)")

            #Try to get feature names if available
            features = []
            if hasattr(model_data, 'feature_names_in'):
                features = list(model_data.feature_names_in_)
            elif hasattr(model_data, 'feature_names_'):
                features = list(model_data.feature_names_)

            #Determine model type
            model_type = type(model_data).__name__
            if 'XGB' in model_type:
                best_model_name = 'XGBoost'
            elif 'LGBM' in model_type or 'LightGBM' in model_type:
                best_model_name = 'LightGBM'
            elif hasattr(model_data, 'commercial_model'):
                #Its a Segment SpecificModel
                best_model_name = 'Segment-Specific Ensemble'
            else:
                best_model_name = model_type

            #wrap in expected structure
            return {
                'best_model': model_data,
                'features': features,
                'best_model_name': best_model_name,
                'results': {}
            }

    except Exception as e:
        st.error(f"❌ Strategy 1 failed to load {model_name}: {str(e)}")
        
    # except Exception as e1:
    #     st.write(f"❌ Strategy 1 failed: {str(e1)}")
        
        try:
            # Strategy 2: Manual pickle with custom unpickler
            import pickle
            
            # class CustomUnpickler(pickle.Unpickler):
            #     def find_class(self, module, name):
            #         if name == 'SegmentSpecificModel':
            #             return SegmentSpecificModel
            #         return super().find_class(module, name)
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                #model_data = CustomUnpickler(f).load()
            #wrap if needed
            if not isinstance(model_data, dict):
                model_data = {
                    'best_model': model_data,
                    'features': [],
                    'best_model_name': type(model_data).__name__
                }
                
            st.sidebar.success(f"✅ {model_name} loaded with pickle")
            return model_data
            
        except Exception as e2:
            st.error(f"❌ Strategy 2 failed: {str(e2)}")
            #st.write(f"❌ Strategy 2 failed: {str(e2)}")
            
            try:
                # Strategy 3: Load and extract only what we need
                import dill
                model_data = dill.load(open(model_path, 'rb'))
                st.sidebar.success(f"✅ {model_name} loaded with dill")
                return model_data
                
            except Exception as e3:
                st.error(f"❌ All strategies failed for {model_name}")
                st.error(f"Errors: {str(e)[:100]}... | {str(e2)[:100]}... | {str(e3)[:100]}...")
                return None

def load_model_data():
    """Load all three trained models without caching"""
    
    if 'models' not in st.session_state:
        models = {}
        
        # Clear any existing cache first
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()

        #debug: show current working directory and structure
        st.write('### Debug information')
        st.write(f"**Current working directory**: `{os.getcwd()}`")
        #list files in current directory
        st.write("**Files in current directory**:")
        for item in os.listdir('.'):
            st.write(f"- {item}")
        #Check if trained_pickled_model exist
        if os.path.exists('trained_pickled_model'):
            st.write("\n**Files in trained_pickled_model/**:")
            for item in os.listdir('trained_pickled_model'):
                size = os.path.getsize(f'trained_pickled_model/{item}')
                st.write(f"- {item} ({size:,} bytes)")
                if size < 1000:
                    st.warning(f" {item} is only {size} bytes - might be a Git LFS pointer!")
        else:
            st.error("Trained_pickled_model directory not found")

            #Try parent directory
            parent_path = os.path.join('..', 'trained_pickled_model')
            if os.path.exists(parent_path):
                st.write(f"\n**Found in parent directory: {parent_path}**")
                for item in os.listdir(parent_path):
                    st.write(f"- {item}")

        model_paths = {
            'Sales': [
                get_data_path("trained_pickled_model/optimized_sales_model.pkl"),
                get_data_path("trained_pickled_model/sales_model.pkl")
            ],
            'Win Rate': [
                get_data_path("trained_pickled_model/win_rate_model.pkl")
            ],
            'Residual': [
                get_data_path("trained_pickled_model/residual_model.pkl"),
                get_data_path("trained_pickled_model/optimized_residual_model.pkl")
            ]
        }

        #Debug: shows what path are being checked
        st.write("\n### Path Resolution")
        for model_name, paths in model_paths.items():
            st.write(f"\n**{model_name} Model Paths**:")
            for path in paths:
                exists = os.path.exists(path)
                if exists:
                    size = os.path.getsize(path)
                    st.write(f"{path} - EXISTS ({size:,} bytes)")
                else:
                    st.write(f" {path} -NOT FOUND")

        
        # model_paths = {
        #     'Sales': [
        #         '../trained_pickled_model/optimized_sales_model.pkl',
        #         '../trained_pickled_model/sales_model.pkl'
        #     ],
        #     'Win Rate': [
        #         '../trained_pickled_model/win_rate_model.pkl'
        #     ],
        #     'Residual': [
        #         '../trained_pickled_model/residual_model.pkl',
        #         '../trained_pickled_model/optimized_residual_model.pkl'
        #     ]
        # }
        
        for model_name, paths in model_paths.items():
            st.write(f"\n### Loading {model_name} Model")
            loaded = False
            
            for path in paths:
                if os.path.exists(path):
                    st.write(f"📁 Found file: {path}")
                    model_data = load_individual_model(path, model_name)
                    
                    if model_data is not None:
                        models[model_name] = model_data
                        loaded = True
                        break
                else:
                    st.write(f"📁 File not found: {path}")
            
            if not loaded:
                st.sidebar.error(f"❌ {model_name} model failed to load from all paths")
        
        st.session_state.models = models
    
    return st.session_state.models

def extract_feature_importance(model_data, model_name):
    """Extract feature importance from models"""
    try:
        st.write(f"\n🔍 Extracting features from {model_name} model...")

        if isinstance(model_data, dict):
            model = model_data.get('best_model')
            features = model_data.get('features', [])
            # Get proper model name from the model data
            best_model_name = model_data.get('best_model_name', 'Unknown')
        else:
            #model_data IS the model itself
            model = model_data
            features = []
            best_model_name = type(model).__name__

        if len(features) == 0 and model is not None:
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
            elif hasattr(model, 'feature_names_'):
                features = list(model.feature_names_)
            elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                features = model.get_booster().feature_names

        st.write(f"Model type: {type(model)}")
        st.write(f"Model name: {best_model_name}")
        st.write(f"Features available: {len(features)}")
        
        # If still unknown, try to extract from results/metrics
        if best_model_name == 'Unknown':
            if 'results' in model_data:
                results = model_data['results']
                # Find the best performing model from results
                best_r2 = -1
                for name, result in results.items():
                    if 'r2' in result and result['r2'] > best_r2:
                        best_r2 = result['r2']
                        best_model_name = name
            
            # Try to infer from model type
            if best_model_name == 'Unknown':
                if isinstance(model, SegmentSpecificModel):
                    base_model = model.commercial_model
                    if 'XGB' in str(type(base_model)):
                        best_model_name = 'XGBoost Ensemble'
                    elif 'LGBM' in str(type(base_model)):
                        best_model_name = 'LightGBM Ensemble'
                    else:
                        best_model_name = f'{type(base_model).__name__} Ensemble'
                else:
                    best_model_name = type(model).__name__
        
        st.write(f"Model type: {type(model)}")
        st.write(f"Model name: {best_model_name}")
        st.write(f"Features available: {len(features)}")
        
        # Handle different model types - check SegmentSpecificModel FIRST
        if isinstance(model, SegmentSpecificModel) or str(type(model)).endswith("SegmentSpecificModel'>"):
            # Handle segment-specific models
            st.write("🔍 Processing SegmentSpecificModel...")
            base_model = model.commercial_model
            st.write(f"Commercial model type: {type(base_model)}")
            
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
                importance_type = 'Feature Importance (Commercial)'
                if hasattr(base_model, 'feature_names_in_'):
                    features = list(base_model.feature_names_in_)
                st.write(f"✅ Extracted from commercial model: {len(importances)}")
            elif hasattr(base_model, 'coef_'):
                importances = np.abs(base_model.coef_)
                importance_type = 'Coefficient Magnitude (Commercial)'
                if hasattr(base_model, 'feature_names_in_'):
                    features = list(base_model.feature_names_in_)
                st.write(f"✅ Extracted coefficients from commercial model: {len(importances)}")
            else:
                st.error(f"❌ Commercial model has no extractable importance")
                return None, None, None
                
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_type = 'Feature Importance'
            st.write(f"✅ Extracted tree-based importances: {len(importances)}")
            
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)
            importance_type = 'Coefficient Magnitude'
            st.write(f"✅ Extracted linear model coefficients: {len(importances)}")
            
        elif hasattr(model, 'estimators_'):
            # Ensemble models - try to extract from components
            st.write("🔍 Examining ensemble components...")
            lgbm_model = None
            
            for i, est in enumerate(model.estimators_):
                st.write(f"Component {i}: {type(est)}")
                if hasattr(est, 'feature_importances_'):
                    lgbm_model = est
                    st.write(f"✅ Found component with feature_importances_")
                    break
            
            if lgbm_model:
                if hasattr(lgbm_model, 'booster_') and hasattr(lgbm_model.booster_, 'feature_name'):
                    features = lgbm_model.booster_.feature_name()
                    st.write(f"✅ Extracted feature names from booster: {len(features)}")
                importances = lgbm_model.feature_importances_
                importance_type = 'Feature Importance (Ensemble)'
                st.write(f"✅ Extracted ensemble importances: {len(importances)}")
            else:
                st.error(f"❌ No suitable component found in ensemble")
                return None, None, None
        else:
            st.error(f"❌ Unknown model type: {type(model)}")
            return None, None, None
        
        # Ensure matching lengths
        if len(features) != len(importances):
            st.warning(f"⚠️ Length mismatch: {len(features)} features vs {len(importances)} importances")
            
            if hasattr(model, 'feature_names_in_'):
                features = list(model.feature_names_in_)
                st.write(f"✅ Used feature_names_in_: {len(features)}")
            elif len(features) == 0:
                features = [f'feature_{i}' for i in range(len(importances))]
                st.write(f"✅ Created generic feature names: {len(features)}")
            else:
                min_len = min(len(features), len(importances))
                features = features[:min_len]
                importances = importances[:min_len]
                st.write(f"✅ Truncated to match: {min_len}")
        
        # Create dataframe
        feature_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Model': model_name,
            'Type': importance_type
        }).sort_values('Importance', ascending=False)
        
        st.write(f"✅ Created feature dataframe with {len(feature_df)} features")
        
        return feature_df, importance_type, best_model_name
        
    except Exception as e:
        st.error(f"❌ Error extracting importance for {model_name}: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None

def categorize_features(feature_names):
    """Categorize features into business groups"""
    categories = {
        'Territory & Geographic': [],
        'Product & Pricing': [],
        'Customer & Market': [],
        'Time & Seasonality': [],
        'Performance Metrics': [],
        'Economic Indicators': [],
        'Other': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['territory', 'state', 'city', 'county', 'region']):
            categories['Territory & Geographic'].append(feature)
        elif any(x in feature_lower for x in ['product', 'price', 'tier', 'discount', 'value_segment']):
            categories['Product & Pricing'].append(feature)
        elif any(x in feature_lower for x in ['customer', 'market', 'industry', 'share', 'concentration']):
            categories['Customer & Market'].append(feature)
        elif any(x in feature_lower for x in ['year', 'month', 'quarter', 'seasonal', 'time']):
            categories['Time & Seasonality'].append(feature)
        elif any(x in feature_lower for x in ['sales', 'revenue', 'win', 'rate', 'residual', 'performance']):
            categories['Performance Metrics'].append(feature)
        elif any(x in feature_lower for x in ['housing', 'permits', 'starts', 'heating', 'cooling', 'economic']):
            categories['Economic Indicators'].append(feature)
        else:
            categories['Other'].append(feature)
    
    return categories

def create_importance_plot(feature_df, model_name, top_n=20):
    """Create feature importance plot"""
    top_features = feature_df.head(top_n)
    
    fig = go.Figure(data=go.Bar(
        y=top_features['Feature'],
        x=top_features['Importance'],
        orientation='h',
        marker=dict(
            color=top_features['Importance'],
            colorscale='Viridis',
            showscale=True
        ),
        text=top_features['Importance'].round(4),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Features - {model_name} Model',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=max(400, top_n * 20),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_category_analysis(feature_df):
    """Create category importance analysis"""
    categories = categorize_features(feature_df['Feature'].tolist())
    
    category_data = []
    for category, features in categories.items():
        if features:
            category_features = feature_df[feature_df['Feature'].isin(features)]
            category_data.append({
                'Category': category,
                'Total_Importance': category_features['Importance'].sum(),
                'Feature_Count': len(features),
                'Top_Feature': category_features.iloc[0]['Feature']
            })
    
    return pd.DataFrame(category_data).sort_values('Total_Importance', ascending=False)

# Clear cache button
if st.sidebar.button("🔄 Clear Cache & Reload"):
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Load data silently
if 'data_loaded' not in st.session_state:
    with st.spinner("Loading models and data..."):
        data_dictionaries = load_data_dictionaries()
        models = load_model_data()
        
        if not models:
            st.error("❌ No models could be loaded. Please check model files exist.")
            st.stop()
        
        # Extract feature importance
        all_features = []
        model_info = {}
        
        for model_name in ['Sales', 'Win Rate', 'Residual']:
            if model_name in models:
                feature_df, importance_type, best_model_name = extract_feature_importance(models[model_name], model_name)
                if feature_df is not None:
                    all_features.append(feature_df)
                    model_info[model_name] = {
                        'importance_type': importance_type,
                        'best_model_name': best_model_name,
                        'feature_count': len(feature_df)
                    }
        
        if not all_features:
            st.error("❌ Could not extract feature importance from any models.")
            st.stop()
        
        st.session_state.all_features_df = pd.concat(all_features, ignore_index=True)
        st.session_state.model_info = model_info
        st.session_state.data_dictionaries = data_dictionaries
        st.session_state.data_loaded = True

# Get data from session state
all_features_df = st.session_state.all_features_df
model_info = st.session_state.model_info
data_dictionaries = st.session_state.data_dictionaries

#all_features_df = pd.concat(all_features, ignore_index=True)

# Sidebar controls
st.sidebar.markdown("### 🎛️ Controls")

available_models = list(model_info.keys())
selected_model = st.sidebar.selectbox(
    "Select Model",
    options=available_models,
    help="Choose model to analyze"
)

top_n_features = st.sidebar.slider(
    "Number of Features",
    min_value=10,
    max_value=50,
    value=20,
    step=5
)

# Model info
st.sidebar.markdown("### 📊 Model Info")
for model_name, info in model_info.items():
    with st.sidebar.expander(f"{model_name}"):
        st.write(f"**Type**: {info['best_model_name']}")
        st.write(f"**Features**: {info['feature_count']}")

# Main analysis
model_features = all_features_df[all_features_df['Model'] == selected_model]

st.header(f"🔍 {selected_model} Model Analysis")

# Key metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Type", model_info[selected_model]['best_model_name'])
with col2:
    st.metric("Total Features", f"{model_info[selected_model]['feature_count']:,}")
with col3:
    st.metric("Importance Type", model_info[selected_model]['importance_type'])

# Feature importance plot
st.subheader(f"📊 Top {top_n_features} Features")
fig = create_importance_plot(model_features, selected_model, top_n_features)
st.plotly_chart(fig, use_container_width=True)

# Feature insights - scale with slider
st.subheader("💡 Comprehensive Feature Insights")
st.markdown(f"*Detailed analysis of top {min(top_n_features, 15)} features with integrated data dictionary definitions*")

top_features_for_insights = model_features.head(min(top_n_features, 15))  # Cap at 15 for readability

for i, (_, row) in enumerate(top_features_for_insights.iterrows(), 1):
    with st.expander(f"#{i}: {row['Feature']} (Importance: {row['Importance']:.4f})", expanded=(i <= 3)):
        insight = get_comprehensive_feature_insight(row['Feature'], selected_model, row['Importance'], data_dictionaries)
        st.markdown(insight)

# Category analysis
st.subheader("📂 Category Analysis")
category_df = create_category_analysis(model_features)

if not category_df.empty:
    fig_cat = px.bar(
        category_df,
        x='Category',
        y='Total_Importance',
        title=f'Feature Categories - {selected_model} Model',
        hover_data=['Feature_Count', 'Top_Feature']
    )
    fig_cat.update_layout(xaxis={'tickangle': 45})
    st.plotly_chart(fig_cat, use_container_width=True)
    
    st.dataframe(category_df, use_container_width=True)

# Feature table
st.subheader(f"📈 Top {top_n_features} Features - Details")
display_features = model_features.head(top_n_features)[['Feature', 'Importance']].copy()
display_features['Rank'] = range(1, len(display_features) + 1)
display_features['Importance'] = display_features['Importance'].round(6)
st.dataframe(display_features[['Rank', 'Feature', 'Importance']], use_container_width=True)

# Export
st.sidebar.markdown("---")
if st.sidebar.button("📥 Export Analysis"):
    import io
    buffer = io.BytesIO()
    
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        all_features_df.to_excel(writer, sheet_name='All_Features', index=False)
        for model_name in model_info.keys():
            model_data = all_features_df[all_features_df['Model'] == model_name]
            model_data.to_excel(writer, sheet_name=f'{model_name}_Features', index=False)
    
    buffer.seek(0)
    st.sidebar.download_button(
        label="📥 Download Excel",
        data=buffer,
        file_name=f"feature_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.ms-excel"
    )

# Navigation
st.sidebar.markdown("---")
if st.sidebar.button("🏠 Home"):
    st.switch_page("home.py")
if st.sidebar.button("🧮 Pipeline Calculator"):
    st.switch_page("pages/pipeline_calculator.py")
if st.sidebar.button("📈 Time Series"):
    st.switch_page("pages/time_series_analysis.py")

# Footer
st.sidebar.markdown("---")
st.sidebar.image("https://www.seekpng.com/png/detail/209-2091306_rheem-logo.png", use_container_width=True)
st.sidebar.markdown("© 2025 Rheem Water Heater Division")
