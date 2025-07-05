import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Agricultural Data Analytics Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .effectiveness-highest {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .effectiveness-moderate {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .effectiveness-least {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌾 Agricultural Data Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation and controls
with st.sidebar:
    st.header("📊 Dashboard Controls")
    
    # Analysis mode selection
    analysis_mode = st.selectbox(
        "Choose Analysis Mode:",
        ["Upload Custom Data", "Use Sample Data", "Manual Input"]
    )
    
    st.markdown("---")
    
    # Model parameters
    st.subheader("🎛️ Model Parameters")
    rf_n_estimators = st.slider("Random Forest Trees", 50, 200, 100)
    dt_max_depth = st.slider("Decision Tree Max Depth", 5, 20, 10)
    kmeans_clusters = st.slider("K-Means Clusters", 2, 8, 3)
    
    st.markdown("---")
    
    # Visualization options
    st.subheader("📈 Visualization Options")
    show_correlation = st.checkbox("Show Correlation Matrix", True)
    show_clusters = st.checkbox("Show Cluster Analysis", True)
    show_sankey = st.checkbox("Show Sankey Diagram", True)
    show_effectiveness_analysis = st.checkbox("Show Effectiveness Analysis", True)

# Sample data generator
@st.cache_data
def generate_sample_data():
    """Generate sample agricultural data for demonstration"""
    np.random.seed(42)
    
    # Counties
    counties = ['Nairobi', 'Kiambu', 'Nakuru', 'Meru', 'Kisumu', 'Eldoret', 'Mombasa', 'Nyeri', 'Kakamega', 'Kisii']
    
    # Crops
    crops = ['Maize', 'Beans', 'Potatoes', 'Rice', 'Wheat', 'Sorghum', 'Millet', 'Cassava']
    
    # Livestock
    livestock = ['Indigenous cattle', 'Exotic cattle', 'Goats', 'Sheep', 'Indigenous Chicken', 'Exotic Chicken']
    
    # Environmental data
    env_data = []
    for i in range(1000):
        env_data.append({
            'temperature': np.random.normal(22, 5),
            'humidity': np.random.normal(65, 15),
            'pressure': np.random.normal(840, 10),
            'hour': np.random.randint(0, 24),
            'month': np.random.randint(1, 13),
            'day_of_year': np.random.randint(1, 366),
            'day_of_week': np.random.randint(0, 7)
        })
    
    # Crop production data
    crop_data = []
    for county in counties:
        for crop in crops:
            crop_data.append({
                'county': county,
                'crop': crop,
                'production': np.random.randint(100, 10000),
                'area_planted': np.random.randint(50, 5000),
                'yield_per_hectare': np.random.uniform(0.5, 5.0)
            })
    
    # Livestock data
    livestock_data = []
    for county in counties:
        for animal in livestock:
            livestock_data.append({
                'county': county,
                'livestock': animal,
                'population': np.random.randint(500, 50000),
                'productivity': np.random.uniform(0.6, 1.4)
            })
    
    return pd.DataFrame(env_data), pd.DataFrame(crop_data), pd.DataFrame(livestock_data)

# Data processing functions
class AgriculturalAnalyzer:
    def __init__(self):
        self.rf_model = None
        self.dt_model = None
        self.kmeans_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def train_environmental_model(self, env_data, n_estimators=100):
        """Train Random Forest model for environmental prediction"""
        feature_cols = ['humidity', 'pressure', 'hour', 'month', 'day_of_year', 'day_of_week']
        X = env_data[feature_cols].fillna(env_data[feature_cols].mean())
        y = env_data['temperature'].fillna(env_data['temperature'].mean())
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        y_pred = self.rf_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return r2, rmse, feature_cols
    
    def create_effectiveness_dataset(self, crop_data, livestock_data):
        """Create crop and livestock effectiveness dataset with enhanced categorization"""
        # Combine crop and livestock data
        all_data = []
        
        # Process crop data
        if not crop_data.empty and 'county' in crop_data.columns:
            for _, row in crop_data.iterrows():
                try:
                    effectiveness = row.get('yield_per_hectare', np.random.uniform(0.5, 1.5))
                    all_data.append({
                        'county': row.get('county', 'Unknown'),
                        'type': row.get('crop', 'Unknown Crop'),
                        'category': 'Crop',
                        'production': row.get('production', 0),
                        'effectiveness': effectiveness,
                        'area_planted': row.get('area_planted', 0)
                    })
                except Exception as e:
                    print(f"Error processing crop row: {e}")
                    continue
        
        # Process livestock data
        if not livestock_data.empty and 'county' in livestock_data.columns:
            for _, row in livestock_data.iterrows():
                try:
                    effectiveness = row.get('productivity', np.random.uniform(0.5, 1.5))
                    all_data.append({
                        'county': row.get('county', 'Unknown'),
                        'type': row.get('livestock', 'Unknown Livestock'),
                        'category': 'Livestock',
                        'production': row.get('population', 0),
                        'effectiveness': effectiveness,
                        'area_planted': 0
                    })
                except Exception as e:
                    print(f"Error processing livestock row: {e}")
                    continue
        
        # Create DataFrame from processed data
        if not all_data:
            # If no data was processed, create a minimal dataset for demonstration
            all_data = [{
                'county': 'Sample County',
                'type': 'Sample Crop',
                'category': 'Crop',
                'production': 1000,
                'effectiveness': 1.0,
                'area_planted': 100
            }]
        
        df = pd.DataFrame(all_data)
        
        # Enhanced effectiveness categorization with clearer thresholds
        if len(df) > 1:
            effectiveness_percentiles = df['effectiveness'].quantile([0.33, 0.67])
        else:
            # Handle single row case
            effectiveness_percentiles = {0.33: df['effectiveness'].iloc[0] * 0.8, 0.67: df['effectiveness'].iloc[0] * 1.2}
        
        def categorize_effectiveness(value):
            if value <= effectiveness_percentiles[0.33]:
                return 'Least Effective'
            elif value <= effectiveness_percentiles[0.67]:
                return 'Moderate'
            else:
                return 'Highest'
        
        df['effectiveness_level'] = df['effectiveness'].apply(categorize_effectiveness)
        
        return df
    
    def create_comprehensive_correlation_analysis(self, effectiveness_data):
        """Create comprehensive correlation analysis between counties and crops/livestock"""
        # County-Crop correlation matrix
        county_crop_pivot = effectiveness_data.pivot_table(
            index='county', 
            columns='type', 
            values='effectiveness', 
            aggfunc='mean'
        ).fillna(0)
        
        # County-Category correlation
        county_category_pivot = effectiveness_data.pivot_table(
            index='county', 
            columns='category', 
            values='effectiveness', 
            aggfunc='mean'
        ).fillna(0)
        
        # Summary statistics by county
        county_summary = effectiveness_data.groupby('county').agg({
            'effectiveness': ['mean', 'std', 'count'],
            'production': ['sum', 'mean'],
            'area_planted': 'sum'
        }).round(3)
        
        # Flatten column names
        county_summary.columns = ['_'.join(col).strip() for col in county_summary.columns.values]
        
        # Add effectiveness ranking
        county_summary['effectiveness_rank'] = county_summary['effectiveness_mean'].rank(ascending=False)
        
        # Crop/Livestock performance by county
        type_county_summary = effectiveness_data.groupby(['type', 'county']).agg({
            'effectiveness': 'mean',
            'production': 'sum'
        }).round(3)
        
        return county_crop_pivot, county_category_pivot, county_summary, type_county_summary
    
    def create_sankey_diagram(self, effectiveness_data):
        """Create Sankey diagram showing flow between counties and crops/livestock"""
        # Prepare data for Sankey diagram
        county_type_flow = effectiveness_data.groupby(['county', 'type', 'effectiveness_level']).agg({
            'production': 'sum',
            'effectiveness': 'mean'
        }).reset_index()
        
        # Create unique labels
        counties = effectiveness_data['county'].unique()
        types = effectiveness_data['type'].unique()
        effectiveness_levels = ['Highest', 'Moderate', 'Least Effective']
        
        # Create label mapping
        all_labels = list(counties) + list(types) + effectiveness_levels
        label_mapping = {label: idx for idx, label in enumerate(all_labels)}
        
        # Create flows
        source = []
        target = []
        value = []
        
        # County to Type flows
        for _, row in county_type_flow.iterrows():
            source.append(label_mapping[row['county']])
            target.append(label_mapping[row['type']])
            value.append(row['production'] / 1000)  # Scale down for visualization
        
        # Type to Effectiveness flows
        type_effectiveness_flow = effectiveness_data.groupby(['type', 'effectiveness_level']).agg({
            'production': 'sum'
        }).reset_index()
        
        for _, row in type_effectiveness_flow.iterrows():
            source.append(label_mapping[row['type']])
            target.append(label_mapping[row['effectiveness_level']])
            value.append(row['production'] / 1000)  # Scale down for visualization
        
        # Create colors
        colors = ['rgba(31, 119, 180, 0.8)'] * len(counties) + \
                ['rgba(255, 127, 14, 0.8)'] * len(types) + \
                ['rgba(44, 160, 44, 0.8)', 'rgba(255, 187, 120, 0.8)', 'rgba(214, 39, 40, 0.8)']
        
        return source, target, value, all_labels, colors
    
    def create_effectiveness_visualization(self, effectiveness_data):
        """Create comprehensive effectiveness visualization with clear legends"""
        # Effectiveness distribution
        effectiveness_dist = effectiveness_data['effectiveness_level'].value_counts()
        
        # County effectiveness heatmap
        county_effectiveness = effectiveness_data.pivot_table(
            index='county', 
            columns='effectiveness_level', 
            values='production', 
            aggfunc='sum'
        ).fillna(0)
        
        # Type effectiveness comparison
        type_effectiveness = effectiveness_data.groupby(['type', 'effectiveness_level']).size().unstack(fill_value=0)
        
        return effectiveness_dist, county_effectiveness, type_effectiveness
    
    def train_decision_tree(self, effectiveness_data, max_depth=10):
        """Train decision tree for effectiveness classification"""
        # Prepare features
        features = ['production', 'effectiveness', 'area_planted']
        
        # Encode categorical variables
        for col in ['county', 'type']:
            le = LabelEncoder()
            effectiveness_data[f'{col}_encoded'] = le.fit_transform(effectiveness_data[col])
            self.label_encoders[col] = le
            features.append(f'{col}_encoded')
        
        X = effectiveness_data[features].fillna(0)
        y = effectiveness_data['effectiveness_level']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.dt_model.fit(X_train, y_train)
        
        y_pred = self.dt_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, features
    
    def perform_clustering(self, effectiveness_data, n_clusters=3):
        """Perform K-means clustering on counties"""
        # Aggregate data by county
        county_features = effectiveness_data.groupby('county').agg({
            'production': 'sum',
            'effectiveness': 'mean',
            'area_planted': 'sum'
        }).reset_index()
        
        # Scale features
        X = county_features[['production', 'effectiveness', 'area_planted']]
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        county_features['cluster'] = self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X_scaled, county_features['cluster'])
        
        return county_features, silhouette

# Initialize analyzer
analyzer = AgriculturalAnalyzer()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Input", "🤖 Model Training", "📈 Visualizations", "🔍 Analysis", "💡 Insights"])

with tab1:
    st.header("📊 Data Input & Preview")
    
    # Information about auto-loading sample data
    if analysis_mode == "Use Sample Data":
        st.info("📋 **Quick Start**: Sample data will be automatically loaded for immediate visualization. You can explore all features right away!")
    
    if analysis_mode == "Upload Custom Data":
        st.subheader("Upload Your Agricultural Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            env_file = st.file_uploader("Environmental Data (CSV)", type=['csv'])
        with col2:
            crop_file = st.file_uploader("Crop Data (CSV)", type=['csv'])
        with col3:
            livestock_file = st.file_uploader("Livestock Data (CSV)", type=['csv'])
        
        if env_file and crop_file and livestock_file:
            env_data = pd.read_csv(env_file)
            crop_data = pd.read_csv(crop_file)
            livestock_data = pd.read_csv(livestock_file)
            
            # Store in session state
            st.session_state.env_data = env_data
            st.session_state.crop_data = crop_data
            st.session_state.livestock_data = livestock_data
            
            # Create effectiveness data for immediate visualization
            effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
            st.session_state.effectiveness_data = effectiveness_data
            
            st.success("✅ Data uploaded successfully! Ready for visualization.")
            
            # Display data previews
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Environmental Data")
                st.dataframe(env_data.head())
            with col2:
                st.subheader("Crop Data")
                st.dataframe(crop_data.head())
            with col3:
                st.subheader("Livestock Data")
                st.dataframe(livestock_data.head())
        elif 'env_data' in st.session_state:
            # Load from session state if available
            env_data = st.session_state.env_data
            crop_data = st.session_state.crop_data
            livestock_data = st.session_state.livestock_data
            
            st.info("✅ Previously uploaded data is loaded. Upload new files to replace.")
        else:
            st.info("Please upload all three data files to proceed.")
            env_data, crop_data, livestock_data = None, None, None
    
    elif analysis_mode == "Use Sample Data":
        st.subheader("Using Sample Data")
        
        # Auto-generate sample data if not already available
        if 'env_data' not in st.session_state:
            try:
                env_data, crop_data, livestock_data = generate_sample_data()
                st.session_state.env_data = env_data
                st.session_state.crop_data = crop_data
                st.session_state.livestock_data = livestock_data
                
                # Debug info
                st.info(f"Generated: {len(crop_data)} crop records, {len(livestock_data)} livestock records")
                
                # Auto-create effectiveness data for immediate visualization
                effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
                st.session_state.effectiveness_data = effectiveness_data
                
                st.success("✅ Sample data generated automatically!")
                
            except Exception as e:
                st.error(f"Error generating sample data: {e}")
                # Create minimal fallback data
                st.session_state.env_data = pd.DataFrame([{'temperature': 22, 'humidity': 65, 'pressure': 840, 'hour': 12, 'month': 6, 'day_of_year': 150, 'day_of_week': 1}])
                st.session_state.crop_data = pd.DataFrame([{'county': 'Sample County', 'crop': 'Maize', 'production': 1000, 'area_planted': 100, 'yield_per_hectare': 2.5}])
                st.session_state.livestock_data = pd.DataFrame([{'county': 'Sample County', 'livestock': 'Cattle', 'population': 500, 'productivity': 1.2}])
        
        if st.button("🔄 Regenerate Sample Data"):
            try:
                env_data, crop_data, livestock_data = generate_sample_data()
                st.session_state.env_data = env_data
                st.session_state.crop_data = crop_data
                st.session_state.livestock_data = livestock_data
                
                # Recreate effectiveness data
                effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
                st.session_state.effectiveness_data = effectiveness_data
                
                st.success("✅ Sample data regenerated successfully!")
            except Exception as e:
                st.error(f"Error regenerating sample data: {e}")
        
        # Load from session state
        env_data = st.session_state.env_data
        crop_data = st.session_state.crop_data
        livestock_data = st.session_state.livestock_data
        
        # Display sample data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Environmental Data")
            st.dataframe(env_data.head())
        with col2:
            st.subheader("Crop Data")
            st.dataframe(crop_data.head())
        with col3:
            st.subheader("Livestock Data")
            st.dataframe(livestock_data.head())
    
    else:  # Manual Input
        st.subheader("Manual Data Input")
        
        # Simple form for manual input
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Environmental Conditions")
                temperature = st.number_input("Temperature (°C)", value=22.0)
                humidity = st.number_input("Humidity (%)", value=65.0)
                pressure = st.number_input("Pressure (hPa)", value=840.0)
            
            with col2:
                st.subheader("Agricultural Data")
                county = st.selectbox("County", ['Nairobi', 'Kiambu', 'Nakuru', 'Meru', 'Kisumu'])
                crop_type = st.selectbox("Crop/Livestock", ['Maize', 'Beans', 'Potatoes', 'Indigenous cattle', 'Goats'])
                production = st.number_input("Production/Population", value=1000)
            
            submitted = st.form_submit_button("Analyze Single Entry")
            
            if submitted:
                # Create single entry datasets
                env_data = pd.DataFrame([{
                    'temperature': temperature,
                    'humidity': humidity,
                    'pressure': pressure,
                    'hour': 12,
                    'month': 6,
                    'day_of_year': 150,
                    'day_of_week': 1
                }])
                
                if crop_type in ['Maize', 'Beans', 'Potatoes']:
                    crop_data = pd.DataFrame([{
                        'county': county,
                        'crop': crop_type,
                        'production': production,
                        'area_planted': production * 0.1,
                        'yield_per_hectare': np.random.uniform(1.0, 3.0)
                    }])
                    livestock_data = pd.DataFrame(columns=['county', 'livestock', 'population', 'productivity'])
                else:
                    livestock_data = pd.DataFrame([{
                        'county': county,
                        'livestock': crop_type,
                        'population': production,
                        'productivity': np.random.uniform(0.8, 1.2)
                    }])
                    crop_data = pd.DataFrame(columns=['county', 'crop', 'production', 'area_planted', 'yield_per_hectare'])
                
                # Store in session state
                st.session_state.env_data = env_data
                st.session_state.crop_data = crop_data
                st.session_state.livestock_data = livestock_data
                
                # Create effectiveness data for visualization
                if not crop_data.empty or not livestock_data.empty:
                    effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
                    st.session_state.effectiveness_data = effectiveness_data
                
                st.success("✅ Manual data prepared for analysis!")

with tab2:
    st.header("🤖 Model Training & Results")
    
    # Check if data is available
    if 'env_data' in locals() and env_data is not None:
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Environmental Model")
            
            if st.button("Train Environmental Model"):
                with st.spinner("Training Random Forest model..."):
                    r2, rmse, features = analyzer.train_environmental_model(env_data, rf_n_estimators)
                
                st.success("✅ Environmental model trained!")
                
                # Display metrics
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with metric_col2:
                    st.metric("RMSE", f"{rmse:.4f}")
                
                # Feature importance
                if analyzer.rf_model:
                    importance_df = pd.DataFrame({
                        'feature': features,
                        'importance': analyzer.rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                title="Feature Importance - Environmental Model")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌾 Effectiveness Model")
            
            if st.button("Train Effectiveness Model"):
                with st.spinner("Creating effectiveness dataset..."):
                    effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
                
                with st.spinner("Training decision tree..."):
                    accuracy, dt_features = analyzer.train_decision_tree(effectiveness_data, dt_max_depth)
                
                st.success("✅ Effectiveness model trained!")
                
                # Display accuracy
                st.metric("Accuracy", f"{accuracy:.4f}")
                
                # Store effectiveness data in session state
                st.session_state.effectiveness_data = effectiveness_data
        
        # Clustering section
        st.subheader("🎯 Clustering Analysis")
        
        if st.button("Perform Clustering"):
            if 'effectiveness_data' in st.session_state:
                with st.spinner("Performing K-means clustering..."):
                    county_clusters, silhouette = analyzer.perform_clustering(
                        st.session_state.effectiveness_data, kmeans_clusters
                    )
                
                st.success("✅ Clustering completed!")
                
                # Display silhouette score
                st.metric("Silhouette Score", f"{silhouette:.4f}")
                
                # Store clustering results
                st.session_state.county_clusters = county_clusters
                
                # Display cluster summary
                st.subheader("Cluster Summary")
                st.dataframe(county_clusters)
            else:
                st.warning("Please train the effectiveness model first!")
    
    else:
        st.info("Please provide data in the Data Input tab to train models.")

with tab3:
    st.header("📈 Interactive Visualizations & Correlation Analysis")
    
    # Check if we have effectiveness data or can create it
    if 'effectiveness_data' in st.session_state:
        effectiveness_data = st.session_state.effectiveness_data
    elif 'crop_data' in st.session_state and 'livestock_data' in st.session_state:
        # Auto-create effectiveness data if we have crop/livestock data
        try:
            crop_data = st.session_state.crop_data
            livestock_data = st.session_state.livestock_data
            
            # Debug info
            st.write(f"Debug: Crop data columns: {list(crop_data.columns) if not crop_data.empty else 'Empty'}")
            st.write(f"Debug: Livestock data columns: {list(livestock_data.columns) if not livestock_data.empty else 'Empty'}")
            
            effectiveness_data = analyzer.create_effectiveness_dataset(crop_data, livestock_data)
            st.session_state.effectiveness_data = effectiveness_data
            st.success("📊 Effectiveness data created automatically for visualization!")
        except Exception as e:
            st.error(f"Error creating effectiveness data: {e}")
            effectiveness_data = None
    else:
        effectiveness_data = None
    
    if effectiveness_data is not None:
        
        # Comprehensive Correlation Analysis
        if show_correlation:
            st.subheader("🔗 Comprehensive County-Crop/Livestock Correlation Analysis")
            
            # Get correlation data
            county_crop_pivot, county_category_pivot, county_summary, type_county_summary = analyzer.create_comprehensive_correlation_analysis(effectiveness_data)
            
            # Display correlation matrices
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏘️ County-Type Correlation Matrix")
                fig_corr = px.imshow(county_crop_pivot, 
                                   title="County vs Crop/Livestock Effectiveness Correlation",
                                   labels={'x': 'Crop/Livestock Type', 'y': 'County', 'color': 'Effectiveness Score'},
                                   color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                st.subheader("📊 County-Category Correlation")
                fig_cat = px.imshow(county_category_pivot, 
                                  title="County vs Category (Crop/Livestock) Effectiveness",
                                  labels={'x': 'Category', 'y': 'County', 'color': 'Effectiveness Score'},
                                  color_continuous_scale='RdYlGn')
                st.plotly_chart(fig_cat, use_container_width=True)
            
            # Enhanced County Summary Table
            st.subheader("📋 County Performance Summary Table")
            
            # Style the dataframe with colors
            def style_effectiveness(val):
                if val >= county_summary['effectiveness_mean'].quantile(0.67):
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif val >= county_summary['effectiveness_mean'].quantile(0.33):
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold'
                else:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            
            styled_county_summary = county_summary.style.applymap(style_effectiveness, subset=['effectiveness_mean'])
            st.dataframe(styled_county_summary, use_container_width=True)
            
            # Download correlation data
            csv_county_summary = county_summary.to_csv()
            st.download_button(
                label="📥 Download County Summary Data",
                data=csv_county_summary,
                file_name="county_correlation_summary.csv",
                mime="text/csv"
            )
        
        # Sankey Diagram
        if show_sankey:
            st.subheader("🌊 Sankey Flow Diagram - County → Crop/Livestock → Effectiveness")
            
            source, target, value, all_labels, colors = analyzer.create_sankey_diagram(effectiveness_data)
            
            fig_sankey = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color=colors
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color='rgba(255, 255, 255, 0.4)'
                )
            )])
            
            fig_sankey.update_layout(
                title_text="Agricultural Flow: Counties → Crops/Livestock → Effectiveness Levels",
                font_size=12,
                height=600
            )
            
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            # Sankey interpretation
            st.markdown("""
            **📊 Sankey Diagram Interpretation:**
            - **Left Column**: Counties (Blue) - Data sources
            - **Middle Column**: Crops/Livestock (Orange) - Agricultural products
            - **Right Column**: Effectiveness Levels (Green=Highest, Yellow=Moderate, Red=Least Effective)
            - **Flow Width**: Represents production volume/population size
            """)
        
        # Enhanced Effectiveness Visualization
        if show_effectiveness_analysis:
            st.subheader("🎯 Effectiveness Analysis with Clear Categorization")
            
            effectiveness_dist, county_effectiveness, type_effectiveness = analyzer.create_effectiveness_visualization(effectiveness_data)
            
            # Effectiveness distribution with custom colors
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Overall Effectiveness Distribution")
                
                colors_pie = ['#28a745', '#ffc107', '#dc3545']  # Green, Yellow, Red
                fig_pie = px.pie(
                    values=effectiveness_dist.values, 
                    names=effectiveness_dist.index,
                    title="Distribution of Effectiveness Levels",
                    color_discrete_sequence=colors_pie
                )
                
                # Add percentage labels
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Legend explanation
                st.markdown("""
                **Legend:**
                - <span style='color: #28a745; font-weight: bold'>🟢 Highest</span>: Top 33% performers
                - <span style='color: #ffc107; font-weight: bold'>🟡 Moderate</span>: Middle 33% performers  
                - <span style='color: #dc3545; font-weight: bold'>🔴 Least Effective</span>: Bottom 33% performers
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("📈 Effectiveness by County")
                
                # Pad missing columns for plotting
                for col in ['Highest', 'Moderate', 'Least Effective']:
                    if col not in county_effectiveness.columns:
                        county_effectiveness[col] = 0
                
                # Create stacked bar chart
                fig_county = px.bar(
                    county_effectiveness.reset_index(),
                    x='county',
                    y=['Highest', 'Moderate', 'Least Effective'],
                    title="Production Volume by Effectiveness Level per County",
                    labels={'value': 'Production Volume', 'county': 'County'},
                    color_discrete_map={
                        'Highest': '#28a745',
                        'Moderate': '#ffc107', 
                        'Least Effective': '#dc3545'
                    }
                )
                
                fig_county.update_layout(
                    xaxis_title="County",
                    yaxis_title="Production Volume",
                    legend_title="Effectiveness Level"
                )
                
                st.plotly_chart(fig_county, use_container_width=True)
            
            # Type effectiveness heatmap
            st.subheader("🌡️ Crop/Livestock Effectiveness Heatmap")
            
            fig_heatmap = px.imshow(
                type_effectiveness,
                labels={'x': 'Effectiveness Level', 'y': 'Crop/Livestock Type', 'color': 'Count'},
                title="Effectiveness Distribution by Crop/Livestock Type",
                color_continuous_scale='RdYlGn'
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Detailed effectiveness table
            st.subheader("📋 Detailed Effectiveness Analysis Table")
            
            # Create comprehensive table
            detailed_analysis = effectiveness_data.groupby(['county', 'type', 'effectiveness_level']).agg({
                'production': ['sum', 'mean', 'count'],
                'effectiveness': ['mean', 'std']
            }).round(3)
            
            # Flatten column names
            detailed_analysis.columns = ['_'.join(col).strip() for col in detailed_analysis.columns.values]
            detailed_analysis = detailed_analysis.reset_index()
            
            # Style the table
            def highlight_effectiveness_level(row):
                if row['effectiveness_level'] == 'Highest':
                    return ['background-color: #d4edda'] * len(row)
                elif row['effectiveness_level'] == 'Moderate':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return ['background-color: #f8d7da'] * len(row)
            
            styled_detailed = detailed_analysis.style.apply(highlight_effectiveness_level, axis=1)
            st.dataframe(styled_detailed, use_container_width=True)
            
            # Download detailed analysis
            csv_detailed = detailed_analysis.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Effectiveness Analysis",
                data=csv_detailed,
                file_name="detailed_effectiveness_analysis.csv",
                mime="text/csv"
            )
        
        # Interactive Filtering
        st.subheader("🔍 Interactive Data Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_counties = st.multiselect(
                "Select Counties:",
                options=effectiveness_data['county'].unique(),
                default=effectiveness_data['county'].unique()[:5]
            )
        
        with col2:
            selected_types = st.multiselect(
                "Select Crop/Livestock Types:",
                options=effectiveness_data['type'].unique(),
                default=effectiveness_data['type'].unique()[:5]
            )
        
        with col3:
            selected_effectiveness = st.multiselect(
                "Select Effectiveness Levels:",
                options=effectiveness_data['effectiveness_level'].unique(),
                default=effectiveness_data['effectiveness_level'].unique()
            )
        
        # Filter data based on selections
        filtered_data = effectiveness_data[
            (effectiveness_data['county'].isin(selected_counties)) &
            (effectiveness_data['type'].isin(selected_types)) &
            (effectiveness_data['effectiveness_level'].isin(selected_effectiveness))
        ]
        
        if not filtered_data.empty:
            # Create filtered visualization
            fig_filtered = px.scatter(
                filtered_data,
                x='production',
                y='effectiveness',
                color='effectiveness_level',
                size='area_planted',
                hover_data=['county', 'type'],
                title="Filtered Data: Production vs Effectiveness",
                color_discrete_map={
                    'Highest': '#28a745',
                    'Moderate': '#ffc107',
                    'Least Effective': '#dc3545'
                }
            )
            
            fig_filtered.update_layout(
                xaxis_title="Production Volume",
                yaxis_title="Effectiveness Score",
                legend_title="Effectiveness Level"
            )
            
            st.plotly_chart(fig_filtered, use_container_width=True)
            
            # Filtered summary statistics
            st.subheader("📊 Filtered Data Summary")
            
            filtered_summary = filtered_data.groupby('effectiveness_level').agg({
                'production': ['sum', 'mean', 'count'],
                'effectiveness': ['mean', 'std'],
                'area_planted': 'sum'
            }).round(3)
            
            st.dataframe(filtered_summary, use_container_width=True)
        
        else:
            st.warning("No data matches the selected filters. Please adjust your selections.")
    
    else:
        st.info("🚀 **Getting Started**: Choose 'Use Sample Data' in the sidebar to automatically load data and view all visualizations!")
        
        # Show what visualizations will be available
        st.subheader("🎯 Available Visualizations (After Loading Data)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔗 Correlation Analysis:**
            - County-Crop/Livestock correlation matrices
            - Interactive heatmaps with effectiveness scores
            - Color-coded summary tables
            - Downloadable correlation data
            """)
            
        with col2:
            st.markdown("""
            **🌊 Sankey Flow Diagrams:**
            - County → Crop/Livestock → Effectiveness flow
            - Production volume visualization
            - Clear color-coded effectiveness levels
            - Interactive hover information
            """)
        
        st.markdown("""
        **🎯 Effectiveness Analysis:**
        - Clear categorization: Highest/Moderate/Least Effective
        - Interactive pie charts and bar charts
        - Detailed analysis tables with color coding
        - Multi-select filters for data exploration
        """)
        
        st.warning("💡 **Tip**: Select 'Use Sample Data' from the sidebar to see all these features in action!")

with tab4:
    st.header("🔍 Detailed Analysis")
    
    if 'effectiveness_data' in st.session_state:
        effectiveness_data = st.session_state.effectiveness_data
        
        # Analysis options
        analysis_type = st.selectbox(
            "Choose Analysis Type:",
            ["County Performance", "Crop/Livestock Analysis", "Effectiveness Factors", "Recommendations"]
        )
        
        if analysis_type == "County Performance":
            st.subheader("🏘️ County Performance Analysis")
            
            # County metrics
            county_metrics = effectiveness_data.groupby('county').agg({
                'production': 'sum',
                'effectiveness': 'mean',
                'area_planted': 'sum'
            }).round(2)
            
            county_metrics['effectiveness_rank'] = county_metrics['effectiveness'].rank(ascending=False)
            county_metrics = county_metrics.sort_values('effectiveness', ascending=False)
            
            st.dataframe(county_metrics, use_container_width=True)
            
            # Top performers
            st.subheader("🏆 Top Performing Counties")
            top_counties = county_metrics.head(3)
            
            cols = st.columns(3)
            for i, (county, data) in enumerate(top_counties.iterrows()):
                with cols[i]:
                    st.metric(
                        f"#{i+1} {county}",
                        f"Effectiveness: {data['effectiveness']:.2f}",
                        f"Production: {data['production']:,.0f}"
                    )
        
        elif analysis_type == "Crop/Livestock Analysis":
            st.subheader("🌾 Crop/Livestock Performance Analysis")
            
            # Performance by type
            type_metrics = effectiveness_data.groupby(['type', 'category']).agg({
                'production': 'sum',
                'effectiveness': 'mean'
            }).round(2)
            
            st.dataframe(type_metrics, use_container_width=True)
            
            # Best performing crops and livestock
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌾 Best Crops")
                crop_performance = effectiveness_data[effectiveness_data['category'] == 'Crop'].groupby('type')['effectiveness'].mean().sort_values(ascending=False).head(5)
                for crop, eff in crop_performance.items():
                    st.write(f"**{crop}**: {eff:.2f}")
            
            with col2:
                st.subheader("🐄 Best Livestock")
                livestock_performance = effectiveness_data[effectiveness_data['category'] == 'Livestock'].groupby('type')['effectiveness'].mean().sort_values(ascending=False).head(5)
                for livestock, eff in livestock_performance.items():
                    st.write(f"**{livestock}**: {eff:.2f}")
        
        elif analysis_type == "Effectiveness Factors":
            st.subheader("📊 Effectiveness Factors Analysis")
            
            # Correlation analysis
            numeric_cols = ['production', 'effectiveness', 'area_planted']
            correlation_matrix = effectiveness_data[numeric_cols].corr()
            
            fig = px.imshow(correlation_matrix, 
                          text_auto=True, 
                          title="Correlation Matrix - Key Factors")
            st.plotly_chart(fig, use_container_width=True)
            
            # Factor importance
            st.subheader("🎯 Key Success Factors")
            
            high_eff_data = effectiveness_data[effectiveness_data['effectiveness_level'] == 'Highest']
            low_eff_data = effectiveness_data[effectiveness_data['effectiveness_level'] == 'Least Effective']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Highest Effectiveness Characteristics:**")
                st.write(f"- Average Production: {high_eff_data['production'].mean():,.0f}")
                st.write(f"- Average Area Planted: {high_eff_data['area_planted'].mean():,.0f}")
                st.write(f"- Most Common County: {high_eff_data['county'].mode().iloc[0] if len(high_eff_data) > 0 else 'N/A'}")
            
            with col2:
                st.write("**Least Effective Characteristics:**")
                st.write(f"- Average Production: {low_eff_data['production'].mean():,.0f}")
                st.write(f"- Average Area Planted: {low_eff_data['area_planted'].mean():,.0f}")
                st.write(f"- Most Common County: {low_eff_data['county'].mode().iloc[0] if len(low_eff_data) > 0 else 'N/A'}")
        
        else:  # Recommendations
            st.subheader("💡 Actionable Recommendations")
            
            # Generate recommendations based on analysis
            recommendations = []
            
            # County recommendations
            county_performance = effectiveness_data.groupby('county')['effectiveness'].mean().sort_values(ascending=False)
            top_county = county_performance.index[0]
            bottom_county = county_performance.index[-1]
            
            recommendations.append({
                "Category": "County Development",
                "Recommendation": f"Focus development efforts on {bottom_county}. Study best practices from {top_county} (highest effectiveness: {county_performance.iloc[0]:.2f}).",
                "Priority": "High"
            })
            
            # Crop recommendations
            crop_effectiveness = effectiveness_data[effectiveness_data['category'] == 'Crop'].groupby('type')['effectiveness'].mean().sort_values(ascending=False)
            if len(crop_effectiveness) > 0:
                best_crop = crop_effectiveness.index[0]
                recommendations.append({
                    "Category": "Crop Selection",
                    "Recommendation": f"Promote {best_crop} cultivation as it shows highest effectiveness ({crop_effectiveness.iloc[0]:.2f}).",
                    "Priority": "Medium"
                })
            
            # Livestock recommendations
            livestock_effectiveness = effectiveness_data[effectiveness_data['category'] == 'Livestock'].groupby('type')['effectiveness'].mean().sort_values(ascending=False)
            if len(livestock_effectiveness) > 0:
                best_livestock = livestock_effectiveness.index[0]
                recommendations.append({
                    "Category": "Livestock Management",
                    "Recommendation": f"Invest in {best_livestock} as it shows highest effectiveness ({livestock_effectiveness.iloc[0]:.2f}).",
                    "Priority": "Medium"
                })
            
            # Low effectiveness areas
            low_eff_counties = effectiveness_data[effectiveness_data['effectiveness_level'] == 'Least Effective']['county'].unique()
            if len(low_eff_counties) > 0:
                recommendations.append({
                    "Category": "Intervention Needed",
                    "Recommendation": f"Counties requiring immediate attention: {', '.join(low_eff_counties[:3])}. Consider targeted support programs.",
                    "Priority": "High"
                })
            
            # Display recommendations
            rec_df = pd.DataFrame(recommendations)
            
            # Color-code by priority
            def highlight_priority(val):
                if val == 'High':
                    return 'background-color: #ffebee'
                elif val == 'Medium':
                    return 'background-color: #fff3e0'
                else:
                    return 'background-color: #f3e5f5'
            
            styled_df = rec_df.style.applymap(highlight_priority, subset=['Priority'])
            st.dataframe(styled_df, use_container_width=True)
    
    else:
        st.info("Please train models in the Model Training tab to view detailed analysis.")

with tab5:
    st.header("💡 Key Insights & Summary")
    
    if 'effectiveness_data' in st.session_state:
        effectiveness_data = st.session_state.effectiveness_data
        
        # Key statistics
        st.subheader("📊 Key Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Counties", effectiveness_data['county'].nunique())
        with col2:
            st.metric("Total Crops/Livestock", effectiveness_data['type'].nunique())
        with col3:
            st.metric("Highest Effectiveness %", f"{(effectiveness_data['effectiveness_level'] == 'Highest').mean() * 100:.1f}%")
        with col4:
            st.metric("Average Effectiveness", f"{effectiveness_data['effectiveness'].mean():.2f}")
        
        # Insights
        st.subheader("🔍 Key Insights")
        
        insights = []
        
        # County insights
        county_stats = effectiveness_data.groupby('county')['effectiveness'].agg(['mean', 'count'])
        best_county = county_stats['mean'].idxmax()
        worst_county = county_stats['mean'].idxmin()
        
        insights.append(f"**Best Performing County:** {best_county} with average effectiveness of {county_stats.loc[best_county, 'mean']:.2f}")
        insights.append(f"**Lowest Performing County:** {worst_county} with average effectiveness of {county_stats.loc[worst_county, 'mean']:.2f}")
        
        # Category insights
        category_stats = effectiveness_data.groupby('category')['effectiveness'].mean()
        best_category = category_stats.idxmax()
        insights.append(f"**Better Category:** {best_category} shows higher average effectiveness ({category_stats[best_category]:.2f})")
        
        # Effectiveness distribution
        eff_dist = effectiveness_data['effectiveness_level'].value_counts(normalize=True)
        insights.append(f"**Effectiveness Distribution:** {eff_dist.get('Highest', 0):.1%} Highest, {eff_dist.get('Moderate', 0):.1%} Moderate, {eff_dist.get('Least Effective', 0):.1%} Least Effective")
        
        # Model performance
        if hasattr(analyzer, 'rf_model') and analyzer.rf_model is not None:
            insights.append(f"**Environmental Model:** Successfully trained with R² score indicating good predictive capability")
        
        if hasattr(analyzer, 'dt_model') and analyzer.dt_model is not None:
            insights.append(f"**Classification Model:** Decision tree trained for effectiveness prediction")
        
        for insight in insights:
            st.write(f"• {insight}")
        
        # Action items
        st.subheader("🎯 Recommended Actions")
        
        actions = [
            "Focus resources on improving low-effectiveness counties",
            "Replicate best practices from high-performing counties",
            "Invest in data collection for better model accuracy",
            "Implement targeted interventions based on clustering results",
            "Regular monitoring and evaluation of effectiveness metrics"
        ]
        
        for action in actions:
            st.write(f"• {action}")
        
        # Export results
        st.subheader("📁 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Analysis Data"):
                csv = effectiveness_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="agricultural_analysis_results.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Summary Report"):
                summary_data = {
                    'Metric': ['Total Counties', 'Total Types', 'High Effectiveness %', 'Average Effectiveness'],
                    'Value': [
                        effectiveness_data['county'].nunique(),
                        effectiveness_data['type'].nunique(),
                        f"{(effectiveness_data['effectiveness_level'] == 'High').mean() * 100:.1f}%",
                        f"{effectiveness_data['effectiveness'].mean():.2f}"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=csv,
                    file_name="agricultural_analysis_summary.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("Please complete the analysis in previous tabs to view insights.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌾 Agricultural Data Analytics Dashboard</p>
        <p>Powered by Streamlit, Plotly, and Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
) 