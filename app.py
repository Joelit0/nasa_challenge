import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Exoplanet Classification System",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
    .feature-desc {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .feature-name {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.3rem;
    }
    .feature-text {
        color: #4b5563;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load ML model and scaler
@st.cache_resource
def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        with open('exoplanet_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure 'exoplanet_model.pkl', 'scaler.pkl', and 'feature_names.pkl' are in the same directory.")
        return None, None, None

def prepare_data(df, feature_names):
    """Prepare data for prediction - ensure correct columns and order"""
    # Check if required features exist
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        st.warning(f"Missing features: {missing_features}")
        return None
    
    # Select and order features correctly
    X = df[feature_names].copy()
    
    # Handle missing values (same as training)
    X = X.fillna(X.median())
    
    return X

def predict_exoplanets(model, scaler, X):
    """Make predictions using the trained model"""
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions and probabilities
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return predictions, probabilities

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    'planet_radius': 'Estimated radius of the exoplanet, usually expressed in Earth radii.',
    'transit_depth': "Fractional decrease in the star's brightness during transit, roughly equal to (Rₚ/R★)².",
    'transit_duration': "Duration (in hours) of the planet's transit across the stellar disk.",
    'orbital_period': 'Time (in days) the planet takes to complete one full orbit around its star.',
    'eq_temperature': 'Estimated equilibrium temperature of the planet, assuming radiative balance with its star.',
    'stellar_temp': 'Effective surface temperature of the host star, in Kelvin.',
    'stellar_radius': 'Radius of the host star, typically in solar radii.',
    'logg': 'Logarithm of the stellar surface gravity (log g, in cgs units).',
    'data_source': 'Indicates whether the data point comes from the TOI (TESS) or KOI (Kepler) catalog.'
}

# Header
st.markdown('<h1 class="main-header">Exoplanet Classification System</h1>', unsafe_allow_html=True)
st.markdown("""
    <p class='sub-header'>
    Machine learning-powered analysis of astronomical data for exoplanet identification
    </p>
""", unsafe_allow_html=True)

# Load model
model, scaler, feature_names = load_model()

if model is None:
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Data Upload & Prediction", 
    "Feature Documentation", 
    "Model Information",
    "About"
])

with tab1:
    st.header("Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Select a CSV file containing exoplanet candidate data",
        type=['csv'],
        help="File should contain the required features for prediction"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"File loaded successfully. {len(df)} records found.")
            
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Run Classification", type="primary", use_container_width=True):
                    with st.spinner("Processing data and running predictions..."):
                        # Prepare data
                        X = prepare_data(df, feature_names)
                        
                        if X is not None:
                            # Make predictions
                            predictions, probabilities = predict_exoplanets(model, scaler, X)
                            
                            # Add results to dataframe
                            df_results = df.copy()
                            df_results['Classification'] = ['Exoplanet' if p == 1 else 'Not Exoplanet' for p in predictions]
                            df_results['Confidence'] = [f"{max(prob):.1%}" for prob in probabilities]
                            df_results['Probability_Exoplanet'] = probabilities[:, 1]
                            df_results['Probability_Not_Exoplanet'] = probabilities[:, 0]
                            
                            st.success("Classification completed successfully")
                            
                            # Metrics
                            st.subheader("Results Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            exoplanets = sum(predictions == 1)
                            avg_confidence = np.mean([max(prob) for prob in probabilities])
                            
                            with col1:
                                st.metric("Total Objects", len(df_results))
                            with col2:
                                st.metric("Classified as Exoplanet", exoplanets, 
                                         delta=f"{exoplanets/len(df_results)*100:.1f}%")
                            with col3:
                                st.metric("Classified as Non-Exoplanet", len(df_results) - exoplanets)
                            with col4:
                                st.metric("Average Confidence", f"{avg_confidence:.1%}")
                            
                            st.markdown("---")
                            
                            # Results table
                            st.subheader("Classification Results")
                            
                            # Filter options
                            col1, col2 = st.columns(2)
                            with col1:
                                filter_option = st.selectbox(
                                    "Filter by classification:",
                                    ["All Objects", "Exoplanets Only", "Non-Exoplanets Only"]
                                )
                            with col2:
                                confidence_threshold = st.slider(
                                    "Minimum confidence threshold:",
                                    0.0, 1.0, 0.0, 0.05,
                                    format="%.0f%%"
                                )
                            
                            # Apply filters
                            df_filtered = df_results.copy()
                            if filter_option == "Exoplanets Only":
                                df_filtered = df_filtered[df_filtered['Classification'] == 'Exoplanet']
                            elif filter_option == "Non-Exoplanets Only":
                                df_filtered = df_filtered[df_filtered['Classification'] == 'Not Exoplanet']
                            
                            df_filtered = df_filtered[df_filtered['Probability_Exoplanet'] >= confidence_threshold]
                            
                            st.dataframe(
                                df_filtered.style.background_gradient(
                                    subset=['Probability_Exoplanet'],
                                    cmap='RdYlGn'
                                ),
                                use_container_width=True
                            )
                            
                            # Download results
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                "Download Results (CSV)",
                                csv,
                                "exoplanet_predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                            st.markdown("---")
                            
                            # Visualizations
                            st.subheader("Data Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                # Classification pie chart
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=['Exoplanet', 'Not Exoplanet'],
                                    values=[exoplanets, len(df_results) - exoplanets],
                                    hole=0.4,
                                    marker_colors=['#667eea', '#764ba2']
                                )])
                                fig_pie.update_layout(
                                    title="Classification Distribution",
                                    height=400
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with viz_col2:
                                # Confidence distribution
                                fig_hist = px.histogram(
                                    df_results,
                                    x='Probability_Exoplanet',
                                    color='Classification',
                                    title="Probability Distribution",
                                    labels={'Probability_Exoplanet': 'Exoplanet Probability'},
                                    nbins=30,
                                    color_discrete_map={
                                        'Exoplanet': '#667eea',
                                        'Not Exoplanet': '#764ba2'
                                    }
                                )
                                fig_hist.update_layout(height=400)
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Feature relationships
                            st.subheader("Feature Analysis")
                            
                            numeric_cols = [col for col in feature_names if col in df_results.columns]
                            
                            if len(numeric_cols) >= 2:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    x_axis = st.selectbox("X-axis feature:", numeric_cols, index=0)
                                with col2:
                                    y_axis = st.selectbox("Y-axis feature:", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                                
                                fig_scatter = px.scatter(
                                    df_results,
                                    x=x_axis,
                                    y=y_axis,
                                    color='Classification',
                                    size='Probability_Exoplanet',
                                    hover_data=numeric_cols[:5],
                                    title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
                                    color_discrete_map={
                                        'Exoplanet': '#667eea',
                                        'Not Exoplanet': '#764ba2'
                                    }
                                )
                                fig_scatter.update_layout(height=500)
                                st.plotly_chart(fig_scatter, use_container_width=True)
                            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

with tab2:
    st.header("Feature Documentation")
    
    st.markdown("""
    The classification model requires the following features. All features must be present in the uploaded CSV file.
    """)
    
    st.markdown("---")
    
    for i, (feature, description) in enumerate(FEATURE_DESCRIPTIONS.items(), 1):
        if feature in feature_names or feature == 'data_source':
            st.markdown(f"""
            <div class="feature-desc">
                <div class="feature-name">{i}. {feature}</div>
                <div class="feature-text">{description}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("Typical Value Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Planetary Properties:**
        - Planet Radius: 0.5 - 20+ Earth radii
        - Transit Depth: 100 - 50,000 ppm
        - Transit Duration: 0.5 - 12 hours
        - Orbital Period: 0.5 - 500+ days
        - Equilibrium Temperature: 200 - 3,000 K
        """)
    
    with col2:
        st.markdown("""
        **Stellar Properties:**
        - Stellar Temperature: 3,000 - 10,000 K
        - Stellar Radius: 0.1 - 10 solar radii
        - Surface Gravity (logg): 3.5 - 5.0
        - Data Source: 'TOI' or 'KOI'
        """)

with tab3:
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **Algorithm:** XGBoost Classifier
        
        **Hyperparameters:**
        - Number of Trees: 300
        - Maximum Depth: 5
        - Learning Rate: 0.05
        - Subsample Ratio: 0.8
        - Feature Sampling: 0.8
        
        **Data Preprocessing:**
        - Normalization: MinMax Scaling (0-1)
        - Missing Values: Median imputation
        - Feature Count: 8
        """)
    
    with col2:
        st.subheader("Performance Metrics")
        st.markdown("""
        **Training Dataset:**
        - Combined KOI (Kepler) + TOI (TESS)
        - Total Records: 17,267
        - Train/Test Split: 80/20
        
        **Model Performance:**
        - Test Accuracy: 76.5%
        - Cross-Validation: 5-fold stratified
        - Mean CV Score: 76.2%
        - Standard Deviation: 0.3%
        """)
    
    st.markdown("---")
    
    if model is not None and hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance Analysis")
        
        st.markdown("""
        The chart below shows the relative importance of each feature in the model's decision-making process.
        Features with higher importance have a greater influence on the classification outcome.
        """)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Format feature names for display
        importance_df['Feature_Display'] = importance_df['Feature'].str.replace('_', ' ').str.title()
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature_Display',
            orientation='h',
            title="Feature Importance Scores",
            color='Importance',
            color_continuous_scale='viridis',
            labels={'Feature_Display': 'Feature', 'Importance': 'Importance Score'}
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Show importance values in a table
        with st.expander("View Detailed Importance Values"):
            importance_display = importance_df[['Feature', 'Importance']].copy()
            importance_display['Importance'] = importance_display['Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(importance_display, use_container_width=True, hide_index=True)

with tab4:
    st.header("About This System")
    
    st.markdown("""
    ### Overview
    
    This application uses machine learning to classify astronomical objects as potential exoplanets 
    based on observational data from NASA's Kepler and TESS missions.
    
    ### Data Sources
    
    **KOI (Kepler Objects of Interest):**
    - Kepler Space Telescope observations (2009-2018)
    - Focused on a single field of view
    - Over 4,000 confirmed exoplanets
    
    **TOI (TESS Objects of Interest):**
    - Transiting Exoplanet Survey Satellite (2018-present)
    - All-sky survey
    - Ongoing discoveries
    
    ### Classification Method
    
    The system employs gradient boosting (XGBoost) to learn patterns in the observational data that
    distinguish confirmed exoplanets from false positives. The model was trained on validated classifications
    from NASA's exoplanet archive.
    
    ### Use Cases
    
    - **Research:** Preliminary screening of exoplanet candidates
    - **Education:** Understanding exoplanet detection methods
    - **Data Analysis:** Exploring relationships between stellar and planetary properties
    
    ### Limitations
    
    - Predictions are probabilistic and should be validated with additional observations
    - Model performance depends on data quality and completeness
    - Best suited for transit method detections similar to training data
    
    ### Technical Details
    
    **Development:**
    - Framework: Streamlit
    - ML Library: XGBoost, scikit-learn
    - Visualization: Plotly
    
    **Version:** 1.0.0
    
    **Last Updated:** 2025
    """)
    
    st.markdown("---")
    
    st.subheader("Contact & Support")
    st.info("""
    For questions, issues, or suggestions, please contact the development team or 
    refer to the project documentation.
    """)
