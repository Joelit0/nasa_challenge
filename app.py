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
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


# Load ML model and scaler
@st.cache_resource
def load_model():
    """Load the trained model pipeline and feature names"""
    try:
        with open("exoplanet_pipeline.pkl", "rb") as f:
            model = pickle.load(f)
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info(
            "Please ensure 'exoplanet_pipeline.pkl' and 'feature_names.pkl' are in the same directory."
        )
        return None, None


def prepare_data(df, feature_names):
    """Prepare data for prediction"""
    missing_features = set(feature_names) - set(df.columns)
    if missing_features:
        st.warning(f"Missing features: {missing_features}")
        return None

    X = df[feature_names].copy()
    X = X.fillna(X.median())

    return X


def predict_exoplanets(model, X):
    """Make predictions using the trained pipeline"""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return predictions, probabilities


# Feature descriptions
FEATURE_DESCRIPTIONS = {
    "planet_radius": "Radio estimado del exoplaneta (en radios terrestres).",
    "transit_depth": "Porcentaje de atenuación del brillo estelar durante el tránsito (ppm).",
    "transit_duration": "Duración total del tránsito planetario (horas).",
    "orbital_period": "Tiempo que tarda el planeta en completar una órbita alrededor de su estrella (días).",
    "stellar_temp": "Temperatura efectiva de la superficie estelar (Kelvin).",
    "stellar_radius": "Radio de la estrella anfitriona (radios solares).",
    "stellar_logg": "Logaritmo de la gravedad superficial estelar (log g, en cm/s²).",
    "snr": "Relación señal-ruido del modelo de tránsito (Signal-to-Noise Ratio).",
    "total_fp_flags": "Número total de banderas de falsos positivos detectadas (solo en Kepler).",
}

# Header
st.markdown(
    '<h1 class="main-header">Exoplanet Classification System</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <p class='sub-header'>
    Machine learning-powered analysis of astronomical data for exoplanet identification
    </p>
""",
    unsafe_allow_html=True,
)

# Load model
model, feature_names = load_model()

if model is None:
    st.stop()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Data Upload & Prediction", "Feature Documentation", "Model Information", "About"]
)

with tab1:
    st.header("Upload Dataset")

    uploaded_file = st.file_uploader(
        "Select a CSV file containing exoplanet candidate data",
        type=["csv"],
        help="File should contain the required features for prediction",
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
                if st.button(
                    "Run Classification", type="primary", use_container_width=True
                ):
                    with st.spinner("Processing data and running predictions..."):
                        # Prepare data
                        X = prepare_data(df, feature_names)

                        if X is not None:
                            # Make predictions
                            predictions, probabilities = predict_exoplanets(model, X)

                            CLASS_LABELS = {
                                0: "False Positive",
                                1: "Candidate",
                                2: "Confirmed Exoplanet",
                            }

                            df_results = df.copy()
                            df_results["Classification"] = [
                                CLASS_LABELS.get(p, "Unknown") for p in predictions
                            ]
                            df_results["Confidence"] = [
                                f"{max(prob):.1%}" for prob in probabilities
                            ]

                            df_results["Prob_False_Positive"] = probabilities[:, 0]
                            df_results["Prob_Candidate"] = probabilities[:, 1]
                            df_results["Prob_Confirmed"] = probabilities[:, 2]

                            # Add a column for the maximum probability (for confidence)
                            df_results["Max_Probability"] = df_results[
                                [
                                    "Prob_False_Positive",
                                    "Prob_Candidate",
                                    "Prob_Confirmed",
                                ]
                            ].max(axis=1)

                            st.success("Classification completed successfully")

                            # Metrics
                            st.subheader("Results Summary")
                            col1, col2, col3, col4 = st.columns(4)

                            count_fp = sum(predictions == 0)
                            count_cand = sum(predictions == 1)
                            count_conf = sum(predictions == 2)
                            avg_confidence = np.mean(
                                [max(prob) for prob in probabilities]
                            )

                            with col1:
                                st.metric("Total Objects", len(df_results))
                            with col2:
                                st.metric(
                                    "Confirmed Exoplanets",
                                    count_conf,
                                    delta=f"{count_conf / len(df_results) * 100:.1f}%",
                                )
                            with col3:
                                st.metric("Candidates", count_cand)
                            with col4:
                                st.metric("False Positives", count_fp)

                            st.caption(f"Average confidence: {avg_confidence:.1%}")

                            # Results table
                            st.subheader("Classification Results")

                            # Filter options
                            col1, col2 = st.columns(2)
                            with col1:
                                filter_option = st.selectbox(
                                    "Filter by classification:",
                                    [
                                        "All Objects",
                                        "Confirmed Exoplanets Only",
                                        "Candidates Only",
                                        "False Positives Only",
                                    ],
                                )
                            with col2:
                                confidence_threshold = st.slider(
                                    "Minimum confidence threshold:",
                                    0.0,
                                    1.0,
                                    0.0,
                                    0.05,
                                    format="%.0f%%",
                                )

                            # Apply filters
                            df_filtered = df_results.copy()
                            if filter_option == "Confirmed Exoplanets Only":
                                df_filtered = df_filtered[
                                    df_filtered["Classification"]
                                    == "Confirmed Exoplanet"
                                ]
                            elif filter_option == "Candidates Only":
                                df_filtered = df_filtered[
                                    df_filtered["Classification"] == "Candidate"
                                ]
                            elif filter_option == "False Positives Only":
                                df_filtered = df_filtered[
                                    df_filtered["Classification"] == "False Positive"
                                ]

                            df_filtered = df_filtered[
                                df_filtered["Max_Probability"] >= confidence_threshold
                            ]

                            st.dataframe(
                                df_filtered.style.background_gradient(
                                    subset=["Max_Probability"], cmap="RdYlGn"
                                ),
                                use_container_width=True,
                            )

                            # Download results
                            csv = df_results.to_csv(index=False)
                            st.download_button(
                                "Download Results (CSV)",
                                csv,
                                "exoplanet_predictions.csv",
                                "text/csv",
                                use_container_width=True,
                            )

                            st.markdown("---")

                            # Visualizations
                            st.subheader("Data Visualizations")

                            viz_col1, viz_col2 = st.columns(2)

                            with viz_col1:
                                fig_pie = go.Figure(
                                    data=[
                                        go.Pie(
                                            labels=[
                                                "Confirmed Exoplanet",
                                                "Candidate",
                                                "False Positive",
                                            ],
                                            values=[count_conf, count_cand, count_fp],
                                            hole=0.4,
                                            marker_colors=[
                                                "#10b981",  # Green for confirmed
                                                "#3b82f6",  # Blue for candidate
                                                "#ef4444",  # Red for false positive
                                            ],
                                        )
                                    ]
                                )
                                fig_pie.update_layout(
                                    title="Classification Distribution", height=400
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)

                            with viz_col2:
                                # Confidence distribution - CORREGIDO
                                # Use Max_Probability instead of Probability_Exoplanet
                                fig_hist = px.histogram(
                                    df_results,
                                    x="Max_Probability",
                                    color="Classification",
                                    title="Confidence Distribution",
                                    labels={"Max_Probability": "Maximum Probability"},
                                    nbins=20,
                                    color_discrete_map={
                                        "Confirmed Exoplanet": "#10b981",
                                        "Candidate": "#3b82f6",
                                        "False Positive": "#ef4444",
                                    },
                                )
                                fig_hist.update_layout(height=400)
                                st.plotly_chart(fig_hist, use_container_width=True)

                            # Feature relationships
                            st.subheader("Feature Analysis")

                            numeric_cols = [
                                col
                                for col in feature_names
                                if col in df_results.columns
                            ]

                            if len(numeric_cols) >= 2:
                                col1, col2 = st.columns(2)

                                with col1:
                                    x_axis = st.selectbox(
                                        "X-axis feature:", numeric_cols, index=0
                                    )
                                with col2:
                                    y_axis = st.selectbox(
                                        "Y-axis feature:",
                                        numeric_cols,
                                        index=1 if len(numeric_cols) > 1 else 0,
                                    )

                                fig_scatter = px.scatter(
                                    df_results,
                                    x=x_axis,
                                    y=y_axis,
                                    color="Classification",
                                    size="Max_Probability",
                                    hover_data=numeric_cols[:5],
                                    title=f"{x_axis.replace('_', ' ').title()} vs {y_axis.replace('_', ' ').title()}",
                                    color_discrete_map={
                                        "Confirmed Exoplanet": "#10b981",
                                        "Candidate": "#3b82f6",
                                        "False Positive": "#ef4444",
                                    },
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
        if feature in feature_names:
            st.markdown(
                f"""
            <div class="feature-desc">
                <div class="feature-name">{i}. {feature}</div>
                <div class="feature-text">{description}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

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
        """)

    with col2:
        st.markdown("""
        **Stellar Properties:**
        - Stellar Temperature: 3,000 - 10,000 K
        - Stellar Radius: 0.1 - 10 solar radii
        - Surface Gravity (logg): 3.5 - 5.0
        - SNR: 5 - 100+
        """)

with tab3:
    st.header("Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        st.markdown("""
        **Algorithm:** XGBoost (Gradient Boosting Decision Trees)
        **Classification Type:** Multi-Class (3 categories: False Positive, Candidate, Confirmed)

        **Hyperparameters:**
        - Number of Trees: 300
        - Maximum Depth: 5
        - Learning Rate: 0.10
        - Objective: `multi:softprob`
        - Evaluation Metric: Multi-class Log Loss (`mlogloss`)

        **Data Preprocessing (inside pipeline):**
        - StandardScaler normalization
        - Median imputation for missing values
        - SMOTE balancing of minority classes
        - Total features: 9
        """)

    with col2:
        st.subheader("Performance Metrics")
        st.markdown("""
        **Training Dataset:**
        - Combined KOI (Kepler) + TOI (TESS)
        - Total Records: ~17,000
        - Train/Test Split: 75/25 (Stratified)

        **Model Performance:**
        - F1-Score (Macro): **0.8017**
        - ROC-AUC (One-vs-Rest): **0.9315**
        - Balanced Accuracy: **~0.78**
        - Confusions mainly between *Candidate* and *Confirmed*, consistent with physical similarities
        """)

    st.markdown("---")

    if model is not None and hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance Analysis")

        st.markdown("""
        The chart below shows how much each feature contributes to the model's decision-making process.
        Features with higher importance values have a stronger influence on the classification outcome.
        """)

        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        ).sort_values("Importance", ascending=False)

        # Format feature names for readability
        importance_df["Feature_Display"] = (
            importance_df["Feature"].str.replace("_", " ").str.title()
        )

        fig_importance = px.bar(
            importance_df,
            x="Importance",
            y="Feature_Display",
            orientation="h",
            title="Feature Importance Scores",
            color="Importance",
            color_continuous_scale="viridis",
            labels={"Feature_Display": "Feature", "Importance": "Importance Score"},
        )
        fig_importance.update_layout(height=400)
        st.plotly_chart(fig_importance, use_container_width=True)

        with st.expander("View Detailed Importance Values"):
            importance_display = importance_df[["Feature", "Importance"]].copy()
            importance_display["Importance"] = importance_display["Importance"].apply(
                lambda x: f"{x:.4f}"
            )
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
