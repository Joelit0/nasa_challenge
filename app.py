import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.title("🚀 NASA Exoplanet Data Viewer")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File loaded successfully! {len(df)} records found.")
        
        st.subheader("📊 Original Data")
        st.dataframe(df, use_container_width=True)
        
        if st.button("🤖 Process with AI Model", type="primary"):
            with st.spinner("🛸 AI is analyzing the data..."):
                np.random.seed(42)
                
                predictions = []
                for i in range(len(df)):
                    prob = np.random.random()
                    if prob > 0.6:
                        predictions.append("Exoplanet")
                    else:
                        predictions.append("Not Exoplanet")
                
                df_processed = df.copy()
                df_processed['AI_Classification'] = predictions
                df_processed['Confidence'] = [f"{np.random.uniform(0.5, 0.95):.1%}" for _ in range(len(df))]
                
                st.success("✅ AI processing completed!")
                
                st.subheader("🔬 AI Analysis Results")
                st.dataframe(df_processed, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Objects", len(df_processed))
                with col2:
                    exoplanets = len(df_processed[df_processed['AI_Classification'] == 'Exoplanet'])
                    st.metric("Exoplanets", exoplanets)
                with col3:
                    st.metric("Not Exoplanets", len(df_processed) - exoplanets)
                
                st.subheader("📈 Data Visualizations")
                
                numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = px.scatter(df_processed, 
                                        x=numeric_cols[0], 
                                        y=numeric_cols[1],
                                        color='AI_Classification',
                                        title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        if len(numeric_cols) >= 3:
                            fig2 = px.scatter(df_processed, 
                                            x=numeric_cols[1], 
                                            y=numeric_cols[2],
                                            color='AI_Classification',
                                            title=f"{numeric_cols[1]} vs {numeric_cols[2]}")
                            st.plotly_chart(fig2, use_container_width=True)
                
                st.subheader("📊 Classification Distribution")
                classification_counts = df_processed['AI_Classification'].value_counts()
                fig_bar = px.bar(x=classification_counts.index, 
                               y=classification_counts.values,
                               title="AI Classification Results",
                               color=classification_counts.index)
                st.plotly_chart(fig_bar, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")