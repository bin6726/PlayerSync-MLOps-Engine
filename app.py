import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="PlayerSync AI", layout="wide", page_icon="🎮")

# 2. Secure ML Asset Loader
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'player_cluster_model.pkl')
    scaler_path = os.path.join(base_dir, 'data_scaler.pkl')
    return joblib.load(model_path), joblib.load(scaler_path)

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("⚠️ Model files missing from folder!")

# 3. Header Section
st.title("🎮 PlayerSync AI — Production MLOps Engine")
st.subheader("Enterprise User Behavioral Segmentation & Machine Learning Interface")
st.markdown("---")

# 4. Two-Column UI Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🕹️ Live Feature Inputs")
    st.write("Adjust sliders to simulate player metrics.")
    
    duration = st.slider("Average Session Duration (Mins)", 2.0, 150.0, 30.0, 1.0)
    clicks = st.slider("Total Screen Clicks", 10, 1500, 250, 10)
    levels = st.slider("Levels Completed", 0, 50, 5, 1)
    spend = st.slider("In-App Expense (INR)", 0.0, 500.0, 0.0, 5.0)
    
    predict_btn = st.button("Analyze Player Persona 🚀", use_container_width=True)

with col2:
    st.header("📊 Deep-Learning Clustering Analysis")
    
    if predict_btn:
        # Create Dataframe with the EXACT corporate feature names the model saw at fit time
        input_df = pd.DataFrame([[duration, clicks, levels, spend]], columns=[
            'Avg_Session_Duration_Mins', 'Total_Screen_Clicks', 'Levels_Completed', 'In_App_Expense_INR'
        ])
        
        # Pass structured dataframe directly into scaler and model
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        
        archetypes = {
            0: {"name": "The Casual Browsers 🐢", "color": "#2ca02c", "desc": "Low playtime, minimal interactions, and zero spending."},
            1: {"name": "The Hardcore Grinders 🏆", "color": "#1f77b4", "desc": "Massive playtime and high level progression, but very low spending."},
            2: {"name": "The Whale Spenders 💎", "color": "#d62728", "desc": "Medium playtime but exceptionally high financial spending."}
        }
        
        result = archetypes.get(prediction, {"name": "Unknown Profile", "color": "#7f7f7f", "desc": "No matching archetype."})
        
        st.markdown(f"""
        <div style="background-color: {result['color']}; padding: 25px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">Predicted Archetype: {result['name']}</h2>
            <p style="font-size: 16px; margin-top: 10px; opacity: 0.9;">{result['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- BULLETPROOF VISUALIZATION ENGINE ---
        st.write("### 📈 Live Cluster Space Visualization")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'player_behavior_metrics.csv')
            df = pd.read_csv(csv_path)
            
            # Map whatever columns exist in the CSV file directly to the required model format
            mapping = {}
            for col in df.columns:
                cleaned = col.lower().replace(" ", "_")
                if "duration" in cleaned: mapping[col] = "Avg_Session_Duration_Mins"
                elif "click" in cleaned: mapping[col] = "Total_Screen_Clicks"
                elif "level" in cleaned: mapping[col] = "Levels_Completed"
                elif "expense" in cleaned or "spend" in cleaned: mapping[col] = "In_App_Expense_INR"
            
            df_scaled_prep = df.rename(columns=mapping)
            
            # Formulate clusters using the exact naming matrix order
            features = ['Avg_Session_Duration_Mins', 'Total_Screen_Clicks', 'Levels_Completed', 'In_App_Expense_INR']
            historical_scaled = scaler.transform(df_scaled_prep[features])
            df['Cluster_Label'] = model.predict(historical_scaled)
            
            cluster_map = {0: 'Casual Browsers', 1: 'Hardcore Grinders', 2: 'Whale Spenders'}
            df['Archetype'] = df['Cluster_Label'].map(cluster_map)
            df['Size'] = 40
            
            # Identify the original graph plotting axis tags safely
            x_axis_source = [col for col in df.columns if "click" in col.lower()][0]
            y_axis_source = [col for col in df.columns if "expense" in col.lower() or "spend" in col.lower()][0]
            
            # Append target tracker node row data matrix
            live_point = pd.DataFrame([{
                x_axis_source: float(clicks), 
                y_axis_source: float(spend), 
                'Archetype': 'CURRENT LIVE TARGET 🎯',
                'Size': 250
            }])
            
            plot_df = pd.concat([df[[x_axis_source, y_axis_source, 'Archetype', 'Size']], live_point], ignore_index=True)
            
            # Render Vega-Lite layout scatter chart layer
            st.vega_lite_chart(plot_df, {
                'width': 'container',
                'height': 400,
                'mark': {'type': 'circle', 'tooltip': True},
                'encoding': {
                    'x': {'field': x_axis_source, 'type': 'quantitative', 'title': 'Total Screen Clicks'},
                    'y': {'field': y_axis_source, 'type': 'quantitative', 'title': 'In-App Expense (INR)'},
                    'color': {
                        'field': 'Archetype', 
                        'type': 'nominal',
                        'scale': {
                            'domain': ['Casual Browsers', 'Hardcore Grinders', 'Whale Spenders', 'CURRENT LIVE TARGET 🎯'],
                            'range': ['#2ca02c', '#1f77b4', '#d62728', '#000000']
                        }
                    },
                    'size': {'field': 'Size', 'type': 'quantitative', 'scale': None}
                }
            })
            st.caption("💡 The massive black dot labeled CURRENT LIVE TARGET 🎯 tracks your slider inputs in real-time.")
            
        except Exception as chart_err:
            st.error(f"📊 Visualization Layer Error: {str(chart_err)}")
        
        st.write("### 🎛️ Input Feature Vector Check")
        st.table(pd.DataFrame([[duration, clicks, levels, spend]], columns=[
            "Avg Session Duration (Mins)", "Total Screen Clicks", "Levels Completed", "In-App Expense (INR)"
        ]))
        
    else:
        st.info("💡 Adjust the sliders and click the button to run the model.")
