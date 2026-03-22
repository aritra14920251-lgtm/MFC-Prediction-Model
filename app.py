import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import euclidean_distances

# --- Page Config ---
st.set_page_config(page_title="MFC Dual-Mode Predictor", page_icon="⚡", layout="wide")

# --- Feature Engineering (Must match mfc_model.py) ---
def engineer_features(df):
    df['BOD_COD_Ratio'] = df['BOD_in'] / (df['COD_in'] + 1e-6)
    df['Organic_Load'] = (df['COD_in'] * df['Volume']) / 1000.0
    df['pH_Dev'] = abs(df['pH_in'] - 7.2)
    return df

# --- Load Model & Data ---
@st.cache_resource
def load_assets():
    model = joblib.load("mfc_trained_model.pkl")
    df = pd.read_csv("mfc_data_v2.csv")
    # Map numbers to names for better visualization
    df['Wastewater_Type'] = df['WW_Type'].map({0: 'SBWW', 1: 'SIWW', 2: 'MIX'})
    return model, df

try:
    model, df = load_assets()
except:
    st.error("Model or data not found. Please run scripts first.")
    st.stop()

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e1e4e8; }
    h1 { color: #1a5276; font-weight: 800; }
    h3 { color: #21618c; margin-top: 25px; border-bottom: 2px solid #d4e6f1; padding-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("⚡ MFC Dual-Mode Prediction Dashboard")
st.markdown("""
Research-grounded predictions for **Slaughterhouse** and **Shrimp** MFC systems.
Toggle between patterns learned from **2000 augmented samples** or **Real-World Corrected** bias.
""")

# --- Sidebar Inputs ---
st.sidebar.header("🔬 Input Parameters")
ww_mapping = {"Slaughterhouse (SBWW)": 0, "Shrimp (SIWW)": 1, "Combined (MIX)": 2}
selected_ww = st.sidebar.selectbox("Wastewater Type", list(ww_mapping.keys()))
ww_type = ww_mapping[selected_ww]

volume = st.sidebar.number_input("Volume (mL)", 100, 10000, 1000, step=100)
cod_in = st.sidebar.number_input("Initial COD (ppm)", 500, 8000, 2500, step=50)
bod_in = st.sidebar.number_input("Initial BOD (ppm)", 200, 5000, 1200, step=50)
ph_in = st.sidebar.number_input("Initial pH Value", 4.0, 11.0, 7.0, step=0.1)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Prediction Control")
prediction_mode = st.sidebar.radio("Primary Mode", ["Real-World Corrected", "Grounded Synthetic"], 
                                   help="Real-World mode adjusts predictions toward specific journal data points.")
compare_mode = st.sidebar.checkbox("Compare Both Modes", value=True)

# --- Prediction Logic ---
def get_prediction(is_real):
    input_df = pd.DataFrame([[ww_type, volume, cod_in, bod_in, ph_in]], 
                            columns=['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in'])
    input_eng = engineer_features(input_df.copy())
    input_eng['Is_Real'] = 1 if is_real else 0
    
    input_scaled = model['scaler'].transform(input_eng[model['feature_cols']])
    preds = []
    for target in model['target_cols']:
        preds.append(model['models'][target].predict(input_scaled)[0])
    return np.array(preds)

pred_real = get_prediction(is_real=True)
pred_synth = get_prediction(is_real=False)

# Choose display based on radio
primary_pred = pred_real if prediction_mode == "Real-World Corrected" else pred_synth
comp_pred = pred_synth if prediction_mode == "Real-World Corrected" else pred_real

# --- Dashboard Tabs ---
tab_res, tab_match, tab_dist, tab_env, tab_heat = st.tabs([
    "💡 Predictive Results", 
    "📚 Research Matches", 
    "📊 Data Insights", 
    "📉 Environmental Trends",
    "🌡️ Correlation Heatmap"
])

with tab_res:
    col_main, col_comp = st.columns([1, 1] if compare_mode else [1, 0.01])
    
    with col_main:
        st.subheader(f"{prediction_mode} Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Voltage", f"{primary_pred[0]:.3f} V")
        c2.metric("Power Density", f"{primary_pred[1]:.2f} mW/m²")
        c3.metric("Coulombic Eff.", f"{primary_pred[2]:.1f} %")
        
        t1, t2, t3 = st.columns(3)
        t1.metric("Resulting COD", f"{int(primary_pred[3])} ppm")
        t2.metric("Resulting BOD", f"{int(primary_pred[4])} ppm")
        t3.metric("Final pH", f"{primary_pred[5]:.1f}")

    if compare_mode:
        with col_comp:
            other_mode = "Grounded Synthetic" if prediction_mode == "Real-World Corrected" else "Real-World Corrected"
            st.subheader(f"🔄 Comparative {other_mode}")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Voltage", f"{comp_pred[0]:.3f} V", f"{comp_pred[0]-primary_pred[0]:.3f} V", delta_color="normal")
            cc2.metric("Power Density", f"{comp_pred[1]:.2f} mW/m²", f"{comp_pred[1]-primary_pred[1]:.2f}", delta_color="normal")
            cc3.metric("Coulombic Eff.", f"{comp_pred[2]:.1f} %", f"{comp_pred[2]-primary_pred[2]:.1f}%")
            
            tt1, tt2, tt3 = st.columns(3)
            tt1.metric("Resulting COD", f"{int(comp_pred[3])} ppm", f"{int(comp_pred[3]-primary_pred[3])} ppm", delta_color="inverse")
            tt2.metric("Resulting BOD", f"{int(comp_pred[4])} ppm", f"{int(comp_pred[4]-primary_pred[4])} ppm", delta_color="inverse")
            tt3.metric("Final pH", f"{comp_pred[5]:.1f}", f"{comp_pred[5]-primary_pred[5]:.1f}")

with tab_match:
    st.subheader("📚 Closest Experimental Matches (From Journals)")
    st.info("Finding the 3 actual journal entries most similar to your input parameters...")
    real_only = df[df['Category'] == 'Real'].copy()
    if not real_only.empty:
        target_vector = np.array([[ww_type, volume, cod_in, bod_in, ph_in]])
        dataset_vectors = real_only[['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in']].values
        distances = euclidean_distances(target_vector, dataset_vectors)[0]
        real_only['similarity_score'] = 1 / (1 + distances)
        top_matches = real_only.sort_values('similarity_score', ascending=False).head(3)
        
        cols = st.columns(3)
        for i, (idx, row) in enumerate(top_matches.iterrows()):
            with cols[i]:
                st.markdown(f"**Match #{i+1}**")
                st.caption(f"Ref: {row['Reference']}")
                st.json({"In_COD": row['COD_in'], "In_pH": row['pH_in'], "Voltage": row['Voltage'], "Power": row['Power_Density']})
    else:
        st.warning("No real data points found for selection.")

with tab_dist:
    st.subheader("📊 Performance: Real vs Synthetic (Power vs Voltage)")
    real_df = df[df['Category'] == 'Real']
    synth_df = df[df['Category'] == 'Synthetic']
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**📖 Real Journal Data**")
        fig1_real = px.scatter(real_df, x="Power_Density", y="Voltage", color="Wastewater_Type", 
                               title="Real: Power Density vs Voltage",
                               labels={"Power_Density": "Power (mW/m²)", "Voltage": "Voltage (V)"},
                               category_orders={"Wastewater_Type": ["SBWW", "SIWW", "MIX"]})
        st.plotly_chart(fig1_real, use_container_width=True)
        
        fig2_real = px.scatter(real_df, x="pH_in", y="COD_out", color="Wastewater_Type",
                               title="Real: pH vs Resulting COD",
                               labels={"pH_in": "Initial pH", "COD_out": "COD Out (ppm)"})
        st.plotly_chart(fig2_real, use_container_width=True)

    with col_v2:
        st.markdown("**🤖 Synthetic Augmented Data**")
        fig1_synth = px.scatter(synth_df, x="Power_Density", y="Voltage", color="Wastewater_Type",
                                title="Synthetic: Power Density vs Voltage",
                                labels={"Power_Density": "Power (mW/m²)", "Voltage": "Voltage (V)"},
                                opacity=0.5)
        st.plotly_chart(fig1_synth, use_container_width=True)
        
        fig2_synth = px.scatter(synth_df, x="pH_in", y="COD_out", color="Wastewater_Type",
                                title="Synthetic: pH vs Resulting COD",
                                opacity=0.5)
        st.plotly_chart(fig2_synth, use_container_width=True)

with tab_env:
    st.subheader("📉 Environmental Trends (Real vs Synthetic)")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**📖 Real Trends**")
        fig3_real = px.scatter(real_df, x="COD_in", y="Power_Density", color="Wastewater_Type",
                               size="Volume", title="Real: COD vs Power",
                               labels={"COD_in": "COD in", "Power_Density": "Power"})
        st.plotly_chart(fig3_real, use_container_width=True)
        
        fig4_real = px.scatter(real_df, x="pH_in", y="Coulombic_Efficiency", color="Wastewater_Type",
                               title="Real: pH vs CE (%)", trendline="lowess")
        st.plotly_chart(fig4_real, use_container_width=True)

    with col_t2:
        st.markdown("**🤖 Synthetic Trends**")
        fig3_synth = px.scatter(synth_df, x="COD_in", y="Power_Density", color="Wastewater_Type",
                                size="Volume", title="Synth: COD vs Power", opacity=0.4)
        st.plotly_chart(fig3_synth, use_container_width=True)
        
        fig4_synth = px.scatter(synth_df, x="pH_in", y="Coulombic_Efficiency", color="Wastewater_Type",
                                title="Synth: pH vs CE (%)", trendline="lowess", opacity=0.4)
        st.plotly_chart(fig4_synth, use_container_width=True)

with tab_heat:
    st.subheader("🌡️ Correlation Map: Real-World vs Synthetic")
    numeric_cols = ['Volume', 'COD_in', 'BOD_in', 'pH_in', 'Voltage', 'Power_Density', 'Coulombic_Efficiency']
    
    # Real World Map (Full Width for visibility)
    st.markdown("### 📖 Real-World Journal Correlation")
    st.info("Showing the direct relationships found in the source research data.")
    corr_real = real_df[numeric_cols].corr()
    fig5_real = px.imshow(corr_real, text_auto=True, height=600,
                          color_continuous_scale='RdBu_r', range_color=[-1,1])
    st.plotly_chart(fig5_real, use_container_width=True)
    
    st.divider()
    
    # Synthetic Map (Full Width for visibility)
    st.markdown("### 🤖 Synthetic Pattern Correlation")
    st.info("Showing the generalized patterns learned by the augmented model.")
    corr_synth = synth_df[numeric_cols].corr()
    fig5_synth = px.imshow(corr_synth, text_auto=True, height=600,
                           color_continuous_scale='RdBu_r', range_color=[-1,1])
    st.plotly_chart(fig5_synth, use_container_width=True)

st.divider()

# --- Methodology & References ---
with st.expander("📚 Data Grounding & Journal References"):
    st.write("""
    Every data point used to train this model is grounded in peer-reviewed research for Slaughterhouse and Shrimp waste:
    - **SBWW (Slaughterhouse):** wasabisys/ResearchGate (up to 98% COD removal)
    - **SIWW (Shrimp):** inarah.co.id/IRJET (Voltages up to 0.96V)
    - **Methodology**: 11+ Real Experimental Runs + 2000 Grounded Synthetic Samples using ML ensembles.
    """)
    st.markdown("---")
    st.markdown("### [1] Microbial Fuel Cell for Slaughterhouse Wastewater ([ResearchGate Link](https://www.researchgate.net/publication/322158864))")
    st.markdown("### [2] Electricity generation from slaughterhouse wastewater ([IntechOpen Link](https://www.intechopen.com/chapters/71343))")
    st.markdown("### [3] Bio-electricity generation from shrimp wastewater ([inarah.co.id Link](https://www.researchgate.net/publication/344158864))")
    st.markdown("### [4] Seafood processing wastewater treatment in dual chambered MFC ([IRJET Link](https://www.irjet.net/archives/V4/i12/IRJET-V4I12285.pdf))")
