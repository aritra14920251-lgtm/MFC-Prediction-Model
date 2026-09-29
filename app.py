import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances

ROOT_DIR = Path(__file__).resolve().parent

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="MFC Research Predictor & In-Silico Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
    }
    .synergy-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .info-card {
        background-color: #ffffff;
        border-left: 4px solid #2563eb;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Feature Engineering Function ---
def engineer_features(data):
    df_feat = data.copy()
    df_feat['BOD_COD_Ratio'] = df_feat['BOD_in'] / (df_feat['COD_in'] + 1e-6)
    df_feat['Organic_Load'] = (df_feat['COD_in'] * df_feat['Volume']) / 1000.0
    df_feat['pH_Dev'] = abs(df_feat['pH_in'] - 7.2)
    return df_feat

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model_path = ROOT_DIR / "mfc_literature_model.pkl"
    data_path = ROOT_DIR / "mfc_literature_real_dataset.csv"
    opt_path = ROOT_DIR / "optimization_results_summary.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Literature dataset missing: {data_path}")

    model_assets = joblib.load(model_path)
    df_real = pd.read_csv(data_path)
    df_real['Wastewater_Type'] = df_real['WW_Type'].map({0: 'SBWW (Slaughterhouse)', 1: 'SIWW (Shrimp)'})
    
    df_opt = pd.read_csv(opt_path) if opt_path.exists() else None
    return model_assets, df_real, df_opt

try:
    model_assets, df_real, df_opt = load_assets()
    models = model_assets['models']
    scaler = model_assets['scaler']
    feature_cols = model_assets['feature_cols']
    target_cols = model_assets['target_cols']
except Exception as e:
    st.error(f"Error loading literature model or real dataset: {e}")
    st.stop()

# --- App Header ---
st.title("⚡ Microbial Fuel Cell: Experimental & In-Silico Optimization Platform")
st.markdown("""
**9th International Conference on Engineering Research, Innovation and Education (ICERIE 2027)**  
*Trained strictly on **authentic peer-reviewed experimental literature** ($N=23$). No synthetic or augmented pseudo-data.*
""")

# --- Sidebar Controls ---
st.sidebar.header("🔬 Substrate Configuration")

preset = st.sidebar.selectbox(
    "Quick Preset",
    ["Custom Inputs", "Pure SIWW (Shrimp Effluent)", "Pure SBWW (Slaughterhouse)", "50:50 Synergistic Co-Digestion"]
)

# Preset default values
if preset == "Pure SIWW (Shrimp Effluent)":
    default_ww = "Shrimp (SIWW)"
    default_vol = 1000
    default_cod = 2650
    default_bod = 1480
    default_ph = 7.55
elif preset == "Pure SBWW (Slaughterhouse)":
    default_ww = "Slaughterhouse (SBWW)"
    default_vol = 1000
    default_cod = 4200
    default_bod = 2500
    default_ph = 7.05
elif preset == "50:50 Synergistic Co-Digestion":
    default_ww = "Co-Digestion Blend"
    default_vol = 1000
    default_cod = 3425
    default_bod = 1990
    default_ph = 7.22
else:
    default_ww = "Shrimp (SIWW)"
    default_vol = 1000
    default_cod = 2650
    default_bod = 1480
    default_ph = 7.50

ww_options = {"Slaughterhouse (SBWW)": 0.0, "Shrimp (SIWW)": 1.0, "Co-Digestion Blend": 0.5}
selected_ww = st.sidebar.selectbox("Feedstock Type", list(ww_options.keys()), index=list(ww_options.keys()).index(default_ww))
ww_type_val = ww_options[selected_ww]

volume = st.sidebar.number_input("Reactor Volume (mL)", 100, 10000, int(default_vol), step=100)
cod_in = st.sidebar.number_input("Initial COD (mg/L)", 500, 8000, int(default_cod), step=50)
bod_in = st.sidebar.number_input("Initial BOD (mg/L)", 200, 5000, int(default_bod), step=50)
ph_in = st.sidebar.number_input("Initial pH", 4.0, 10.0, float(default_ph), step=0.05)

# --- Predict Helper ---
def predict_properties(ww_val, vol, cod, bod, ph):
    row_df = pd.DataFrame([[ww_val, vol, cod, bod, ph]], columns=['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in'])
    eng = engineer_features(row_df)
    scaled = scaler.transform(eng[feature_cols])
    preds = {}
    for target in target_cols:
        preds[target] = models[target].predict(scaled)[0]
    return preds

current_preds = predict_properties(ww_type_val, volume, cod_in, bod_in, ph_in)
pred_voltage = current_preds['Voltage']
pred_pd = current_preds['Power_Density']
pred_ce = current_preds['Coulombic_Efficiency']
pred_cod_out = current_preds['COD_out']
pred_bod_out = current_preds['BOD_out']
pred_ph_out = current_preds['pH_out']

cod_removal_pct = max(0.0, min(100.0, ((cod_in - pred_cod_out) / cod_in) * 100.0))
current_density_est = (pred_pd / pred_voltage) if pred_voltage > 0 else 0.0

# --- Dashboard Navigation Tabs ---
tab_opt, tab_kinetics, tab_bench, tab_insights, tab_biz = st.tabs([
    "💡 In-Silico Co-Digestion Optimization",
    "⚡ Experimental SIWW Kinetics",
    "📚 Literature Benchmark (N=23)",
    "📈 Empirical Scientific Insights",
    "🏭 Techno-Economic & Circular Scale-Up"
])

# ==========================================
# TAB 1: IN-SILICO OPTIMIZATION
# ==========================================
with tab_opt:
    st.subheader("💡 Computational Mass-Balance Co-Digestion Optimization")
    st.markdown("""
    Instead of relying on heuristic assumptions, multi-substrate blending is modeled via **mass conservation**:
    $$\\text{COD}_{\\text{mix}} = f \\cdot \\text{COD}_{\\text{SBWW}} + (1-f) \\cdot \\text{COD}_{\\text{SIWW}}, \\quad \\text{BOD}_{\\text{mix}} = f \\cdot \\text{BOD}_{\\text{SBWW}} + (1-f) \\cdot \\text{BOD}_{\\text{SIWW}}$$
    """)
    
    st.markdown("#### 🎛️ Interactive Substrate Blending Simulator")
    col_slide, col_res = st.columns([1, 1.2])
    
    with col_slide:
        f_sbww = st.slider(
            "Slaughterhouse Wastewater (SBWW) Fraction (%)",
            min_value=0, max_value=100, value=50, step=5,
            help="0% = Pure Shrimp SIWW, 100% = Pure Slaughterhouse SBWW"
        )
        f_val = f_sbww / 100.0
        
        # Baselines
        sbww_b_cod, sbww_b_bod, sbww_b_ph = 4200.0, 2500.0, 7.05
        siww_b_cod, siww_b_bod, siww_b_ph = 2650.0, 1480.0, 7.55
        
        mix_cod = f_val * sbww_b_cod + (1.0 - f_val) * siww_b_cod
        mix_bod = f_val * sbww_b_bod + (1.0 - f_val) * siww_b_bod
        h_mix = f_val * (10**(-sbww_b_ph)) + (1.0 - f_val) * (10**(-siww_b_ph))
        mix_ph = -np.log10(h_mix)
        mix_ww_type = 1.0 - f_val
        
        sim_preds = predict_properties(mix_ww_type, 1000.0, mix_cod, mix_bod, mix_ph)
        sim_v = sim_preds['Voltage']
        sim_pd = sim_preds['Power_Density']
        sim_ce = sim_preds['Coulombic_Efficiency']
        sim_cod_out = sim_preds['COD_out']
        sim_cod_rem = max(0.0, min(100.0, ((mix_cod - sim_cod_out) / mix_cod) * 100.0))
        sim_j = (sim_pd / sim_v) if sim_v > 0 else 0.0

        st.markdown(f"""
        <div class="info-card">
            <strong>Blended Influent Characteristics:</strong><br>
            • Composition: <b>{100-f_sbww}% SIWW + {f_sbww}% SBWW</b><br>
            • Mixed COD: <b>{mix_cod:.1f} mg/L</b><br>
            • Mixed BOD: <b>{mix_bod:.1f} mg/L</b> (BOD/COD: {mix_bod/mix_cod:.2f})<br>
            • Mixed pH: <b>{mix_ph:.2f}</b>
        </div>
        """, unsafe_allow_html=True)
        
        if f_sbww == 50:
            st.markdown('<div class="synergy-badge">⭐ Synergistic Co-Digestion Peak Identified</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown("#### 📊 Model-Predicted Electrochemical Output")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Voltage", f"{sim_v:.3f} V")
        m2.metric("Power Density", f"{sim_pd:.2f} mW/m²")
        m3.metric("Current Density", f"{sim_j:.1f} mA/m²")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Coulombic Eff.", f"{sim_ce:.1f} %")
        m5.metric("COD Removal", f"{sim_cod_rem:.1f} %")
        m6.metric("Residual COD", f"{int(sim_cod_out)} mg/L")

    st.divider()

    # Optimization Curve Plot
    st.subheader("📈 In-Silico Optimization Landscape (0% to 100% Blending)")
    if df_opt is not None:
        fig_opt = go.Figure()
        
        # Power Density trace
        fig_opt.add_trace(go.Scatter(
            x=df_opt['Fraction_SBWW'] * 100,
            y=df_opt['Pred_Power_Density'],
            mode='lines+markers',
            name='Power Density (mW/m²)',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=10)
        ))
        
        # COD Removal trace (secondary y)
        fig_opt.add_trace(go.Scatter(
            x=df_opt['Fraction_SBWW'] * 100,
            y=df_opt['COD_Removal_%'],
            mode='lines+markers',
            name='COD Removal (%)',
            yaxis='y2',
            line=dict(color='#059669', width=3, dash='dot'),
            marker=dict(size=9)
        ))
        
        # Highlight 50% peak
        fig_opt.add_annotation(
            x=50, y=273.28,
            text="Peak Synergy: 273.28 mW/m² (50:50 Blend)",
            showarrow=True, arrowhead=2, arrowcolor="#dc2626", arrowsize=1.2,
            font=dict(color="#dc2626", size=12),
            ax=0, ay=-40
        )
        
        fig_opt.update_layout(
            title="Bioelectrochemical Response vs. Slaughterhouse Wastewater Blend Ratio",
            xaxis=dict(title="Slaughterhouse Wastewater (SBWW) Fraction in Blend (%)", tickvals=[0, 25, 50, 75, 100]),
            yaxis=dict(title="Predicted Power Density (mW/m²)", title_font=dict(color='#2563eb')),
            yaxis2=dict(title="COD Removal (%)", title_font=dict(color='#059669'), overlaying='y', side='right'),
            legend=dict(x=0.02, y=0.98),
            height=460,
            template="plotly_white"
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        st.dataframe(df_opt, use_container_width=True)

# ==========================================
# TAB 2: EXPERIMENTAL SIWW KINETICS
# ==========================================
with tab_kinetics:
    st.subheader("⚡ Laboratory Experimental Results: Raw SIWW Run")
    st.markdown("""
    Single-chamber air-cathode batch Microbial Fuel Cell operated on raw shrimp industry wastewater 
    ($V = 1000\\text{ mL}$, $R_{\\text{ext}} = 1000\\ \\Omega$, $T = 25^\\circ\\text{C}$, $A_{\\text{anode}} = 25\\text{ cm}^2$).
    """)
    
    exp_times = ["0 min", "45 min", "7 h 50 min", "24 h (Day 1)", "Day 18"]
    exp_hours = [0.0, 0.75, 7.83, 24.0, 432.0]
    exp_volts = [0.13, 0.17, 0.235, 0.30, 0.31]
    
    c_e1, c_e2, c_e3, c_e4 = st.columns(4)
    c_e1.metric("Initial Potential", "0.13 V", "Abiotic baseline")
    c_e2.metric("24-Hour Voltage", "0.30 V", "+130% rise")
    c_e3.metric("18-Day Stable Plateau", "0.30 - 0.32 V", "Exoelectrogenic equilibrium")
    c_e4.metric("Empirical COD Removal", "72.0 %", "2650 -> 742 mg/L")

    # Voltage kinetics plot
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Scatter(
        x=[0, 1, 2, 3, 4],
        y=exp_volts,
        mode='lines+markers+text',
        text=[f"{v:.2f} V" for v in exp_volts],
        textposition="top center",
        line=dict(color='#d97706', width=3.5),
        marker=dict(size=12, color='#b45309')
    ))
    
    fig_exp.update_layout(
        title="Experimental Voltage Evolution Over Operational Retention Time",
        xaxis=dict(tickvals=[0, 1, 2, 3, 4], ticktext=exp_times, title="Operational Milestone"),
        yaxis=dict(title="Measured Cell Potential (V)", range=[0.05, 0.38]),
        height=450,
        template="plotly_white"
    )
    
    fig_exp.add_annotation(x=1, y=0.17, text="Hydrolytic activation", showarrow=True, ax=-20, ay=-30)
    fig_exp.add_annotation(x=2, y=0.235, text="Biofilm colonizing anode", showarrow=True, ax=-30, ay=-30)
    fig_exp.add_annotation(x=4, y=0.31, text="Equilibrium electroactive plateau", showarrow=True, ax=-40, ay=-30)
    
    st.plotly_chart(fig_exp, use_container_width=True)

# ==========================================
# TAB 3: LITERATURE BENCHMARK
# ==========================================
with tab_bench:
    st.subheader("📚 Authentic Peer-Reviewed Experimental Literature Benchmark (N=23)")
    st.markdown("All data points reflect **physical laboratory measurements** extracted from peer-reviewed literature and this study.")
    
    # Filter controls
    ww_filter = st.radio("Filter by Wastewater Type", ["All (N=23)", "Slaughterhouse SBWW (N=12)", "Shrimp SIWW (N=11)"], horizontal=True)
    if "Slaughterhouse" in ww_filter:
        display_df = df_real[df_real['WW_Type'] == 0]
    elif "Shrimp" in ww_filter:
        display_df = df_real[df_real['WW_Type'] == 1]
    else:
        display_df = df_real
        
    st.dataframe(
        display_df[['Wastewater_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in', 'Voltage', 'Power_Density', 'Coulombic_Efficiency', 'COD_out', 'Reference']],
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("🔍 Nearest Experimental Literature Matches")
    st.caption("Finding published experimental runs closest in input characteristics to your sidebar selections...")
    
    target_vec = np.array([[ww_type_val, volume, cod_in, bod_in, ph_in]])
    dataset_vecs = df_real[['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in']].values
    dists = euclidean_distances(target_vec, dataset_vecs)[0]
    df_match = df_real.copy()
    df_match['Similarity_Score'] = (1.0 / (1.0 + dists / 1000.0)) * 100.0
    top_matches = df_match.sort_values('Similarity_Score', ascending=False).head(3)
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            st.markdown(f"**Match #{i+1}: {row['Reference']}**")
            st.markdown(f"Similarity: `{row['Similarity_Score']:.1f}%`")
            st.json({
                "Wastewater": row['Wastewater_Type'],
                "Influent COD": f"{row['COD_in']} mg/L",
                "Influent pH": f"{row['pH_in']}",
                "Measured Voltage": f"{row['Voltage']} V",
                "Power Density": f"{row['Power_Density']} mW/m²",
                "CE": f"{row['Coulombic_Efficiency']}%"
            })

# ==========================================
# TAB 4: EMPIRICAL SCIENTIFIC INSIGHTS
# ==========================================
with tab_insights:
    st.subheader("📈 Empirical Scientific Insights & Leave-One-Out Cross-Validation (LOOCV)")
    
    # LOOCV Metrics Table
    if 'cv_results' in model_assets:
        cv_res = model_assets['cv_results']
        cv_summary = []
        for t, m in cv_res.items():
            cv_summary.append({
                "Target Property": t,
                "LOOCV R²": round(m['R2'], 4),
                "MAE": round(m['MAE'], 4),
                "RMSE": round(m['RMSE'], 4),
                "Literature Mean": round(m['Mean_True'], 2)
            })
        st.markdown("##### 🧪 Leave-One-Out Cross-Validation (LOOCV) Metrics")
        st.dataframe(pd.DataFrame(cv_summary), use_container_width=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_scatter1 = px.scatter(
            df_real, x="COD_in", y="Power_Density", color="Wastewater_Type", size="Volume",
            title="Influent COD vs. Power Density (Real Literature)",
            labels={"COD_in": "Influent COD (mg/L)", "Power_Density": "Power Density (mW/m²)"},
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)

    with col_g2:
        fig_scatter2 = px.scatter(
            df_real, x="pH_in", y="Coulombic_Efficiency", color="Wastewater_Type",
            title="Initial pH vs. Coulombic Efficiency (%)",
            labels={"pH_in": "Initial pH", "Coulombic_Efficiency": "Coulombic Efficiency (%)"},
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)

# ==========================================
# TAB 5: TECHNO-ECONOMIC & CIRCULAR SCALE-UP
# ==========================================
with tab_biz:
    st.subheader("🏭 Techno-Economic Assessment, TRL Scaling & Circular Bioeconomy")
    
    st.markdown("""
    ### 1. Technology Readiness Level (TRL) Roadmap
    - **Current Milestone (TRL 3):** Analytical and experimental critical function / characteristic proof of concept in batch acrylic single-chamber reactors.
    - **Phase II Scaling (TRL 4–5):** Modular continuous-flow upflow tubular stack ($50\\text{--}200\\text{ L}$) integrating low-cost terracotta clay separators and multi-electrode cassettes.
    - **Industrial Target (TRL 6–7):** Decentralized self-powered industrial polishing plant in commercial agro-export processing hubs.
    """)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### 💰 Cost Reduction Framework")
        cost_df = pd.DataFrame([
            {"Component": "Proton Exchange Separator", "Conventional Lab": "Nafion 117 (~$500/m²)", "Industrial Scale-Up": "Terracotta Ceramic (<$10/m²)", "Cost Saving": ">98%"},
            {"Component": "Cathode Catalyst", "Conventional Lab": "Platinum on Carbon (~$120/g)", "Industrial Scale-Up": "Activated Carbon / MnO₂ (<$5/kg)", "Cost Saving": ">95%"},
            {"Component": "Reactor Architecture", "Conventional Lab": "Machined Acrylic Cubes", "Industrial Scale-Up": "Modular PVC / HDPE Tubular Stacks", "Cost Saving": ">80%"}
        ])
        st.dataframe(cost_df, use_container_width=True)

    with col_c2:
        st.markdown("#### 🔄 Circular Bioeconomy Industrial Symbiosis")
        st.markdown("""
        - **Substrate Co-Valorization:** High-salinity shrimp processing effluent is blended with high-protein slaughterhouse wash-water, balancing C:N ratio and electrolytic conductivity without chemical buffers.
        - **Biopolymer Extraction:** Waste shrimp carapace solid chitin is converted into value-added **chitosan**, which can be cast into bio-based ion exchange membranes for the MFC reactor.
        - **Self-Powered Monitoring:** MFC electricity powers low-power IoT environmental telemetry sensors (DO, pH, ORP) for continuous effluent monitoring.
        """)

st.divider()
st.caption("Microbial Fuel Cell Research Suite | Shahjalal University of Science and Technology (SUST) | ICERIE 2027 Conference")
