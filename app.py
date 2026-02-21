import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from preprocessing import preprocess_data, get_features_and_target, get_unique_values
from model import train_and_get_best_model, predict_yield
from recommendation import (
    recommend_crop, 
    get_seasonal_analysis, 
    get_top_crops_by_season
)

st.set_page_config(
    page_title="Smart Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 50%, #A5D6A7 100%);
    }
    
    .stApp p, .stApp label {
        color: #1a1a1a;
    }
    
    .stMarkdown p {
        color: #1a1a1a;
    }
    
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1B5E20, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-title {
        font-family: 'Poppins', sans-serif;
        text-align: center;
        color: #444 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2E7D32 !important;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #4CAF50;
        display: inline-block;
    }
    
    .custom-card {
        background: linear-gradient(135deg, #ffffff 0%, #F1F8E9 100%);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.12);
        border: 1px solid rgba(76, 175, 80, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #E8F5E9 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.15);
        border-left: 5px solid #4CAF50;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.25);
    }
    
    .metric-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-family: 'Poppins', sans-serif;
        font-size: 0.9rem;
        color: #333 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .result-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 16px;
        padding: 2rem;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.2);
        margin: 1.5rem 0;
    }
    
    .result-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #1B5E20;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E7D32;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(33, 150, 243, 0.15);
        color: #1a1a1a !important;
    }
    
    .info-box *, .info-box p, .info-box strong, .info-box td {
        color: #1a1a1a !important;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(76, 175, 80, 0.15);
        color: #1a1a1a !important;
    }
    
    .success-box *, .success-box p, .success-box strong {
        color: #1a1a1a !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 50%, #388E3C 100%);
    }
    
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255, 255, 255, 0.15);
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
        background: linear-gradient(135deg, #66BB6A 0%, #388E3C 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #E0E0E0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2) !important;
    }
    
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #4CAF50, transparent);
        margin: 2rem 0;
        border: none;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-item {
        background: linear-gradient(135deg, #ffffff 0%, #F1F8E9 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.12);
        transition: all 0.3s ease;
        border-top: 4px solid #4CAF50;
    }
    
    .feature-item:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-family: 'Poppins', sans-serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #333 !important;
    }
    
    .stats-bar {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: space-around !important;
        align-items: center !important;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
        width: 100%;
    }
    
    .stat-item {
        text-align: center;
        color: white !important;
        flex: 1;
        padding: 0.5rem;
    }
    
    .stat-item *, .stats-bar * {
        color: #ffffff !important;
    }
    
    .stat-number {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #ffffff !important;
    }
    
    .rec-card {
        background: linear-gradient(135deg, #ffffff 0%, #E8F5E9 100%);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 8px 30px rgba(76, 175, 80, 0.2);
        border: 2px solid #4CAF50;
        position: relative;
        overflow: hidden;
    }
    
    .rec-card-label {
        font-size: 0.8rem;
        color: #555 !important;
    }
    
    .rec-card-value-green {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4CAF50 !important;
    }
    
    .rec-card-value-blue {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2196F3 !important;
    }
    
    .rec-card-title {
        font-size: 1rem;
        color: #555 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .rec-crop {
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1B5E20 !important;
        margin: 1rem 0;
    }
    
    .rec-badge {
        background: #4CAF50;
        color: white !important;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 1rem;
    }
    
    .rec-badge * {
        color: white !important;
    }
    
    .footer {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 3rem;
    }
    
    .footer *, .footer p, .footer-text {
        color: white !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.6s ease forwards;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #ffffff 0%, #E8F5E9 100%);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(46, 125, 50, 0.3);
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: #1a1a1a !important;
        background: rgba(255, 255, 255, 0.8);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: white !important;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fdf8 0%, #e8f5e9 100%);
        border-radius: 12px;
        font-weight: 600;
        color: #1a1a1a !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #1a1a1a !important;
        font-weight: 500 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #2E7D32 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_preprocess_data():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'crop_data.csv')
    df_encoded, original_df, crop_encoder, season_encoder = preprocess_data(data_path)
    X, y = get_features_and_target(df_encoded)
    unique_values = get_unique_values(original_df)
    return df_encoded, original_df, crop_encoder, season_encoder, X, y, unique_values


@st.cache_resource
def train_models(_X, _y):
    best_model, best_model_name, results_df, all_models = train_and_get_best_model(_X, _y, perform_cv=False)
    return best_model, best_model_name, results_df, all_models


def render_metric_card(icon, value, label, color="#4CAF50"):
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def render_stats_bar(stats):
    items_html = ""
    for stat in stats:
        items_html += f'<div class="stat-item" style="flex:1;text-align:center;"><div class="stat-number" style="color:#ffffff !important;font-size:2rem;font-weight:700;">{stat["value"]}</div><div class="stat-label" style="color:#ffffff !important;font-size:0.85rem;text-transform:uppercase;">{stat["label"]}</div></div>'
    return f'<div class="stats-bar" style="display:flex;flex-direction:row;justify-content:space-around;align-items:center;background:linear-gradient(135deg,#2E7D32 0%,#4CAF50 100%);border-radius:16px;padding:1.5rem 2rem;margin:2rem 0;">{items_html}</div>'


def render_feature_card(icon, title, description):
    return f"""
    <div class="feature-item">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    """


def main():
    st.markdown('<h1 class="main-header">🌾 AI-Based Smart Crop Yield Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict crop production • Analyze seasonal trends • Get smart recommendations</p>', unsafe_allow_html=True)
    
    try:
        df_encoded, original_df, crop_encoder, season_encoder, X, y, unique_values = load_and_preprocess_data()
        best_model, best_model_name, results_df, all_models = train_models(X, y)
    except Exception as e:
        st.error(f"Error loading data or training models: {str(e)}")
        st.stop()
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="margin: 0; color: #ffffff !important;">🌱 CropAI</h2>
            <p style="opacity: 0.9; font-size: 0.9rem; color: #ffffff !important;">Smart Farming Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🌱 Yield Prediction", "📊 Seasonal Analysis", "💡 Crop Recommendation", "📈 Data Overview"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #ffffff !important;">✨ Features</h4>
            <ul style="font-size: 0.85rem; padding-left: 1.2rem; margin: 0; color: #ffffff !important;">
                <li style="color: #ffffff !important;">ML-Powered Predictions</li>
                <li style="color: #ffffff !important;">Seasonal Insights</li>
                <li style="color: #ffffff !important;">Smart Recommendations</li>
                <li style="color: #ffffff !important;">Data Visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 1rem;">
            <h4 style="margin: 0 0 0.5rem 0; color: #ffffff !important;">🤖 Active Model</h4>
            <p style="font-size: 1rem; font-weight: 600; margin: 0; color: #ffffff !important;">{best_model_name}</p>
            <p style="font-size: 0.8rem; opacity: 0.9; margin: 0.3rem 0 0 0; color: #ffffff !important;">Auto-selected for best accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "🌱 Yield Prediction":
        st.markdown('<div class="section-header">🌱 Crop Yield Prediction</div>', unsafe_allow_html=True)
        
        best_r2 = results_df[results_df['Model'] == best_model_name]['R² Score'].values[0]
        stats = [
            {"value": f"{best_r2*100:.1f}%", "label": "Model Accuracy"},
            {"value": str(len(all_models)), "label": "ML Models"},
            {"value": str(len(unique_values['crops'])), "label": "Crops Supported"},
            {"value": str(len(unique_values['seasons'])), "label": "Seasons"}
        ]
        st.markdown(render_stats_bar(stats), unsafe_allow_html=True)
        
        with st.expander("📊 View Model Performance Comparison", expanded=False):
            styled_results = results_df.style.highlight_max(
                subset=['R² Score'], color='#C8E6C9'
            ).highlight_min(
                subset=['MAE', 'RMSE'], color='#C8E6C9'
            ).format({
                'R² Score': '{:.4f}',
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}'
            })
            st.dataframe(styled_results, use_container_width=True, hide_index=True)
        
        st.markdown("### 📝 Enter Crop Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            crop = st.selectbox("🌾 Select Crop", unique_values['crops'])
            rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=3000.0, value=1000.0, step=10.0)
            area = st.number_input("📐 Area (hectares)", min_value=0.0, max_value=100000.0, value=100.0, step=10.0)
        
        with col2:
            season = st.selectbox("🗓️ Select Season", unique_values['seasons'])
            temperature = st.number_input("🌡️ Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 Predict Yield", use_container_width=True)
        
        if predict_button:
            with st.spinner("Calculating prediction..."):
                prediction = predict_yield(
                    best_model, crop_encoder, season_encoder,
                    crop, season, rainfall, temperature, area
                )
                
                st.markdown(f"""
                <div class="result-box animate-fade-in">
                    <div class="result-title">📊 Predicted Production</div>
                    <div class="result-value">{prediction:,.2f} tons</div>
                    <p style="margin-top: 1rem; color: #333;">
                        For <strong>{area:,.0f} hectares</strong> of <strong>{crop}</strong> 
                        during <strong>{season}</strong> season
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown(render_metric_card("🌧️", f"{rainfall:.0f} mm", "Rainfall"), unsafe_allow_html=True)
                with col_m2:
                    st.markdown(render_metric_card("🌡️", f"{temperature:.1f}°C", "Temperature", "#FF9800"), unsafe_allow_html=True)
                with col_m3:
                    yield_per_hectare = prediction / area if area > 0 else 0
                    st.markdown(render_metric_card("📈", f"{yield_per_hectare:.2f}", "Tons/Hectare", "#2196F3"), unsafe_allow_html=True)
    
    elif page == "📊 Seasonal Analysis":
        st.markdown('<div class="section-header">📊 Seasonal Analysis</div>', unsafe_allow_html=True)
        
        seasonal_data = get_seasonal_analysis(original_df)
        
        tab1, tab2 = st.tabs(["📊 Production by Season", "🌧️ Rainfall Analysis"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#2E7D32', '#4CAF50', '#81C784']
                bars = ax.bar(seasonal_data['Season'], seasonal_data['Average Production'], color=colors, edgecolor='white', linewidth=2)
                ax.set_xlabel('Season', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average Production (tons)', fontsize=12, fontweight='bold')
                ax.set_title('Average Crop Production by Season', fontsize=14, fontweight='bold', pad=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_facecolor('#f8fdf8')
                fig.patch.set_facecolor('#f8fdf8')
                for bar, val in zip(bars, seasonal_data['Average Production']):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                           f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📈 Seasonal Statistics")
                for _, row in seasonal_data.iterrows():
                    st.markdown(f"""
                    <div class="custom-card">
                        <h4 style="color: #2E7D32; margin: 0;">{row['Season']}</h4>
                        <p style="font-size: 1.5rem; font-weight: 700; color: #1B5E20; margin: 0.5rem 0;">
                            {row['Average Production']:,.0f} tons
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(original_df['Rainfall'], original_df['Production'], 
                               c=original_df['Temperature'], cmap='RdYlGn_r', 
                               alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
            ax.set_xlabel('Rainfall (mm)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Production (tons)', fontsize=12, fontweight='bold')
            ax.set_title('Rainfall vs Production (colored by Temperature)', fontsize=14, fontweight='bold', pad=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_facecolor('#f8fdf8')
            fig.patch.set_facecolor('#f8fdf8')
            cbar = plt.colorbar(scatter)
            cbar.set_label('Temperature (°C)', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif page == "💡 Crop Recommendation":
        st.markdown('<div class="section-header">💡 Smart Crop Recommendation</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>ℹ️ How it works:</strong> Enter your field conditions below, and our AI will recommend the best crop 
            based on historical data and similarity matching with optimal growing conditions.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_rainfall = st.number_input("🌧️ Expected Rainfall (mm)", min_value=0.0, max_value=3000.0, value=1000.0, step=10.0, key="rec_rainfall")
        with col2:
            rec_temperature = st.number_input("🌡️ Average Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.5, key="rec_temp")
        with col3:
            rec_season = st.selectbox("🗓️ Season", unique_values['seasons'], key="rec_season")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            recommend_button = st.button("🎯 Get Recommendation", use_container_width=True)
        
        if recommend_button:
            with st.spinner("Analyzing conditions..."):
                recommendation = recommend_crop(original_df, rec_rainfall, rec_temperature, rec_season)
                
                if recommendation['recommended_crop']:
                    st.markdown(f"""
                    <div class="rec-card animate-fade-in">
                        <div class="rec-card-title">🏆 Recommended Crop</div>
                        <div class="rec-crop">🌾 {recommendation['recommended_crop']}</div>
                        <div class="rec-badge">✓ Best Match for Your Conditions</div>
                        <div style="display: flex; justify-content: space-around; margin-top: 1.5rem; flex-wrap: wrap;">
                            <div style="text-align: center; padding: 0.5rem;">
                                <div class="rec-card-label">Confidence</div>
                                <div class="rec-card-value-green">{recommendation['confidence_score']:.1f}%</div>
                            </div>
                            <div style="text-align: center; padding: 0.5rem;">
                                <div class="rec-card-label">Expected Production</div>
                                <div class="rec-card-value-blue">{recommendation['expected_production']:,.0f} tons</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 All Recommendations (Ranked)")
                    
                    for i, rec in enumerate(recommendation['all_recommendations'][:5], 1):
                        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📍"
                        match_icons = ""
                        if rec['rainfall_match']:
                            match_icons += "🌧️ "
                        if rec['temp_match']:
                            match_icons += "🌡️"
                        
                        st.markdown(f"""
                        <div class="custom-card" style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.5rem;">{medal}</span>
                                <strong style="font-size: 1.2rem; color: #2E7D32;">{rec['crop']}</strong>
                                <span style="margin-left: 0.5rem;">{match_icons}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="color: #4CAF50; font-weight: 700;">{rec['confidence']:.1f}% match</div>
                                <div style="color: #666; font-size: 0.85rem;">{rec['expected_production']:,.0f} tons expected</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning(recommendation['message'])
    
    elif page == "📈 Data Overview":
        st.markdown('<div class="section-header">📈 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(render_metric_card("📊", str(len(original_df)), "Total Records"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card("🌾", str(len(unique_values['crops'])), "Crop Types"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("🗓️", str(len(unique_values['seasons'])), "Seasons"), unsafe_allow_html=True)
        with col4:
            st.markdown(render_metric_card("📐", "5", "Features", "#2196F3"), unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📋 Dataset", "📊 Statistics", "🏆 Top Performers"])
        
        with tab1:
            st.dataframe(original_df, use_container_width=True, hide_index=True, height=400)
        
        with tab2:
            st.markdown("### 📊 Numerical Summary")
            st.dataframe(original_df.describe().round(2), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🌾 Crops in Dataset")
                for crop in unique_values['crops']:
                    count = len(original_df[original_df['Crop'] == crop])
                    st.markdown(f"- **{crop}**: {count} records")
            with col2:
                st.markdown("### 🗓️ Seasons in Dataset")
                for season in unique_values['seasons']:
                    count = len(original_df[original_df['Season'] == season])
                    st.markdown(f"- **{season}**: {count} records")
        
        with tab3:
            selected_season = st.selectbox("Select Season for Top Crops", unique_values['seasons'])
            top_crops = get_top_crops_by_season(original_df, selected_season, top_n=5)
            
            st.markdown(f"### 🏆 Top 5 Crops for {selected_season} Season")
            
            for i, (_, row) in enumerate(top_crops.iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"#{i}"
                st.markdown(f"""
                <div class="custom-card" style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.5rem;">{medal}</span>
                        <strong style="font-size: 1.1rem; color: #2E7D32;">{row['Crop']}</strong>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #4CAF50; font-weight: 700;">{row['Avg Production']:,.0f} tons</div>
                        <div style="color: #666; font-size: 0.85rem;">
                            🌧️ {row['Avg Rainfall']:.0f}mm | 🌡️ {row['Avg Temperature']:.1f}°C
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        <p class="footer-text" style="margin: 0; font-size: 1rem;">
            🌾 AI-Based Smart Crop Yield Prediction System
        </p>
        <p class="footer-text" style="margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.9;">
            Built with Streamlit, Scikit-learn & XGBoost
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
