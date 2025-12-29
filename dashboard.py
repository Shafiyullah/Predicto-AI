import streamlit as st

# --- PAGE CONFIGURATION (Must be first) ---
st.set_page_config(
    page_title="Predicto AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

import time

def load_dependencies():
    status_container = st.empty()
    try:
        status_container.info("Loading Graphics Engine (1/4)...")
        import matplotlib
        matplotlib.use('Agg')
        import plotly.express as px
        import plotly.graph_objects as go
        
        status_container.info("Loading Data Engine (2/4)...")
        import pandas as pd
        
        status_container.info("Loading AI Core (3/4)...")
        try:
            from analyzer.data_analyzer import DataAnalyzer, XGBOOST_AVAILABLE, PROPHET_AVAILABLE, RL_AVAILABLE
        except ImportError as e:
            st.error(f"AI Core Load Failed: {e}")
            st.stop()
        
        status_container.success("✅ Dashboard Ready!")
        time.sleep(0.5)
        status_container.empty()
        
        return pd, px, go, DataAnalyzer, XGBOOST_AVAILABLE, PROPHET_AVAILABLE, RL_AVAILABLE
    except Exception as e:
        st.error(f"❌ Critical Dependency Error: {e}")
        st.stop()

# Load dependencies
pd, px, go, DataAnalyzer, XGBOOST_AVAILABLE, PROPHET_AVAILABLE, RL_AVAILABLE = load_dependencies()

# --- CUSTOM CSS & STYLING (Modern Dark Theme) ---
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Force Dark Theme Backgrounds */
    .stApp {
        background-color: #0e1117; /* Very Dark Blue-Grey */
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric Cards (Glassmorphism) */
    div.metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    div.metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
    }
    div.metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #22d3ee; /* Cyan */
    }
    div.metric-label {
        font-size: 14px;
        color: #94a3b8;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.5);
        color: white;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363d;
        border-radius: 12px;
        background-color: #0d1117;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE (Lazy Loading to prevent timeout) ---
@st.cache_resource
def get_analyzer():
    return DataAnalyzer()

try:
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = get_analyzer()
    analyzer = st.session_state.analyzer
except Exception as e:
    st.error(f"Critical Error Initializing Engine: {e}")
    st.stop()

def display_metric_card(col, label, value, help_text=None):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("⚡ Predicto AI")
        st.markdown("*Professional ML Platform*")
        st.markdown("---")
        
        page = st.radio("Navigation", 
            ["📂 Data Upload", "📊 Data Analysis", "🧠 Model Training", "📈 Forecasting", "🎮 Reinforcement Learning"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption("SYSTEM STATUS")
        
        status_col1, status_col2 = st.columns([1, 4])
        with status_col1: st.write("✅" if XGBOOST_AVAILABLE else "❌")
        with status_col2: st.write("XGBoost Engine")
        
        with status_col1: st.write("✅" if PROPHET_AVAILABLE else "❌")
        with status_col2: st.write("Prophet Forecaster")
        
        with status_col1: st.write("✅" if RL_AVAILABLE else "❌")
        with status_col2: st.write("RL Agent")

    # --- MAIN CONTENT ---
    if page == "📂 Data Upload":
        st.title("Data Integration")
        st.markdown("Upload your dataset to begin the analysis pipeline. Supported formats: CSV, Excel, JSON.")
        
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'json'])
        
        if uploaded_file:
            # Check if this file is already loaded
            if 'loaded_file_name' not in st.session_state or st.session_state.loaded_file_name != uploaded_file.name:
                with st.spinner("Processing data..."):
                    with open("temp_upload.csv", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    try:
                        # Removed specific 'pyarrow' engine request in dashboard to be safe, 
                        # relying on Data_Analyzer's internal logic or pandas default
                        analyzer.load_data("temp_upload.csv")
                        st.session_state.loaded_file_name = uploaded_file.name
                        st.toast("Data loaded successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"Failed to load file: {e}")
            else:
                 # Data already loaded, just show success state silently or permanent success message
                 st.info(f"Using loaded file: **{uploaded_file.name}**")

            # Quick Stats (Always show if data exists)
            if analyzer.df is not None:
                st.markdown("### 📋 Dataset Overview")
                m1, m2, m3 = st.columns(3)
                display_metric_card(m1, "Total Rows", f"{analyzer.df.shape[0]:,}")
                display_metric_card(m2, "Total Columns", f"{analyzer.df.shape[1]}")
                display_metric_card(m3, "Memory Usage", f"{analyzer.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔎 View Raw Data", expanded=True):
                    st.dataframe(analyzer.df.head(10), use_container_width=True)

    # ... (Data Analysis and Model Training sections remain largely the same) ...
    elif page == "📊 Data Analysis":
        st.title("Exploratory Data Analysis")
        
        if analyzer.df is None:
            st.info("👋 Please upload a dataset in the **Data Upload** section to start.")
        else:
            options = analyzer.get_analysis_options()
            
            # Smart Layout with Tabs
            tab1, tab2, tab3 = st.tabs(["📝 Feature Overview", "📈 Visualizations", "🔍 Correlations"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### Numeric Features")
                    st.dataframe(pd.DataFrame(options['numeric_columns'], columns=["Feature Name"]), use_container_width=True)
                with c2:
                    st.markdown("### Categorical Features")
                    st.dataframe(pd.DataFrame(options['categorical_columns'], columns=["Feature Name"]), use_container_width=True)

            with tab2:
                col_ctrl, col_plot = st.columns([1, 3])
                
                with col_ctrl:
                    st.markdown("### Plot Controls")
                    plot_type = st.selectbox("Chart Type", ["Histogram", "Scatter Plot", "Box Plot"])
                    
                    if plot_type == "Histogram":
                        x_col = st.selectbox("X Axis", options['numeric_columns'])
                    elif plot_type == "Scatter Plot":
                        x_col = st.selectbox("X Axis", options['numeric_columns'])
                        y_col = st.selectbox("Y Axis", options['numeric_columns'])
                        color_col = st.selectbox("Color By (Optional)", [None] + options['categorical_columns'])
                    elif plot_type == "Box Plot":
                        y_col = st.selectbox("Value Column", options['numeric_columns'])
                        x_col = st.selectbox("Group By", options['categorical_columns'])
                
                with col_plot:
                    if plot_type == "Histogram":
                        fig = px.histogram(analyzer.df, x=x_col, title=f"Distribution of {x_col}", template="plotly_white")
                    elif plot_type == "Scatter Plot":
                        fig = px.scatter(analyzer.df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}", template="plotly_white")
                    elif plot_type == "Box Plot":
                        fig = px.box(analyzer.df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_white")
                    
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if len(options['numeric_columns']) > 1:
                    corr = analyzer.df[options['numeric_columns']].corr()
                    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Feature Correlation Matrix", template="plotly_white")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for correlation analysis.")

    elif page == "🧠 Model Training":
        st.title("Predictive Modeling")
        
        if analyzer.df is None:
            st.info("👋 Upload data to train models.")
        else:
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("### Configuration")
                target_col = st.selectbox("🎯 Target Variable", analyzer.df.columns)
                model_type_hint = "Regression" if pd.api.types.is_numeric_dtype(analyzer.df[target_col]) else "Classification"
                st.info(f"Detected Task: **{model_type_hint}**")
                
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner("Training models (AutoML with XGBoost)..."):
                        try:
                            # Capture logs
                            import io
                            from contextlib import redirect_stdout
                            f = io.StringIO()
                            with redirect_stdout(f):
                                analyzer.train_model(target_col)
                            output = f.getvalue()
                            
                            st.session_state.training_logs = output
                            st.session_state.training_success = True
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            st.session_state.training_success = False

            with c2:
                if st.session_state.get('training_success'):
                    st.success("Training completed successfully!")
                    
                    with st.expander("📄 View Training Logs", expanded=False):
                        st.code(st.session_state.get('training_logs'))
                    
                    if analyzer.model_pipeline:
                        st.markdown("### 🎉 Best Model Selected")
                        model_name = analyzer.model_pipeline.named_steps['model'].__class__.__name__
                        st.markdown(f"**{model_name}**")
                        
                        st.info("The model is now ready for predictions or saving.")

    # --- PAGE: Forecasting ---
    elif page == "📈 Forecasting":
        st.title("Time-Series Forecasting")
        st.markdown("Powered by **Facebook Prophet**")
        
        if not PROPHET_AVAILABLE:
            st.error("Prophet is not installed. Please check your environment.")
        elif analyzer.df is None:
            st.info("👋 Upload data containing a date column.")
        else:
            col_conf, col_viz = st.columns([1, 3])
            
            with col_conf:
                st.markdown("### Setup")
                date_col = st.selectbox("📅 Date Column", analyzer.df.columns)
                target_col = st.selectbox("🎯 Value to Forecast", analyzer.get_analysis_options()['numeric_columns'])
                periods = st.slider("Horizon (Days)", 7, 365, 30)
                
                if st.button("✨ Generate Forecast", type="primary"):
                    with st.spinner("Running Prophet Model..."):
                        try:
                            results = analyzer.run_forecasting(target_col, date_col, periods)
                            st.session_state.forecast_results = results
                        except Exception as e:
                            st.error(f"Forecasting failed: {e}")

            with col_viz:
                if 'forecast_results' in st.session_state:
                    res = st.session_state.forecast_results
                    st.success("Forecast generated!")
                    
                    # PROPHET INTERACTIVE PLOTS (Plotly)
                    from prophet.plot import plot_plotly, plot_components_plotly
                    
                    st.markdown("### 📊 Interactive Forecast Plot")
                    try:
                        fig1 = plot_plotly(res['model'], res['forecast'])
                        st.plotly_chart(fig1, use_container_width=True)
                    except Exception:
                        st.warning("Could not generate interactive plot. Showing static plot.")
                        st.pyplot(res['figures'][0], use_container_width=True)
                    
                    with st.expander("🧩 View Components"):
                        try:
                            fig2 = plot_components_plotly(res['model'], res['forecast'])
                            st.plotly_chart(fig2, use_container_width=True)
                        except:
                            st.pyplot(res['figures'][1], use_container_width=True)
                    
                    with st.expander("🔢 View Forecast Data"):
                        st.dataframe(res['forecast_tail'], use_container_width=True)

    elif page == "🎮 Reinforcement Learning":
        st.title("Deep Reinforcement Learning")
        st.markdown("Train a **PPO Agent** to predict trends.")
        
        if not RL_AVAILABLE:
            st.error("Stable-Baselines3 is not installed.")
        elif analyzer.df is None:
            st.info("👋 Upload data first.")
        else:
            st.markdown("### 🤖 Agent Configuration")
            target = st.selectbox("Target Trend", analyzer.get_analysis_options()['numeric_columns'])
            steps = st.number_input("Training Timesteps", 1000, 100000, 10000)
            
            if st.button("🥋 Train Agent", type="primary"):
                with st.spinner("Training PPO Agent..."):
                     try:
                        import io
                        from contextlib import redirect_stdout
                        f = io.StringIO()
                        with redirect_stdout(f):
                            analyzer.train_rl_agent(target, total_timesteps=steps)
                        
                        st.success("Agent successfully trained!")
                        st.text_area("Training Output", f.getvalue(), height=200)
                     except Exception as e:
                        st.error(f"Training failed: {e}")

if __name__ == "__main__":
    main()
