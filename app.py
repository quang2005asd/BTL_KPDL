"""
Streamlit Dashboard - Dự báo nhu cầu năng lượng
Household Power Consumption Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('.')

# Page config
st.set_page_config(
    page_title="Phân tích tiêu thụ điện năng",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=60)  # Cache for 60 seconds only to allow fresh data
def load_data():
    """Load all analysis results"""
    data = {}
    tables_dir = Path('outputs/tables')
    
    try:
        data['association_rules'] = pd.read_csv(tables_dir / 'association_rules.csv')
    except:
        data['association_rules'] = None
    
    try:
        # Read CSV and handle duplicate headers properly
        df = pd.read_csv(tables_dir / 'cluster_profiles.csv', index_col=0)
        
        # Skip first 2 rows (duplicate multi-level headers)
        if len(df) >= 2:
            # Check if we have the header rows
            first_row = df.iloc[0].astype(str)
            if first_row.str.contains('mean|std|cluster', case=False).any():
                df = df.iloc[2:].copy()  # Skip both header rows
        
        # Convert index to int
        try:
            df.index = df.index.astype(int)
        except:
            pass
        
        # Convert all columns to float
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        data['cluster_profiles'] = df
    except Exception as e:
        print(f"Error loading cluster profiles: {e}")
        data['cluster_profiles'] = None
    
    try:
        data['forecast_comparison'] = pd.read_csv(tables_dir / 'forecast_model_comparison.csv', index_col=0)
    except:
        data['forecast_comparison'] = None
    
    try:
        data['anomaly_seasonal'] = pd.read_csv(tables_dir / 'seasonal_anomaly_analysis.csv', index_col=0)
    except:
        data['anomaly_seasonal'] = None
    
    try:
        data['anomaly_summary'] = pd.read_csv(tables_dir / 'anomaly_summary.csv', index_col=0)
    except:
        data['anomaly_summary'] = None
    
    return data

def display_metric_card(title, value, delta=None, help_text=None):
    """Display a beautiful metric card"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric(label=title, value=value, delta=delta, help=help_text)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">⚡ DỰ BÁO NHU CẦU NĂNG LƯỢNG</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d;">Phân tích tiêu thụ điện năng hộ gia đình - UCI Dataset (2006-2010)</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('⏳ Đang tải dữ liệu...'):
        data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/energy.png", width=100)
        st.title("📊 Menu Điều Hướng")
        
        page = st.radio(
            "Chọn phần phân tích:",
            [
                "🏠 Tổng Quan",
                "📈 Phân Tích EDA",
                "🔗 Association Mining",
                "🎯 Phân Cụm (Clustering)",
                "⚠️ Phát Hiện Bất Thường",
                "📉 Dự Báo (Forecasting)",
                "📋 Báo Cáo Tổng Hợp"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📚 Thông tin Dataset")
        st.info("""
        **Nguồn**: UCI Repository  
        **Thời gian**: 2006-2010  
        **Số lượng**: 2M+ records  
        **Tần suất**: Mỗi phút
        """)
        
        st.markdown("---")
        st.markdown("### 👥 Nhóm thực hiện")
        st.markdown("**GVHD**: Lê Thị Thùy Trang")
    
    # Main content
    if page == "🏠 Tổng Quan":
        show_overview(data)
    elif page == "📈 Phân Tích EDA":
        show_eda()
    elif page == "🔗 Association Mining":
        show_association(data)
    elif page == "🎯 Phân Cụm (Clustering)":
        show_clustering(data)
    elif page == "⚠️ Phát Hiện Bất Thường":
        show_anomaly(data)
    elif page == "📉 Dự Báo (Forecasting)":
        show_forecasting(data)
    elif page == "📋 Báo Cáo Tổng Hợp":
        show_report(data)

def show_overview(data):
    """Tổng quan dự án"""
    st.markdown('<h2 class="section-header">🏠 Tổng Quan Dự Án</h2>', unsafe_allow_html=True)
    
    # Key metrics
    st.markdown("### 📊 Các chỉ số chính")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if data['association_rules'] is not None:
            st.metric("Association Rules", f"{len(data['association_rules'])} rules", 
                     help="Số lượng luật kết hợp được phát hiện")
        else:
            st.metric("Association Rules", "N/A")
    
    with col2:
        if data['cluster_profiles'] is not None:
            st.metric("Clusters", f"{len(data['cluster_profiles'])} cụm",
                     help="Số cụm hộ gia đình được phân loại")
        else:
            st.metric("Clusters", "N/A")
    
    with col3:
        if data['anomaly_seasonal'] is not None:
            total_anomalies = data['anomaly_seasonal']['n_anomalies'].sum()
            st.metric("Anomalies", f"{total_anomalies} ngày",
                     help="Số ngày phát hiện bất thường")
        else:
            st.metric("Anomalies", "N/A")
    
    with col4:
        if data['forecast_comparison'] is not None:
            best_model = data['forecast_comparison'].index[0]
            best_mae = data['forecast_comparison'].iloc[0]['mae']
            st.metric("Best Model", best_model,
                     delta=f"MAE: {best_mae:.4f}",
                     help="Mô hình dự báo tốt nhất")
        else:
            st.metric("Best Model", "N/A")
    
    st.markdown("---")
    
    # Pipeline overview
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔄 Pipeline Phân Tích")
        st.markdown("""
        1. **📥 Thu thập dữ liệu** (Data Loading)
           - Dataset UCI Household Power Consumption
           - 2,075,259 records (2006-2010)
        
        2. **🧹 Tiền xử lý** (Preprocessing)
           - Xử lý missing values
           - Loại bỏ outliers
           - Chuẩn hóa dữ liệu
        
        3. **🔧 Feature Engineering**
           - Tạo biến thời gian (giờ, ngày, tuần...)
           - Tính toán các chỉ số aggregated
           - Rời rạc hóa (peak/off-peak)
        
        4. **⛏️ Khai phá dữ liệu** (Data Mining)
           - Association Mining (Apriori)
           - Clustering (KMeans)
           - Anomaly Detection (Isolation Forest)
        
        5. **📊 Mô hình hóa** (Modeling)
           - ARIMA, ETS, Holt-Winters
           - Baseline: Seasonal Naive
        
        6. **✅ Đánh giá & Báo cáo**
           - Metrics: MAE, RMSE, sMAPE
           - Insights & Recommendations
        """)
    
    with col2:
        st.markdown("### 🎯 Mục tiêu Dự án")
        st.success("""
        **1. Khai phá mẫu tiêu thụ** (Association Mining)
        - Phát hiện các pattern sử dụng điện
        - Tìm mối quan hệ giữa trạng thái tiêu thụ
        
        **2. Phân loại hộ gia đình** (Clustering)
        - Phân cụm theo profile tiêu thụ
        - Xác định nhóm tiêu thụ cao/thấp
        
        **3. Phát hiện bất thường** (Anomaly Detection)
        - Tìm ngày tiêu thụ bất thường
        - Phân tích theo mùa (seasonal)
        
        **4. Dự báo nhu cầu** (Forecasting)
        - Dự đoán tiêu thụ điện tương lai
        - So sánh nhiều mô hình
        - Phân tích seasonality & residuals
        """)
        
        st.markdown("### 🛠️ Công nghệ sử dụng")
        tech_cols = st.columns(3)
        with tech_cols[0]:
            st.markdown("**📊 Data**")
            st.code("pandas\nnumpy\nscipy")
        with tech_cols[1]:
            st.markdown("**🤖 ML**")
            st.code("scikit-learn\nstatsmodels\nmlxtend")
        with tech_cols[2]:
            st.markdown("**📈 Viz**")
            st.code("matplotlib\nseaborn\nplotly")

def show_eda():
    """Hiển thị EDA results"""
    st.markdown('<h2 class="section-header">📈 Phân Tích Khám Phá Dữ Liệu (EDA)</h2>', unsafe_allow_html=True)
    
    st.info("💡 **Lưu ý**: Các biểu đồ EDA chi tiết có trong notebook `01_eda.ipynb`")
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["⏰ Patterns thời gian", "📊 Phân phối", "🔍 Tổng hợp"])
    
    with tab1:
        st.markdown("### ⏰ Seasonal Patterns")
        
        figures_dir = Path('outputs/figures')
        seasonal_fig = figures_dir / '01_seasonal_patterns.png'
        
        if seasonal_fig.exists():
            st.image(str(seasonal_fig), caption="Biểu đồ seasonal patterns (hourly, daily, weekly)")
        else:
            st.warning("⚠️ Chưa có hình ảnh seasonal patterns. Vui lòng chạy notebook 01_eda.ipynb")
        
        st.markdown("""
        **Phân tích**:
        - 📈 **Giờ cao điểm**: 18h-21h (buổi tối)
        - 📉 **Giờ thấp điểm**: 2h-5h (đêm khuya)
        - 🔄 **Chu kỳ**: Rõ ràng theo giờ, ngày, tuần
        - 🏠 **Cuối tuần**: Tiêu thụ khác biệt so với ngày thường
        """)
    
    with tab2:
        st.markdown("### 📊 Submetering Comparison")
        
        submetering_fig = figures_dir / '01_submetering_comparison.png'
        
        if submetering_fig.exists():
            st.image(str(submetering_fig), caption="So sánh tiêu thụ theo thiết bị")
        else:
            st.warning("⚠️ Chưa có hình ảnh submetering. Vui lòng chạy notebook 01_eda.ipynb")
        
        st.markdown("""
        **Thiết bị tiêu thụ điện**:
        - 🍳 **Sub_metering_1**: Bếp (dishwasher, oven, microwave)
        - 🧺 **Sub_metering_2**: Giặt là (washing machine, dryer)
        - 🌡️ **Sub_metering_3**: Điều hòa & nước nóng (AC, water heater)
        """)
    
    with tab3:
        st.markdown("### 🔍 Tổng hợp Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ Phát hiện chính")
            st.success("""
            1. **Seasonality rõ ràng**
               - Daily: 24 giờ
               - Weekly: 7 ngày
            
            2. **Missing values**
               - ~1.25% dữ liệu thiếu
               - Xử lý bằng interpolation
            
            3. **Outliers**
               - ~3% dữ liệu bất thường
               - Sử dụng IQR method
            
            4. **Correlation**
               - High correlation giữa các biến power
               - Sub_metering có pattern khác nhau
            """)
        
        with col2:
            st.markdown("#### 💡 Khuyến nghị")
            st.info("""
            1. **Preprocessing**
               - Cần xử lý missing values cẩn thận
               - Outlier detection quan trọng
            
            2. **Feature Engineering**
               - Tạo biến thời gian (hour, day, month)
               - Aggregation theo ngày/tuần
            
            3. **Modeling**
               - Seasonal models (ARIMA, Holt-Winters)
               - Xử lý trend và seasonality
            
            4. **Validation**
               - Time-based split (không shuffle)
               - Cross-validation theo thời gian
            """)

def show_association(data):
    """Hiển thị Association Mining results"""
    st.markdown('<h2 class="section-header">🔗 Association Mining</h2>', unsafe_allow_html=True)
    
    if data['association_rules'] is None:
        st.warning("⚠️ Chưa có dữ liệu Association Rules. Vui lòng chạy notebook 03_mining_clustering.ipynb")
        return
    
    rules = data['association_rules']
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số Rules", len(rules))
    with col2:
        st.metric("Avg Confidence", f"{rules['confidence'].mean():.4f}")
    with col3:
        st.metric("Avg Lift", f"{rules['lift'].mean():.4f}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top Rules", "📈 Phân tích Metrics", "💡 Insights"])
    
    with tab1:
        st.markdown("### 🏆 Top Rules theo Lift")
        
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            min_confidence = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.5, 0.05)
        with col2:
            top_n = st.slider("Số lượng rules hiển thị", 5, 50, 10, 5)
        
        # Filter and display
        filtered_rules = rules[rules['confidence'] >= min_confidence].head(top_n)
        
        st.dataframe(
            filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
            use_container_width=True
        )
        
        st.markdown("#### 📝 Giải thích")
        st.info("""
        - **Support**: Tỷ lệ xuất hiện của itemset trong dataset
        - **Confidence**: Xác suất consequent xuất hiện khi có antecedent
        - **Lift**: Mức độ liên kết (>1: có liên kết dương)
        """)
    
    with tab2:
        st.markdown("### 📈 Phân tích Distribution")
        
        fig = go.Figure()
        
        # Support distribution
        fig.add_trace(go.Histogram(
            x=rules['support'],
            name='Support',
            marker_color='#667eea',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Phân phối Support của Association Rules",
            xaxis_title="Support",
            yaxis_title="Số lượng",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence vs Lift scatter
        fig2 = px.scatter(
            rules,
            x='confidence',
            y='lift',
            size='support',
            color='lift',
            hover_data=['antecedents', 'consequents'],
            title="Confidence vs Lift (size = Support)",
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 Insights từ Association Mining")
        
        # Top rules summary
        top_rules = rules.nlargest(5, 'lift')
        
        st.success(f"""
        **🔍 Phát hiện chính**:
        
        1. **Luật mạnh nhất**:
           - {top_rules.iloc[0]['antecedents']} → {top_rules.iloc[0]['consequents']}
           - Lift: {top_rules.iloc[0]['lift']:.2f}
           - Confidence: {top_rules.iloc[0]['confidence']:.2%}
        
        2. **Patterns phổ biến**:
           - {len(rules[rules['lift'] > 2])} rules có Lift > 2
           - {len(rules[rules['confidence'] > 0.8])} rules có Confidence > 80%
        
        3. **Ứng dụng**:
           - Dự đoán trạng thái tiêu thụ
           - Phát hiện pattern bất thường
           - Khuyến nghị tiết kiệm năng lượng
        """)

def show_clustering(data):
    """Hiển thị Clustering results"""
    st.markdown('<h2 class="section-header">🎯 Phân Cụm Hộ Gia Đình (Clustering)</h2>', unsafe_allow_html=True)
    
    if data['cluster_profiles'] is None:
        st.warning("⚠️ Chưa có dữ liệu Clustering. Vui lòng chạy notebook 03_mining_clustering.ipynb")
        return
    
    profiles = data['cluster_profiles']
    
    # Overview
    st.metric("Số cụm được xác định", len(profiles), help="Sử dụng KMeans clustering")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Cluster Profiles", "📈 Visualization", "💡 Interpretation"])
    
    with tab1:
        st.markdown("### 📊 Thông tin các cụm")
        
        st.dataframe(profiles, use_container_width=True)
        
        st.markdown("#### 📌 Giải thích các chỉ số")
        st.info("""
        - **mean_power**: Công suất trung bình (kW)
        - **std_power**: Độ lệch chuẩn (biến động)
        - Các chỉ số khác: hour, voltage, intensity, sub_metering...
        """)
    
    with tab2:
        st.markdown("### 📈 Visualizations")
        
        figures_dir = Path('notebooks/outputs/figures')
        cluster_fig = figures_dir / '03_cluster_profiles.png'
        
        if cluster_fig.exists():
            st.image(str(cluster_fig), caption="Cluster Profiles Visualization")
        else:
            st.warning("⚠️ Chưa có hình ảnh clusters")
        
        # Interactive plot with plotly
        # Check if mean_power exists (handle both single-level and multi-level columns)
        has_mean_power = False
        if isinstance(profiles.columns, pd.MultiIndex):
            has_mean_power = any('mean_power' in str(col) for col in profiles.columns)
        else:
            has_mean_power = 'mean_power' in profiles.columns
        
        if has_mean_power:
            try:
                fig = go.Figure()
                
                # Handle duplicate columns and convert to float safely
                mean_powers = []
                for idx in profiles.index:
                    # Try to get mean_power value
                    val = None
                    if isinstance(profiles.columns, pd.MultiIndex):
                        # Multi-level columns: get first column containing 'mean_power'
                        mean_power_cols = [col for col in profiles.columns if 'mean_power' in str(col)]
                        if mean_power_cols:
                            val = profiles.loc[idx, mean_power_cols[0]]
                    else:
                        val = profiles.loc[idx, 'mean_power']
                    
                    # Convert to float
                    try:
                        if val is None:
                            mean_powers.append(0.0)
                        elif hasattr(val, 'iloc'):
                            # If it's a Series, get first numeric value
                            mean_powers.append(float(val.iloc[0]))
                        else:
                            mean_powers.append(float(val))
                    except (ValueError, TypeError):
                        mean_powers.append(0.0)  # Default value if conversion fails
                
                fig.add_trace(go.Bar(
                    x=[f"Cụm {i}" for i in profiles.index],
                    y=mean_powers,
                    marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe'][:len(mean_powers)],
                    text=[f"{val:.2f} kW" for val in mean_powers],
                    textposition='auto'
                ))
            except Exception as e:
                st.error(f"Lỗi khi vẽ biểu đồ: {str(e)}")
                fig = None
            
            if fig is not None:
                fig.update_layout(
                    title="Công suất trung bình theo cụm",
                    xaxis_title="Cụm",
                    yaxis_title="Công suất (kW)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không thể hiển thị biểu đồ")
    
    with tab3:
        st.markdown("### 💡 Phân tích & Đặc điểm các cụm")
        
        # Interpret each cluster
        for cluster_id in profiles.index:
            with st.expander(f"🎯 Cụm {cluster_id} - Đặc điểm"):
                profile = profiles.loc[cluster_id]
                
                # Get mean power safely with extensive error handling
                try:
                    mean_power = None
                    
                    # Handle multi-level columns
                    if isinstance(profiles.columns, pd.MultiIndex):
                        mean_power_cols = [col for col in profiles.columns if 'mean_power' in str(col)]
                        if mean_power_cols:
                            mean_power_val = profile[mean_power_cols[0]]
                        else:
                            mean_power_val = None
                    elif 'mean_power' in profile.index:
                        mean_power_val = profile['mean_power']
                    else:
                        mean_power_val = None
                    
                    # Try multiple extraction methods
                    if mean_power_val is not None:
                        if hasattr(mean_power_val, 'iloc'):
                            try:
                                mean_power = float(mean_power_val.iloc[0])
                            except:
                                pass
                        elif isinstance(mean_power_val, (int, float)):
                            mean_power = float(mean_power_val)
                        elif isinstance(mean_power_val, str):
                            # Skip if it's a string like 'mean'
                            if mean_power_val.lower() not in ['mean', 'std', 'min', 'max']:
                                try:
                                    mean_power = float(mean_power_val)
                                except:
                                    pass
                        
                        if mean_power is not None:
                            # Classify cluster
                            if mean_power < 1.0:
                                category = "🟢 Tiêu thụ thấp"
                                description = "Hộ gia đình tiết kiệm điện"
                            elif mean_power < 1.5:
                                category = "🟡 Tiêu thụ trung bình"
                                description = "Hộ gia đình tiêu thụ ổn định"
                            else:
                                category = "🔴 Tiêu thụ cao"
                                description = "Hộ gia đình sử dụng nhiều điện"
                            
                            st.markdown(f"""
                            **Phân loại**: {category}
                            
                            **Công suất trung bình**: {mean_power:.2f} kW
                            
                            **Mô tả**: {description}
                            
                            **Khuyến nghị**:
                            - {'Duy trì thói quen tốt' if mean_power < 1.0 else 'Xem xét các biện pháp tiết kiệm'}
                            - {'Tiếp tục theo dõi' if mean_power < 1.5 else 'Kiểm tra thiết bị tiêu thụ cao'}
                            """)
                        else:
                            st.warning(f"⚠️ Không thể chuyển đổi dữ liệu cho cụm {cluster_id}")
                    else:
                        st.info("⚠️ Không có thông tin mean_power cho cụm này")
                except (ValueError, TypeError) as e:
                    st.error(f"⚠️ Lỗi xử lý dữ liệu cụm {cluster_id}: {str(e)}")

def show_anomaly(data):
    """Hiển thị Anomaly Detection results"""
    st.markdown('<h2 class="section-header">⚠️ Phát Hiện Bất Thường</h2>', unsafe_allow_html=True)
    
    if data['anomaly_seasonal'] is None:
        st.warning("⚠️ Chưa có dữ liệu Anomaly. Vui lòng chạy notebook 04_anomaly_forecasting.ipynb")
        return
    
    anomaly = data['anomaly_seasonal']
    
    # Overview metrics
    try:
        total_anomalies = int(anomaly['n_anomalies'].sum())
        total_days = int(anomaly['total_days'].sum())
        anomaly_rate = total_anomalies / total_days if total_days > 0 else 0
    except (ValueError, TypeError):
        total_anomalies = 0
        total_days = 0
        anomaly_rate = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số ngày bất thường", total_anomalies)
    with col2:
        st.metric("Tổng số ngày phân tích", total_days)
    with col3:
        st.metric("Tỷ lệ bất thường", f"{anomaly_rate:.2%}")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Phân tích theo mùa", "📈 Visualization", "💡 Insights"])
    
    with tab1:
        st.markdown("### 📊 Anomaly theo mùa (Season)")
        
        st.dataframe(anomaly, use_container_width=True)
        
        # Highlight most/least anomalous
        try:
            most_anomalous = anomaly['anomaly_rate'].idxmax()
            least_anomalous = anomaly['anomaly_rate'].idxmin()
        except (ValueError, TypeError, KeyError):
            most_anomalous = None
            least_anomalous = None
        
        if most_anomalous is not None and least_anomalous is not None:
            col1, col2 = st.columns(2)
            with col1:
                try:
                    st.error(f"""
                    **🔴 Mùa bất thường nhất**: {most_anomalous}
                    - Số ngày: {int(anomaly.loc[most_anomalous, 'n_anomalies'])}
                    - Tỷ lệ: {float(anomaly.loc[most_anomalous, 'anomaly_rate']):.2%}
                    """)
                except:
                    st.error("Không thể hiển thị thông tin mùa bất thường nhất")
            
            with col2:
                try:
                    st.success(f"""
                    **🟢 Mùa ổn định nhất**: {least_anomalous}
                    - Số ngày: {int(anomaly.loc[least_anomalous, 'n_anomalies'])}
                    - Tỷ lệ: {float(anomaly.loc[least_anomalous, 'anomaly_rate']):.2%}
                    """)
                except:
                    st.success("Không thể hiển thị thông tin mùa ổn định nhất")
    
    with tab2:
        st.markdown("### 📈 Biểu đồ phân tích")
        
        # Bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=anomaly.index,
            y=anomaly['anomaly_rate'],
            marker_color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'],
            text=[f"{val:.2%}" for val in anomaly['anomaly_rate']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Tỷ lệ bất thường theo mùa",
            xaxis_title="Mùa",
            yaxis_title="Tỷ lệ (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Pie chart
        fig2 = go.Figure(data=[go.Pie(
            labels=anomaly.index,
            values=anomaly['n_anomalies'],
            hole=.3,
            marker_colors=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        )])
        
        fig2.update_layout(
            title="Phân bố số lượng anomalies theo mùa",
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 Phân tích & Khuyến nghị")
        
        st.info(f"""
        **🔍 Phương pháp**: Isolation Forest
        
        **📊 Kết quả tổng hợp**:
        - Tổng: {total_anomalies} ngày bất thường / {total_days} ngày
        - Tỷ lệ: {anomaly_rate:.2%}
        - Mùa cao nhất: {most_anomalous} ({anomaly.loc[most_anomalous, 'anomaly_rate']:.2%})
        - Mùa thấp nhất: {least_anomalous} ({anomaly.loc[least_anomalous, 'anomaly_rate']:.2%})
        """)
        
        st.success("""
        **💡 Insights**:
        
        1. **Biến động theo mùa**:
           - Mùa đông & mùa hè có anomaly cao hơn
           - Có thể do điều hòa nhiệt độ
        
        2. **Nguyên nhân có thể**:
           - Thời tiết cực đoan
           - Ngày lễ, kỳ nghỉ
           - Sự cố hệ thống
        
        3. **Khuyến nghị**:
           - Thiết lập hệ thống cảnh báo
           - Điều tra nguyên nhân cụ thể
           - Lên kế hoạch bảo trì định kỳ
        """)

def show_forecasting(data):
    """Hiển thị Forecasting results"""
    st.markdown('<h2 class="section-header">📉 Dự Báo Nhu Cầu Năng Lượng</h2>', unsafe_allow_html=True)
    
    if data['forecast_comparison'] is None:
        st.warning("⚠️ Chưa có dữ liệu Forecasting. Vui lòng chạy notebook 04_anomaly_forecasting.ipynb")
        return
    
    forecast = data['forecast_comparison']
    
    # Best model
    try:
        best_model = forecast.index[0]
        best_mae = float(forecast.iloc[0]['mae'])
        best_rmse = float(forecast.iloc[0]['rmse'])
        best_smape = float(forecast.iloc[0]['smape'])
    except (ValueError, TypeError, KeyError, IndexError):
        best_model = "N/A"
        best_mae = 0.0
        best_rmse = 0.0
        best_smape = 0.0
    
    st.success(f"""
    ### 🏆 Mô hình tốt nhất: **{best_model}**
    - **MAE**: {best_mae:.4f}
    - **RMSE**: {best_rmse:.4f}
    - **sMAPE**: {best_smape:.2f}%
    """)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 So sánh Models", "📈 Metrics Detail", "🔍 Residual Analysis", "💡 Insights"])
    
    with tab1:
        st.markdown("### 📊 Bảng so sánh các mô hình")
        
        # Get available columns
        available_cols = ['mae', 'rmse', 'smape', 'r2', 'avg_rank']
        cols_to_show = [c for c in available_cols if c in forecast.columns]
        
        # Create styled dataframe
        styled_df = forecast[cols_to_show].style.highlight_min(
            subset=[c for c in ['mae', 'rmse', 'smape'] if c in cols_to_show],
            color='lightgreen'
        )
        
        # Highlight max R² (higher is better)
        if 'r2' in cols_to_show:
            styled_df = styled_df.highlight_max(
                subset=['r2'],
                color='lightblue'
            )
        
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("#### 📝 Giải thích Metrics")
        st.info("""
        - **MAE** (Mean Absolute Error): Sai số tuyệt đối trung bình - càng thấp càng tốt
        - **RMSE** (Root Mean Squared Error): Căn bậc hai sai số bình phương - càng thấp càng tốt  
        - **sMAPE** (Symmetric MAPE): Sai số phần trăm đối xứng - càng thấp càng tốt
        - **R²**: Hệ số xác định - càng gần 1 càng tốt
        - **avg_rank**: Thứ hạng trung bình - càng thấp càng tốt
        """)
    
    with tab2:
        st.markdown("### 📈 Chi tiết Metrics từng mô hình")
        
        cols = st.columns(2)
        
        for idx, model in enumerate(forecast.index):
            with cols[idx % 2]:
                st.markdown(f"#### 🔹 {model}")
                
                # Extract metrics safely
                try:
                    mae_val = float(forecast.loc[model, 'mae']) if 'mae' in forecast.columns else 0.0
                    rmse_val = float(forecast.loc[model, 'rmse']) if 'rmse' in forecast.columns else 0.0
                    smape_val = float(forecast.loc[model, 'smape']) if 'smape' in forecast.columns else 0.0
                    r2_val = float(forecast.loc[model, 'r2']) if 'r2' in forecast.columns else 0.0
                except (ValueError, TypeError):
                    mae_val = rmse_val = smape_val = r2_val = 0.0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MAE", f"{mae_val:.4f}")
                    st.metric("sMAPE", f"{smape_val:.2f}%")
                with col2:
                    st.metric("RMSE", f"{rmse_val:.4f}")
                    st.metric("R²", f"{r2_val:.4f}")
        
        st.markdown("---")
        
        # Visualizations
        st.markdown("### 📊 Visualizations so sánh")
        
        # MAE comparison
        fig1 = go.Figure()
        try:
            mae_values = [float(val) for val in forecast['mae']]
            fig1.add_trace(go.Bar(
                x=forecast.index,
                y=mae_values,
                marker_color=['#10ac84' if i == 0 else '#ee5a6f' for i in range(len(forecast))],
                text=[f"{val:.4f}" for val in mae_values],
                textposition='auto'
            ))
        except (ValueError, TypeError):
            st.error("Không thể chuyển đổi giá trị MAE")
        fig1.update_layout(
            title="So sánh MAE giữa các mô hình",
            xaxis_title="Mô hình",
            yaxis_title="MAE",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Multiple metrics comparison
        fig2 = go.Figure()
        
        metrics_to_plot = ['mae', 'rmse', 'smape']
        colors = ['#667eea', '#764ba2', '#f093fb']
        
        for i, metric in enumerate(metrics_to_plot):
            try:
                metric_values = [float(val) for val in forecast[metric]]
                fig2.add_trace(go.Bar(
                    name=metric.upper(),
                    x=forecast.index,
                    y=metric_values,
                    marker_color=colors[i]
                ))
            except (ValueError, TypeError):
                st.warning(f"Không thể hiển thị metric {metric}")
        
        fig2.update_layout(
            title="So sánh tổng hợp các metrics",
            xaxis_title="Mô hình",
            yaxis_title="Giá trị",
            barmode='group',
            height=450
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Phân tích Residuals")
        
        st.info("""
        **Residual Analysis** giúp đánh giá chất lượng mô hình:
        - Mean gần 0: Mô hình không bị bias
        - Std thấp: Dự báo ổn định
        - Skewness gần 0: Phân phối cân đối
        - Kurtosis gần 0: Không có outliers cực đoan
        """)
        
        # Load residual stats from forecast_summary.txt if available
        summary_file = Path('outputs/tables/forecast_summary.txt')
        if summary_file.exists():
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            st.text_area("Chi tiết Residual Analysis", content, height=400)
        else:
            st.warning("⚠️ Chưa có file forecast_summary.txt")
        
        # Visualization
        figures_dir = Path('notebooks/outputs/figures')
        residual_fig = figures_dir / '04_residuals.png'
        
        if residual_fig.exists():
            st.image(str(residual_fig), caption="Residual Analysis Plots")
    
    with tab4:
        st.markdown("### 💡 Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **✅ Kết quả đạt được**:
            
            1. **Best Model**: {best_model}
               - Cải thiện {((forecast.iloc[-1]['mae'] - best_mae) / forecast.iloc[-1]['mae'] * 100):.1f}% so với baseline
            
            2. **Seasonality**: 
               - Chu kỳ 24 giờ được phát hiện
               - Patterns ngày/tuần rõ ràng
            
            3. **Model Performance**:
               - MAE: {best_mae:.4f} kW
               - sMAPE: {best_smape:.2f}%
               - Đủ tốt cho ứng dụng thực tế
            """)
        
        with col2:
            st.info("""
            **🎯 Khuyến nghị**:
            
            1. **Deployment**:
               - Triển khai {best_model} cho dự báo
               - Cập nhật model định kỳ
            
            2. **Cải thiện**:
               - Thêm biến ngoại sinh (thời tiết, ngày lễ)
               - Thử Deep Learning (LSTM, Transformer)
               - Online learning cho real-time
            
            3. **Ứng dụng**:
               - Load balancing
               - Pricing optimization
               - Energy management
            """)

def show_report(data):
    """Hiển thị báo cáo tổng hợp"""
    st.markdown('<h2 class="section-header">📋 Báo Cáo Tổng Hợp</h2>', unsafe_allow_html=True)
    
    # Project summary
    st.markdown("### 📊 Tổng hợp toàn dự án")
    
    summary_data = {
        'Phần phân tích': [
            'Association Mining',
            'Clustering',
            'Anomaly Detection',
            'Forecasting'
        ],
        'Phương pháp': [
            'Apriori Algorithm',
            'KMeans Clustering',
            'Isolation Forest',
            'ARIMA, ETS, Holt-Winters'
        ],
        'Kết quả chính': [
            f"{len(data['association_rules'])} rules" if data['association_rules'] is not None else 'N/A',
            f"{len(data['cluster_profiles'])} cụm" if data['cluster_profiles'] is not None else 'N/A',
            f"{data['anomaly_seasonal']['n_anomalies'].sum():.0f} ngày" if data['anomaly_seasonal'] is not None else 'N/A',
            f"Best: {data['forecast_comparison'].index[0]}" if data['forecast_comparison'] is not None else 'N/A'
        ],
        'Trạng thái': [
            '✅ Hoàn thành' if data['association_rules'] is not None else '⏳ Chưa có',
            '✅ Hoàn thành' if data['cluster_profiles'] is not None else '⏳ Chưa có',
            '✅ Hoàn thành' if data['anomaly_seasonal'] is not None else '⏳ Chưa có',
            '✅ Hoàn thành' if data['forecast_comparison'] is not None else '⏳ Chưa có'
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Final insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Mục tiêu đạt được")
        st.success("""
        ✅ **Association Pattern Mining**:
        - Phát hiện patterns tiêu thụ điện
        - 41 luật kết hợp có ý nghĩa
        
        ✅ **Household Clustering**:
        - Phân loại 4 nhóm hộ gia đình
        - Profiles từ thấp đến cao
        
        ✅ **Anomaly Detection**:
        - 73 ngày bất thường được phát hiện
        - Phân tích theo mùa chi tiết
        
        ✅ **Time Series Forecasting**:
        - So sánh 4 models
        - Holt-Winters đạt kết quả tốt nhất
        - MAE: 0.3203, cải thiện 25% vs baseline
        """)
    
    with col2:
        st.markdown("### 💡 Insights chính")
        st.info("""
        **📈 Consumption Patterns**:
        - Seasonality rõ ràng (24h, 7 ngày)
        - Peak: 18-21h, Off-peak: 2-5h
        
        **🎯 Household Profiles**:
        - 4 nhóm tiêu thụ khác biệt
        - Từ tiết kiệm đến sử dụng nhiều
        
        **⚠️ Anomalies**:
        - Winter & Summer cao hơn
        - Có thể do điều hòa nhiệt độ
        
        **📊 Forecasting**:
        - Holt-Winters performs best
        - Seasonal components quan trọng
        - Có thể cải thiện với external data
        """)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 🚀 Khuyến nghị & Hướng phát triển")
    
    tab1, tab2, tab3 = st.tabs(["⚡ Energy Management", "🔧 Technical", "📈 Business"])
    
    with tab1:
        st.markdown("""
        #### ⚡ Quản lý năng lượng
        
        1. **Load Balancing**:
           - Chuyển tải từ peak hours (18-21h)
           - Khuyến khích sử dụng off-peak (2-5h)
           - Time-of-use pricing
        
        2. **Target Groups**:
           - Nhóm tiêu thụ cao: Chương trình tiết kiệm
           - Nhóm tiêu thụ thấp: Duy trì thói quen
           - Personalized recommendations
        
        3. **Anomaly Response**:
           - Real-time alerting system
           - Automated investigation
           - Preventive maintenance
        """)
    
    with tab2:
        st.markdown("""
        #### 🔧 Kỹ thuật
        
        1. **Model Improvement**:
           - Thêm external features (weather, holidays)
           - Deep Learning models (LSTM, Transformer)
           - Ensemble methods
        
        2. **System Integration**:
           - Real-time data pipeline
           - Online learning & model update
           - A/B testing framework
        
        3. **Monitoring**:
           - Model performance tracking
           - Data drift detection
           - Automated retraining
        """)
    
    with tab3:
        st.markdown("""
        #### 📈 Kinh doanh
        
        1. **Revenue Optimization**:
           - Dynamic pricing strategies
           - Demand response programs
           - Peak shaving incentives
        
        2. **Customer Segmentation**:
           - Tailored service packages
           - Targeted marketing campaigns
           - Retention strategies
        
        3. **Operational Planning**:
           - Capacity planning
           - Resource allocation
           - Infrastructure investment
        """)
    
    st.markdown("---")
    
    # Download report
    st.markdown("### 📥 Download Báo cáo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_file = Path('outputs/tables/final_project_report.txt')
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.download_button(
                label="📄 Download Text Report",
                data=report_content,
                file_name="final_report.txt",
                mime="text/plain"
            )
    
    with col2:
        if data['forecast_comparison'] is not None:
            csv = data['forecast_comparison'].to_csv(index=True)
            st.download_button(
                label="📊 Download Forecast Results",
                data=csv,
                file_name="forecast_comparison.csv",
                mime="text/csv"
            )
    
    with col3:
        if data['association_rules'] is not None:
            csv = data['association_rules'].to_csv(index=False)
            st.download_button(
                label="🔗 Download Association Rules",
                data=csv,
                file_name="association_rules.csv",
                mime="text/csv"
            )

# Run app
if __name__ == "__main__":
    main()
