# 🚀 Hướng dẫn chạy Streamlit Dashboard

## Bước 1: Cài đặt Streamlit (nếu chưa có)

```bash
pip install streamlit plotly
```

Hoặc cài toàn bộ dependencies:

```bash
pip install -r requirements.txt
```

## Bước 2: Đảm bảo đã chạy pipeline

Dashboard cần các file kết quả trong `outputs/tables/`:
- association_rules.csv
- cluster_profiles.csv  
- forecast_model_comparison.csv
- seasonal_anomaly_analysis.csv

Nếu chưa có, chạy:

```bash
python scripts/run_pipeline.py
```

## Bước 3: Chạy Dashboard

```bash
streamlit run app.py
```

Dashboard sẽ mở tại: **http://localhost:8501**

## ✨ Tính năng Dashboard

### 🏠 Tổng Quan
- Overview toàn dự án
- Key metrics
- Pipeline explanation

### 📈 Phân Tích EDA
- Seasonal patterns
- Submetering comparison  
- Distribution analysis

### 🔗 Association Mining
- Top rules theo Lift
- Metrics distribution
- Confidence vs Lift scatter plot

### 🎯 Phân Cụm (Clustering)
- Cluster profiles table
- Interactive visualizations
- Phân tích đặc điểm từng cụm

### ⚠️ Phát Hiện Bất Thường
- Anomaly theo mùa
- Bar charts & Pie charts
- Insights & recommendations

### 📉 Dự Báo (Forecasting)
- So sánh 4 models
- Best model highlighting
- Residual analysis
- Interactive charts

### 📋 Báo Cáo Tổng Hợp
- Tổng hợp toàn bộ kết quả
- Download reports (CSV, TXT)
- Insights & recommendations

## 🎨 Giao diện

- ✅ **Tiếng Việt** (trừ thuật ngữ kỹ thuật)
- ✅ **Design đẹp** (gradient, cards, charts)
- ✅ **Đầy đủ** (tất cả phần phân tích)
- ✅ **Mạch lạc** (sidebar navigation, tabs)

## 🛠️ Troubleshooting

### Lỗi: Module not found

```bash
pip install streamlit plotly
```

### Lỗi: File not found

Chạy pipeline trước:

```bash
python scripts/run_pipeline.py
```

### Port đã được sử dụng

Đổi port:

```bash
streamlit run app.py --server.port 8502
```

## 📱 Tips

- **F5** hoặc **R**: Reload page
- **Ctrl + Shift + R**: Clear cache & reload
- **Settings** (góc trên phải): Wide mode, theme...

---

**Enjoy! 🎉**
