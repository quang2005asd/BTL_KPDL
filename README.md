# Dự báo nhu cầu năng lượng (Household Power Consumption)

## Mô tả đề tài
Dự án khai phá dữ liệu để phân tích và dự báo nhu cầu năng lượng hộ gia đình sử dụng dataset UCI Household Power Consumption.

## Dataset
- **Nguồn**: [UCI Machine Learning Repository - Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Mô tả**: Dữ liệu tiêu thụ điện của một hộ gia đình tại Pháp trong giai đoạn 2006-2010
- **Tần suất**: Mỗi phút
- **Biến chính**: 
  - `Global_active_power`: Công suất hoạt động trung bình (kW)
  - `Global_reactive_power`: Công suất phản kháng (kW)
  - `Voltage`: Điện áp (V)
  - `Global_intensity`: Cường độ dòng điện (A)
  - `Sub_metering_1/2/3`: Tiêu thụ điện của các thiết bị con

## Mục tiêu
1. **Khai phá mẫu (Association Mining)**: Tìm các pattern sử dụng điện năng (thiết bị/trạng thái đồng xuất hiện)
2. **Phân cụm (Clustering)**: Phân loại hộ gia đình theo profile tiêu thụ (night-owl, peak-heavy, stable...)
3. **Phát hiện bất thường (Anomaly Detection)**: Tìm ngày có mức tiêu thụ bất thường
4. **Dự báo (Time Series Forecasting)**: Dự báo nhu cầu năng lượng tương lai

## Cấu trúc project
```
DATA_MINING_PROJECT/
├── configs/
│   └── params.yaml              # Tham số cấu hình
├── data/
│   ├── raw/                     # Dữ liệu gốc (không commit)
│   └── processed/               # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_eda.ipynb            # Khám phá dữ liệu
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý & feature engineering
│   ├── 03_mining_clustering.ipynb   # Khai phá mẫu & phân cụm
│   ├── 04_anomaly_forecasting.ipynb # Phát hiện bất thường & dự báo
│   └── 05_evaluation_report.ipynb   # Đánh giá & báo cáo
├── src/
│   ├── data/                    # Module xử lý dữ liệu
│   ├── features/                # Feature engineering
│   ├── mining/                  # Khai phá mẫu, phân cụm, anomaly
│   ├── models/                  # Mô hình dự báo
│   ├── evaluation/              # Đánh giá kết quả
│   └── visualization/           # Vẽ biểu đồ
├── scripts/
│   └── run_pipeline.py          # Chạy toàn bộ pipeline
├── outputs/
│   ├── figures/                 # Hình ảnh kết quả
│   ├── tables/                  # Bảng kết quả
│   ├── models/                  # Mô hình đã train
│   └── reports/                 # Báo cáo
└── requirements.txt             # Dependencies

```

## Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd DATA_MINING_PROJECT
```

### 2. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Tải dataset
- Tải dataset từ [UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip)
- Giải nén và đặt file `household_power_consumption.txt` vào thư mục `data/raw/`

### 5. Cấu hình
Chỉnh sửa file `configs/params.yaml` để cập nhật đường dẫn dữ liệu và tham số

## Hướng dẫn chạy

### Option 1: Chạy Web Dashboard (Streamlit) ⭐ Khuyến nghị
```bash
streamlit run app.py
```
Mở trình duyệt tại http://localhost:8501

### Option 2: Chạy toàn bộ pipeline
```bash
python scripts/run_pipeline.py
```

### Option 3: Chạy từng bước qua notebooks
Mở Jupyter và chạy lần lượt các notebook trong thư mục `notebooks/`:
1. `01_eda.ipynb` - Khám phá dữ liệu
2. `02_preprocess_feature.ipynb` - Tiền xử lý
3. `03_mining_clustering.ipynb` - Khai phá & phân cụm
4. `04_anomaly_forecasting.ipynb` - Phát hiện bất thường & dự báo
5. `05_evaluation_report.ipynb` - Đánh giá tổng hợp

## Kết quả dự kiến
- Luật kết hợp về sử dụng thiết bị/trạng thái năng lượng
- Các cụm profile tiêu thụ điện khác nhau
- Phát hiện các ngày bất thường (F1-score, phân tích theo mùa)
- Dự báo nhu cầu năng lượng (MAE, RMSE, sMAPE)

## Nhóm thực hiện
- Thành viên 1: Nguyễn Việt Quang

# Giảng viên hướng dẫn
Lê Thị Thùy Trang
