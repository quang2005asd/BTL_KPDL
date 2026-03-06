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

## Data Dictionary

| Tên cột | Ý nghĩa | Đơn vị | Khoảng giá trị | Target/Feature |
|---------|---------|--------|----------------|----------------|
| `Date` | Ngày đo | - | 2006-12-16 đến 2010-11-26 | Index |
| `Time` | Thời gian đo | - | 00:00:00 đến 23:59:00 | Index |
| `Global_active_power` | Công suất hoạt động trung bình (toàn nhà) | kilowatt (kW) | 0.076 - 11.122 | **Target** (dự báo) |
| `Global_reactive_power` | Công suất phản kháng | kilowatt (kW) | 0.0 - 1.39 | Feature |
| `Voltage` | Điện áp | volt (V) | 223.2 - 254.15 | Feature |
| `Global_intensity` | Cường độ dòng điện | ampere (A) | 0.2 - 48.4 | Feature |
| `Sub_metering_1` | Tiêu thụ bếp (dishwasher, oven, microwave) | watt-hour (Wh) | 0 - 88 | Feature |
| `Sub_metering_2` | Tiêu thụ phòng giặt (washing machine, dryer, light) | watt-hour (Wh) | 0 - 80 | Feature |
| `Sub_metering_3` | Tiêu thụ điều hòa & nước nóng | watt-hour (Wh) | 0 - 31 | Feature |

**Tổng số records**: 2,075,259 (gần 4 năm dữ liệu)

## Data Quality Issues & Solutions

### 1. Missing Values (~1.25%)
- **Vấn đề**: 25,979 records có giá trị "?" (missing)
- **Nguyên nhân**: Lỗi đọc từ sensor/meter
- **Giải pháp**: 
  - Time-series interpolation (linear) cho missing ngắn (<1h)
  - Forward fill cho missing dài hơn
  - Drop nếu toàn bộ row missing

### 2. Outliers (~3%)
- **Vấn đề**: Giá trị bất thường (spike đột ngột, giá trị 0 bất thường)
- **Phát hiện**: IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- **Giải pháp**: 
  - Capping tại boundary thay vì remove (giữ lại data points)
  - Phân tích riêng outliers như anomaly detection task

### 3. Data Leakage Risk
- **Risk**: Time series có thể leak future information vào past
- **Prevention**: 
  - **Time-based split**: Train 70%, Val 15%, Test 15% theo thứ tự thời gian
  - **No shuffling**: Giữ nguyên thứ tự chronological
  - **Feature engineering**: Chỉ dùng lag features (past → predict future)
  - **Cross-validation**: Time Series Split (không dùng K-Fold random)

### 4. Class Imbalance
- **N/A**: Đây là bài toán regression (dự báo giá trị liên tục), không có class imbalance

### 5. Data Drift
- **Vấn đề tiềm ẩn**: Hành vi tiêu thụ có thể thay đổi theo năm (mua thiết bị mới, thay đổi thói quen)
- **Monitor**: Kiểm tra distribution shift giữa train/test
- **Mitigation**: Seasonal decomposition, adaptive models

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
