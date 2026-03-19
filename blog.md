# Blog – Phân tích kết quả đồ án Khai phá dữ liệu (Household Power Consumption)

> Workspace: `DATA_MINING_PROJECT`  
> Nguồn dữ liệu: UCI Household Power Consumption (2006–2010)  
> Dạng bài toán: Chuỗi thời gian (forecasting) + khai phá mẫu (association) + phân cụm theo ngày (clustering) + phát hiện bất thường (anomaly)

## 1) Bài toán & mục tiêu
Mục tiêu của đồ án là biến một tập dữ liệu tiêu thụ điện theo thời gian (ghi theo phút) thành một hệ thống phân tích hoàn chỉnh, gồm:

- **Tiền xử lý**: xử lý missing/outlier, chuẩn hoá tần suất.
- **Feature engineering**: tạo đặc trưng thời gian, rời rạc hoá trạng thái tiêu thụ, lag/rolling (phục vụ dự báo).
- **Khai phá dữ liệu**:
  - **Association mining** để tìm luật kết hợp giữa các trạng thái như giờ cao điểm/ban đêm/cuối tuần/ngoại lai.
  - **Clustering** để phân nhóm “hồ sơ tiêu thụ theo ngày”.
  - **Anomaly detection** để phát hiện ngày tiêu thụ bất thường và phân tích theo mùa.
- **Forecasting**: so sánh baseline và các mô hình dự báo kinh điển (ARIMA/ETS/Holt–Winters) và đánh giá bằng MAE/RMSE/sMAPE.

Toàn bộ kết quả được lưu vào thư mục `outputs/` và trình bày lại bằng notebook + dashboard Streamlit.

---

## 2) Pipeline dữ liệu (tóm tắt)
Các bước chính (chạy bằng `scripts/run_pipeline.py`):

1. **Load data** từ file raw.
2. **Clean**:
   - Nội suy theo thời gian để xử lý thiếu.
   - Resample về **1 giờ** (aggregation theo **mean**).
   - Gắn cờ `is_outlier` bằng phương pháp IQR.
3. **Feature engineering**:
   - Time features: `hour`, `day_of_week`, `month`, `season`, `is_weekend`, thêm `is_night`, `is_peak_hour`.
   - Rời rạc hoá mức tiêu thụ thành 3 trạng thái: `off-peak`, `normal`, `peak` + các cờ `is_*`.
   - Lag: 1h, 24h, 168h; rolling windows: 24h, 168h (mean/std/min/max).
4. **Association mining** (Apriori) trên các cờ `is_*` theo cửa sổ giao dịch (ví dụ 6H).
5. **Clustering**: tạo profile theo ngày → chuẩn hoá → KMeans (4 cụm).
6. **Anomaly detection**: gộp theo ngày → Isolation Forest (contamination 0.05) → phân tích theo mùa.
7. **Forecasting**: baseline Seasonal Naive + ARIMA + ETS + Holt–Winters → so sánh metric và residual.

---

## 3) Kết quả chính theo từng phần

### 3.1 Association Mining – Luật kết hợp
Kết quả lưu ở `outputs/tables/association_rules.csv`.

- Hệ thống trích xuất **20 luật** lọc theo ngưỡng support/confidence/lift.
- Một số luật nổi bật (đọc theo hướng “IF … THEN …”):
  - **`is_peak → is_outlier`** có confidence rất cao (~0.99) và lift > 2.
    - Diễn giải: khi xảy ra trạng thái tiêu thụ “peak”, xác suất xuất hiện mẫu ngoại lai tăng mạnh so với ngẫu nhiên.
  - **`is_weekend & is_normal → is_outlier`** cho thấy cuối tuần dễ xuất hiện biến động bất thường.

**Ý nghĩa thực tế**: luật kết hợp giúp “giải thích” (explainability) cho các hiện tượng tăng vọt tiêu thụ theo ngữ cảnh thời gian, là phần bổ trợ tốt cho anomaly/forecasting.

---

### 3.2 Clustering – Phân cụm hồ sơ tiêu thụ theo ngày
Kết quả lưu ở `outputs/tables/cluster_profiles.csv`.

- KMeans phân thành **4 cụm** dựa trên profile theo ngày.
- Các cụm khác nhau rõ ở:
  - **mức tiêu thụ trung bình (mean_power)**,
  - **độ biến động (std_power)**,
  - **giờ đạt đỉnh (peak_hour)**,
  - và các tỷ lệ như tiêu thụ ban đêm / cuối tuần.

Một vài con số minh hoạ:
- Có cụm **tiêu thụ cao** với mean_power khoảng **2.144**.
- Có cụm **tiêu thụ thấp** với mean_power khoảng **0.805**.
- `peak_hour` trung bình giữa các cụm lệch nhau đáng kể (cụm đỉnh buổi sáng / trưa / chiều / tối), phản ánh hành vi sử dụng điện theo lịch sinh hoạt.

**Ý nghĩa thực tế**: phân cụm giúp phân đoạn hành vi (segmentation), từ đó gợi ý chiến lược quản lý tải theo nhóm ngày/hành vi.

---

### 3.3 Anomaly Detection – Phát hiện ngày bất thường
Kết quả lưu ở `outputs/tables/anomaly_summary.csv` và `outputs/tables/seasonal_anomaly_analysis.csv`.

- Tổng số ngày bất thường phát hiện: **73 / 1442 ngày** → tỷ lệ khoảng **5.06%** (phù hợp contamination 0.05).
- Bất thường theo mùa:
  - **Winter**: 30 ngày (~8.67%)
  - **Summer**: 29 ngày (~7.88%)
  - **Spring**: 8 ngày (~2.17%)
  - **Autumn**: 6 ngày (~1.67%)

**Diễn giải**:
- Winter/Summer thường có nhu cầu sưởi/điều hoà (hoặc các thiết bị điện hoạt động theo mùa), nên xác suất xuất hiện ngày bất thường cao hơn.
- Đây là một “insight” tốt vì vừa hợp lý trực giác, vừa chứng minh mô hình anomaly không hoạt động ngẫu nhiên.

---

### 3.4 Forecasting – So sánh mô hình dự báo chuỗi thời gian
Kết quả lưu ở `outputs/tables/forecast_model_comparison.csv` và `outputs/tables/forecast_summary.txt`.

Các mô hình so sánh:
- Baseline: **Seasonal Naive**
- **ARIMA**
- **ETS**
- **Holt–Winters**

**Kết quả nổi bật (Test set):**
- Best (đồng hạng): **Holt–Winters / ETS**
  - **MAE = 0.3203**
  - **RMSE = 0.3342**
  - **sMAPE = 25.0865%**
- Baseline Seasonal Naive:
  - MAE = 0.4286, RMSE = 0.5542, sMAPE = 45.3548%

**Mức cải thiện của Holt–Winters so với baseline:**
- MAE giảm **~25.27%**
- RMSE giảm **~39.70%**
- sMAPE giảm **~44.69%**

**So với ARIMA**, Holt–Winters cải thiện:
- MAE giảm **~10.51%**, RMSE giảm **~25.48%**, sMAPE giảm **~12.19%**.

**Residual analysis** (tóm tắt từ `forecast_summary.txt`):
- ETS/Holt–Winters có residual mean hơi âm (~ -0.0529), cho thấy xu hướng dự báo hơi cao hơn thực tế một chút.
- Baseline có residual mean dương (~0.4286), thường phản ánh mô hình baseline “đuổi theo” chu kỳ nhưng không bắt được biến động thực tế.

**Lưu ý kỹ thuật**: trong chuỗi thời gian, $R^2$ có thể âm và không phản ánh tốt bằng MAE/RMSE/sMAPE; vì vậy báo cáo ưu tiên các metric sai số tuyệt đối/tương đối.

---

### 3.5 Training Data Efficiency (Learning Curve) – Nhánh tương đương tiêu chí bán giám sát
Kết quả lưu ở `outputs/tables/learning_curve_analysis.csv`.

Ý tưởng: thay vì “semi-supervised” theo nghĩa nhãn (vì đây là regression/forecasting), đồ án bổ sung thí nghiệm **hiệu quả dữ liệu huấn luyện**: thay đổi % dữ liệu train (10/25/50/75/100) và quan sát sai số.

Kết quả quan sát được:
- MAE không giảm đơn điệu khi tăng dữ liệu train. Ví dụ:
  - 10% train: MAE ~ 0.3866
  - 100% train: MAE ~ 0.7937

**Diễn giải khả dĩ**:
- Chuỗi có thể bị **non-stationarity / concept drift**: dữ liệu quá cũ không còn đại diện tốt cho hành vi hiện tại.
- Do đó, chiến lược hợp lý là thử:
  - huấn luyện theo **sliding window** (ưu tiên dữ liệu gần),
  - hoặc walk-forward evaluation và retrain định kỳ.

---

## 4) Thảo luận tổng hợp: điểm mạnh và hạn chế

### Điểm mạnh
- Pipeline rõ ràng, chạy end-to-end và lưu output đầy đủ.
- Kết quả forecasting cho thấy mô hình mùa vụ (ETS/Holt–Winters) vượt baseline đáng kể.
- Anomaly có phân bố theo mùa hợp lý.
- Association + clustering cung cấp góc nhìn giải thích/segmentation hỗ trợ forecasting.

### Hạn chế
- Anomaly/clustering là bài toán không nhãn → đánh giá chủ yếu dựa vào phân tích mô tả và tính hợp lý theo mùa.
- Learning curve cho thấy thêm dữ liệu không luôn tốt → cần thiết kế lại theo “cửa sổ huấn luyện” hoặc đánh giá theo nhiều giai đoạn thời gian.
- Chưa có biến ngoại sinh (thời tiết/ngày lễ/giá điện), nên forecast vẫn bị giới hạn khi có sự kiện bất thường.

---

## 5) Hướng phát triển
- **Thêm dữ liệu ngoại sinh**: thời tiết, ngày lễ, lịch làm việc.
- **Nâng cấp forecasting**: SARIMA theo mùa, Prophet/TBATS, hoặc ML model dùng lag/rolling (GBM/XGBoost).
- **Đánh giá chuẩn chuỗi thời gian**: walk-forward, rolling window, nhiều horizon dự báo.
- **Anomaly**: ensemble nhiều detector + quy trình phản hồi/ghi nhãn từ vận hành để đánh giá precision/recall thực.
- **Dashboard**: bổ sung phần hiển thị learning curve và đồng bộ các số mô tả (không hard-code) để khớp đúng với file output.

---

## 6) Kết luận
Đồ án đã xây dựng thành công một pipeline khai phá dữ liệu tiêu thụ điện theo thời gian, kết hợp mô hình hoá dự báo và phân tích bất thường, đồng thời tạo thêm lớp diễn giải bằng luật kết hợp và phân cụm hồ sơ theo ngày. Kết quả thực nghiệm cho thấy ETS/Holt–Winters là lựa chọn phù hợp trong bối cảnh dữ liệu theo giờ có seasonality rõ rệt, và phân tích theo mùa giúp giải thích các ngày bất thường tập trung vào Winter/Summer.
