import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import json
import xarray as xr
from datetime import datetime, timedelta, timezone
import warnings

warnings.filterwarnings("ignore")

# ---------------- 1. Hằng số ----------------
MULTI_LEVEL_VARS_TRAIN = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
SURFACE_LEVEL_VARS_TRAIN = ['pressfc', 'tmpsfc', 'landmask']
MULTI_LEVEL_VARS_REMOTE = ['ugrdprs', 'vgrdprs', 'vvelprs', 'tmpprs', 'rhprs', 'hgtprs', 'absvprs']
SURFACE_LEVEL_VARS_REMOTE = ['pressfc', 'tmpsig995', 'vegsfc']

NUM_MULTI_LEVEL_VARS = len(MULTI_LEVEL_VARS_TRAIN)
NUM_SURFACE_VARS = len(SURFACE_LEVEL_VARS_TRAIN)
NUM_PRESSURE_LEVELS = 19
IMG_SIZE = 8
PRESSURE_LEVELS = [1000.0, 975.0, 950.0, 925.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0,
                   550.0, 500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0]

# ---------------- 2. Kiến trúc Mạng ----------------
class Model_Cach_1_CNN_RNN(nn.Module):
    def __init__(self, ml_channels=NUM_MULTI_LEVEL_VARS, sf_channels=NUM_SURFACE_VARS,
                 gru_hidden_size=128, gru_layers=2, cnn_features=64, dropout=0.3):
        super(Model_Cach_1_CNN_RNN, self).__init__()
        self.cnn_ml = nn.Sequential(
            nn.Conv2d(ml_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Conv2d(32, cnn_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.gru = nn.GRU(input_size=cnn_features, hidden_size=gru_hidden_size,
                          num_layers=gru_layers, batch_first=False,
                          dropout=dropout if gru_layers > 1 else 0)
        self.cnn_sf = nn.Sequential(
            nn.Conv2d(sf_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Conv2d(32, cnn_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size + cnn_features, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 2)
        )

    def forward(self, x_ml, x_sf):
        batch_size = x_ml.size(0)
        x_ml = x_ml.permute(0, 2, 1, 3, 4)
        x_ml = x_ml.reshape(batch_size * NUM_PRESSURE_LEVELS, NUM_MULTI_LEVEL_VARS, IMG_SIZE, IMG_SIZE)
        ml_features = self.cnn_ml(x_ml)
        ml_features = ml_features.reshape(batch_size, NUM_PRESSURE_LEVELS, -1)
        ml_features = ml_features.permute(1, 0, 2)
        _, h_n = self.gru(ml_features)
        gru_output = h_n[-1]
        sf_features = self.cnn_sf(x_sf)
        combined_features = torch.cat((gru_output, sf_features), dim=1)
        output = self.classifier(combined_features)
        return output

# ---------------- 3. Xử lý dữ liệu ----------------
# def buoc_3_get_and_process_data(lon_target, lat_target, grid_size=8):
#     # Thử tối đa 3 mốc thời gian (Hiện tại -> Lùi 6h -> Lùi 12h)
#     for attempt in range(3):
#         try:
#             now_utc = datetime.now(timezone.utc)
#             # Tăng offset mặc định lên 5h để an toàn hơn với độ trễ server NOAA
#             hours_offset = 5 + (attempt * 6)
            
#             target_time = now_utc - timedelta(hours=hours_offset)
#             run_hour_int = (target_time.hour // 6) * 6
            
#             run_time_obj = datetime(target_time.year, target_time.month, target_time.day, run_hour_int, tzinfo=timezone.utc)
#             run_date_str = run_time_obj.strftime('%Y%m%d')
#             run_hour_str = run_time_obj.strftime('%H')

#             # URL mới (Bỏ đuôi _anl, dùng file forecast chính)
#             opendap_url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{run_date_str}/gfs_0p25_{run_hour_str}z"
            
#             print(f"🔄 [Thử lần {attempt+1}] Kết nối: .../gfs{run_date_str}/gfs_0p25_{run_hour_str}z")

#             # Tính toán vùng cắt (Slicing)
#             # Lưu ý: Cần cắt dư ra một chút (+- 2 độ) để nội suy chính xác
#             lat_min, lat_max = lat_target - grid_size/2 - 2, lat_target + grid_size/2 + 2
#             lon_min, lon_max = lon_target - grid_size/2 - 2, lon_target + grid_size/2 + 2
            
#             # Chuyển đổi Lon về 0-360 cho GFS
#             lon_min_gfs = (lon_min + 360) % 360
#             lon_max_gfs = (lon_max + 360) % 360

#             # Mở dataset (Dùng engine='netcdf4')
#             with xr.open_dataset(opendap_url, engine='netcdf4') as ds:
#                 # Kiểm tra kết nối bằng biến nhẹ nhất (time)
#                 _ = ds['time'].values
                
#                 # Cắt vùng dữ liệu thô trước (Slice) để giảm tải download
#                 # Xử lý trường hợp vắt qua kinh tuyến 0 độ (hiếm gặp ở VN nhưng cần thiết)
#                 if lon_min_gfs > lon_max_gfs:
#                     # Vùng qua kinh tuyến 0: Phải tải 2 mảnh rồi ghép lại (phức tạp, ở đây ta dùng cách đơn giản là tải rộng hơn)
#                      # Hoặc dùng roll (nhưng chậm). 
#                      # Ở đây ta giả định vùng VN (100-120E) nên lon_min < lon_max là bình thường.
#                      ds_slice = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_gfs, 360)).isel(time=0)
#                 else:
#                     # Cắt vùng thông thường
#                     ds_slice = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_gfs, lon_max_gfs)).isel(time=0)
                
#                 # Tạo lưới đích để nội suy
#                 new_lats = np.arange(lat_target - (grid_size//2) + 0.5, lat_target + (grid_size//2) + 0.5, 1.0)
#                 new_lons = np.arange(lon_target - (grid_size//2) + 0.5, lon_target + (grid_size//2) + 0.5, 1.0)
#                 # new_lons_gfs = (new_lons + 360) % 360 # Không cần thiết vì interp xử lý được

#                 # 1. Xử lý Multi-level
#                 ds_ml_subset = ds_slice[MULTI_LEVEL_VARS_REMOTE].sel(lev=PRESSURE_LEVELS)
#                 ds_ml_interp = ds_ml_subset.interp(lat=new_lats, lon=new_lons, method="linear").load()
                
#                 # 2. Xử lý Surface
#                 ds_sf_subset = ds_slice[SURFACE_LEVEL_VARS_REMOTE]
#                 ds_sf_interp = ds_sf_subset.interp(lat=new_lats, lon=new_lons, method="linear").load()

#             print(f"✅ Tải thành công dữ liệu phiên: {run_date_str} {run_hour_str}z")

#             ml_arrays = []
#             ds_ml_sorted = ds_ml_interp.sortby('lev', ascending=False)
#             for var_remote in MULTI_LEVEL_VARS_REMOTE:
#                 var_data = ds_ml_sorted[var_remote].values
#                 ml_arrays.append(var_data)
#             data_ml_np = np.stack(ml_arrays, axis=0).astype(np.float32)

#             pressfc_data = ds_sf_interp['pressfc'].values
#             tmpsfc_sub_data = ds_sf_interp['tmpsig995'].values
#             vegsfc_data = ds_sf_interp['vegsfc'].values
#             landmask_sub_data = (vegsfc_data > 0).astype(np.float32)

#             data_sf_np = np.stack([pressfc_data, tmpsfc_sub_data, landmask_sub_data], axis=0).astype(np.float32)

#             return data_ml_np, data_sf_np

#         except Exception as e:
#             print(f"⚠️ Thất bại lần {attempt+1}: {e}")
#             if attempt == 2:
#                 print("❌ Đã hết số lần thử. Không thể kết nối Server NOAA.")
#                 return np.array([]), np.array([])
#             else:
#                 print("➡️ Đang thử lùi lại khung giờ trước...")
#                 continue 

#     return np.array([]), np.array([])
# TRONG FILE storm_model.py

def buoc_3_get_and_process_data(lon_target, lat_target, grid_size=8):
    # Thử tối đa 3 mốc thời gian (Hiện tại -> Lùi 6h -> Lùi 12h)
    for attempt in range(3):
        try:
            now_utc = datetime.now(timezone.utc)
            hours_offset = 5 + (attempt * 6)
            
            target_time = now_utc - timedelta(hours=hours_offset)
            run_hour_int = (target_time.hour // 6) * 6
            
            run_time_obj = datetime(target_time.year, target_time.month, target_time.day, run_hour_int, tzinfo=timezone.utc)
            run_date_str = run_time_obj.strftime('%Y%m%d')
            run_hour_str = run_time_obj.strftime('%H')

            opendap_url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{run_date_str}/gfs_0p25_{run_hour_str}z"
            print(f"🔄 [Thử lần {attempt+1}] Kết nối: .../gfs{run_date_str}/gfs_0p25_{run_hour_str}z")

            # --- SỬA LỖI KÍCH THƯỚC TẠI ĐÂY ---
            # Thay vì dùng np.arange (dễ lỗi dư 1 phần tử), ta cắt lát rộng hơn rồi resize sau
            # Lấy dư ra +- 5 độ để chắc chắn bao phủ
            lat_min, lat_max = lat_target - 5, lat_target + 5
            lon_min, lon_max = lon_target - 5, lon_target + 5
            
            lon_min_gfs = (lon_min + 360) % 360
            lon_max_gfs = (lon_max + 360) % 360

            with xr.open_dataset(opendap_url, engine='netcdf4') as ds:
                # Kiểm tra kết nối
                _ = ds['time'].values
                
                # Cắt vùng rộng trước để giảm tải
                if lon_min_gfs > lon_max_gfs:
                     ds_slice = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_gfs, 360)).isel(time=0)
                else:
                    ds_slice = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_gfs, lon_max_gfs)).isel(time=0)
                
                # Tạo lưới đích chính xác bằng linspace (đảm bảo luôn đủ grid_size điểm)
                # grid_size = 8
                # Khoảng cách mỗi điểm là 1 độ. Từ tâm ra mỗi bên là 3.5 độ (tổng 7 độ, 8 điểm)
                half_span = (grid_size - 1) / 2.0  # Ví dụ 8 điểm -> span 3.5
                new_lats = np.linspace(lat_target - half_span, lat_target + half_span, grid_size)
                new_lons = np.linspace(lon_target - half_span, lon_target + half_span, grid_size)

                # 1. Xử lý Multi-level
                ds_ml_subset = ds_slice[MULTI_LEVEL_VARS_REMOTE].sel(lev=PRESSURE_LEVELS)
                ds_ml_interp = ds_ml_subset.interp(lat=new_lats, lon=new_lons, method="linear").load()
                
                # 2. Xử lý Surface
                ds_sf_subset = ds_slice[SURFACE_LEVEL_VARS_REMOTE]
                ds_sf_interp = ds_sf_subset.interp(lat=new_lats, lon=new_lons, method="linear").load()

            print(f"✅ Tải thành công dữ liệu phiên: {run_date_str} {run_hour_str}z")

            ml_arrays = []
            ds_ml_sorted = ds_ml_interp.sortby('lev', ascending=False)
            for var_remote in MULTI_LEVEL_VARS_REMOTE:
                var_data = ds_ml_sorted[var_remote].values
                ml_arrays.append(var_data)
            data_ml_np = np.stack(ml_arrays, axis=0).astype(np.float32)

            pressfc_data = ds_sf_interp['pressfc'].values
            tmpsfc_sub_data = ds_sf_interp['tmpsig995'].values
            vegsfc_data = ds_sf_interp['vegsfc'].values
            landmask_sub_data = (vegsfc_data > 0).astype(np.float32)

            data_sf_np = np.stack([pressfc_data, tmpsfc_sub_data, landmask_sub_data], axis=0).astype(np.float32)

            # --- QUAN TRỌNG NHẤT: CẮT GỌT LẦN CUỐI ĐỂ TRÁNH LỖI SHAPE ---
            # Đảm bảo output luôn là (..., 8, 8) kể cả khi nội suy bị dư
            data_ml_np = data_ml_np[:, :, :grid_size, :grid_size]
            data_sf_np = data_sf_np[:, :grid_size, :grid_size]
            
            return data_ml_np, data_sf_np

        except Exception as e:
            print(f"⚠️ Thất bại lần {attempt+1}: {e}")
            if attempt == 2:
                print("❌ Đã hết số lần thử. Không thể kết nối Server NOAA.")
                return np.array([]), np.array([])
            else:
                print("➡️ Đang thử lùi lại khung giờ trước...")
                continue 

    return np.array([]), np.array([])
# ---------------- 4. Lớp StormPredictor ----------------
class StormPredictor:
    def __init__(self, model_path, stats_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Thiết bị sử dụng: {self.device}")
        
        self.model = Model_Cach_1_CNN_RNN()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except:
             print("⚠️ Cảnh báo: Đang load model state_dict (có thể lỗi nếu file sai định dạng)")
             pass
             
        self.model.to(self.device)
        self.model.eval()

        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.transform_ml = T.Compose([T.Normalize(mean=stats['mean_ml'], std=stats['std_ml'])])
        self.transform_sf = T.Compose([T.Normalize(mean=stats['mean_sf'], std=stats['std_sf'])])

    def predict_for_location(self, lon, lat):
        print(f"\n--- BẮT ĐẦU DỰ BÁO CHO ({lon}, {lat}) ---")
        try:
            # 1. TẢI DỮ LIỆU
            print("Bước 1: Đang tải dữ liệu GFS từ Server...")
            data_ml_np, data_sf_np = buoc_3_get_and_process_data(lon, lat, grid_size=IMG_SIZE)
            
            if data_ml_np.size == 0 or data_sf_np.size == 0:
                print("❌ Thất bại tại Bước 1: Không tải được dữ liệu.")
                return -1.0

            print(">>> Bước 1: Tải GFS thành công.")

            # 2. TIỀN XỬ LÝ (CHUẨN HÓA)
            print("Bước 2: Đang chuẩn hóa dữ liệu cho PyTorch...")
            data_ml = torch.tensor(data_ml_np, dtype=torch.float32)
            data_sf = torch.tensor(data_sf_np, dtype=torch.float32)

            data_ml_norm = self.transform_ml(data_ml.reshape(NUM_MULTI_LEVEL_VARS * NUM_PRESSURE_LEVELS, IMG_SIZE, IMG_SIZE))
            data_ml_norm = data_ml_norm.reshape(NUM_MULTI_LEVEL_VARS, NUM_PRESSURE_LEVELS, IMG_SIZE, IMG_SIZE)

            data_sf_norm = self.transform_sf(data_sf)
            
            inputs_ml = data_ml_norm.unsqueeze(0).to(self.device)
            inputs_sf = data_sf_norm.unsqueeze(0).to(self.device)
            print(">>> Bước 2: Chuẩn hóa thành công.")
            
            # 3. DỰ ĐOÁN
            print("Bước 3: Đang chạy mô hình AI...")
            with torch.no_grad():
                outputs = self.model(inputs_ml, inputs_sf)
                probabilities = torch.softmax(outputs, dim=1)
                prob_storm = probabilities[0, 1].item()
            
            print(f">>> Bước 3: Dự đoán thành công. Xác suất: {prob_storm:.4f}")
            return prob_storm
        
        except Exception as e:
            print(f"!!!!!! LỖI NGHIÊM TRỌNG TRONG predict_for_location: {e} !!!!!!")
            import traceback
            traceback.print_exc()
            return -1.0