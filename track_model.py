# ===============================================================
#  track_model.py - MODULE DỰ BÁO QUỸ ĐẠO (GBDT + GFS + ANALOG)
# ===============================================================

import os
import joblib
import pandas as pd
import numpy as np
import xarray as xr
import traceback
from datetime import datetime, timedelta, timezone
from scipy.spatial import KDTree
import warnings

warnings.filterwarnings('ignore')

class TrackPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.ibtracs_path = os.path.join(model_dir, 'ibtracs.WP.list.v04r01.csv')
        self.predictors_path = os.path.join(model_dir, 'predictors_common.joblib')
        
        self.global_storm_data = None
        self.global_predictors = None
        self.models = {}
        
        # Tự động tải dữ liệu khi khởi tạo
        self._load_resources()

    def _load_resources(self):
        print("🔄 [TrackModel] Đang khởi tạo hệ thống dự báo quỹ đạo...")
        
        # 1. Tải IBTrACS
        if os.path.exists(self.ibtracs_path):
            use_cols = ['SID', 'ISO_TIME', 'BASIN', 'NAME', 'LAT', 'LON', 'WMO_WIND', 'WMO_PRES']
            df = pd.read_csv(self.ibtracs_path, usecols=lambda c: c in use_cols, low_memory=False, na_values=[' ', 'MM'], keep_default_na=True)
            df = df[df['BASIN'] == 'WP']
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
            for c in ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']: df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.sort_values(['SID', 'ISO_TIME'])
            
            # Lag features
            for h in [6, 12, 18, 24]:
                steps = int(h/6)
                for col in ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES']:
                    df[f'{col}_{h}h'] = df.groupby('SID')[col].shift(steps)
            
            df['U_vel'] = df['LON'] - df['LON_6h']
            df['V_vel'] = df['LAT'] - df['LAT_6h']
            
            self.global_storm_data = df.dropna(subset=['LAT_24h', 'LON_24h', 'U_vel', 'V_vel']).reset_index(drop=True)
            print(f"✅ [TrackModel] IBTrACS loaded: {len(self.global_storm_data)} records.")
        else:
            print(f"❌ [TrackModel] Không tìm thấy IBTrACS tại {self.ibtracs_path}")

        # 2. Tải Predictors Common
        if os.path.exists(self.predictors_path):
            self.global_predictors = joblib.load(self.predictors_path)
            print("✅ [TrackModel] Predictors list loaded.")

        # 3. Tải 8 Models GBDT (.joblib)
        horizons = [6, 12, 18, 24]
        for h in horizons:
            # Định dạng tên file: gbdt_lat_06h.joblib
            h_str = f"{h:02d}" 
            lat_path = os.path.join(self.model_dir, f'gbdt_lat_{h_str}h.joblib')
            lon_path = os.path.join(self.model_dir, f'gbdt_lon_{h_str}h.joblib')
            
            if os.path.exists(lat_path) and os.path.exists(lon_path):
                try:
                    self.models[h] = {
                        'lat': joblib.load(lat_path),
                        'lon': joblib.load(lon_path)
                    }
                    print(f"✅ [TrackModel] Model {h}h loaded.")
                except Exception as e:
                    print(f"❌ [TrackModel] CRITICAL ERROR loading model {h}h!")
                    print("="*40)
                    traceback.print_exc() # <--- IN TOÀN BỘ STACK TRACE TẠI ĐÂY
                    print("="*40)
            else:
                 print(f"❌ [TrackModel] Thiếu file model cho {h}h")

    def find_analog(self, lat, lon):
        if self.global_storm_data is None: return None
        # Lọc bão đi sang Tây
        candidates = self.global_storm_data[self.global_storm_data['U_vel'] < -0.1].copy()
        if candidates.empty: candidates = self.global_storm_data
        
        tree = KDTree(candidates[['LAT', 'LON']].values)
        dist, idx = tree.query([lat, lon], k=1)
        return candidates.iloc[idx]

    def get_gfs_data(self, lat, lon):
        """Lấy dữ liệu GFS real-time với cơ chế retry."""
        print(f"🔌 [TrackModel] Đang kết nối GFS cho ({lat}, {lon})...")
        for attempt in range(3):
            try:
                now = datetime.now(timezone.utc)
                offset = 5 + (attempt * 6)
                safe_time = now - timedelta(hours=offset)
                cycle_hour = (safe_time.hour // 6) * 6
                date_str = safe_time.strftime('%Y%m%d')
                
                url = f'https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{cycle_hour:02d}z'
                
                ds = xr.open_dataset(url, engine='netcdf4')
                point = ds.sel(lat=lat, lon=lon, method='nearest').isel(time=0)
                
                mapping = {'u10': ['ugrdsig995'], 'v10': ['vgrdsig995'], 'msl': ['prmslmsl'], 
                           'sst': ['tmpsig995'], 'tcwv': ['pwatclm']}
                
                env_data = {}
                for target, candidates in mapping.items():
                    val = 0.0
                    for c in candidates:
                        if c in ds:
                            try: val = point[c].values.item(); break
                            except: pass
                    env_data[target] = val
                
                wind_ms = np.sqrt(env_data.get('u10', 0)**2 + env_data.get('v10', 0)**2)
                wind_kts = wind_ms * 1.944 
                pres_mb = env_data.get('msl', 101300) / 100.0 
                
                print(f"✅ [TrackModel] GFS OK: Gió={wind_kts:.1f}kt, Áp={pres_mb:.1f}mb")
                return env_data, wind_kts, pres_mb
                
            except Exception as e:
                print(f"⚠️ [TrackModel] GFS Retry {attempt+1}: {e}")
                if attempt == 2:
                    return {'u10': 0, 'v10': 0, 'msl': 101300}, 30.0, 1000.0
                continue

    def predict_track(self, lat, lon, current_month, current_hour):
        if self.global_storm_data is None: raise ValueError("Chưa tải dữ liệu IBTrACS")
        if not self.models: raise ValueError("Chưa tải mô hình GBDT")

        # 1. Analog & GFS
        analog = self.find_analog(lat, lon)
        env_data, gfs_wind, gfs_pres = self.get_gfs_data(lat, lon)
        
        # 2. Chuẩn bị dữ liệu (Shifting)
        d_lat_shift = lat - analog['LAT']
        d_lon_shift = lon - analog['LON']
        
        current_data = {
            'LAT': lat, 'LON': lon, 
            'WMO_WIND': gfs_wind, 'WMO_PRES': gfs_pres,
            'month': current_month, 'hour': current_hour
        }
        
        for h in [6, 12, 18, 24]:
            current_data[f'LAT_{h}h'] = analog[f'LAT_{h}h'] + d_lat_shift
            current_data[f'LON_{h}h'] = analog[f'LON_{h}h'] + d_lon_shift
            current_data[f'WMO_WIND_{h}h'] = gfs_wind
            current_data[f'WMO_PRES_{h}h'] = gfs_pres

        current_data['U_vel'] = current_data['LON'] - current_data['LON_6h']
        current_data['V_vel'] = current_data['LAT'] - current_data['LAT_6h']
        current_data['U_acc'] = current_data['U_vel'] - (current_data['LON_12h'] - current_data['LON_18h'])
        current_data['V_acc'] = current_data['V_vel'] - (current_data['LAT_12h'] - current_data['LAT_18h'])
        current_data.update(env_data)

        # 3. Dự báo với GBDT
        input_df = pd.DataFrame([current_data])
        if self.global_predictors:
            input_df = input_df.reindex(columns=self.global_predictors, fill_value=0)
            if 'U_acc' in input_df.columns: input_df['U_acc'] = 0.0
            if 'V_acc' in input_df.columns: input_df['V_acc'] = 0.0
            input_df = input_df.fillna(0)

        pred_lats, pred_lons = [], []
        horizons = [6, 12, 18, 24]
        
        for h in horizons:
            if h in self.models:
                dlon = self.models[h]['lon'].predict(input_df)[0]
                dlat = self.models[h]['lat'].predict(input_df)[0]
                pred_lats.append(lat + dlat)
                pred_lons.append(lon + dlon)
            else:
                pred_lats.append(lat)
                pred_lons.append(lon)
        
        # Smoothing 12h
        if len(pred_lats) >= 3:
            pred_lats[1] = (pred_lats[0] + pred_lats[2]) / 2
            pred_lons[1] = (pred_lons[0] + pred_lons[2]) / 2
            
        return pred_lats, pred_lons, gfs_wind, gfs_pres