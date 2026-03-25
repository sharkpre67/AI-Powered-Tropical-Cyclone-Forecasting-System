from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os
import sys
import warnings
import types
import sklearn.ensemble
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import KDTree 
# Import class từ file rời
from storm_model import StormPredictor 
from track_model import TrackPredictor 

warnings.filterwarnings('ignore')



# --- CẤU HÌNH ---
app = Flask(__name__)
CORS(app)
DEBUG = True 

# --- HÀM HỖ TRỢ (ĐƯA LÊN TRÊN CÙNG ĐỂ TRÁNH LỖI NOT DEFINED) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def calculate_movement_speed(lat1, lon1, lat2, lon2, h):
    dist = haversine(lat1, lon1, lat2, lon2)
    return dist / h if h > 0 else 0

def get_direction_name(lat1, lon1, lat2, lon2):
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    angle = (90 - np.degrees(np.arctan2(d_lon, d_lat))) % 360
    dirs = ['Bắc', 'Đông Bắc', 'Đông', 'Đông Nam', 'Nam', 'Tây Nam', 'Tây', 'Tây Bắc']
    return dirs[round(angle / 45) % 8]

def get_storm_nature(wind_speed):
    if wind_speed < 34: return 'TD'
    elif wind_speed < 48: return 'TS'
    elif wind_speed < 64: return 'STS'
    elif wind_speed < 100: return 'TY'
    else: return 'STY'

def prepare_cuongdo_features(data, model):
    exp = model.feature_names_in_ if hasattr(model,'feature_names_in_') else []
    base = {
        'year': datetime.now().year, 'month': data['month'], 'day': 1, 'hour': data['hour'],
        'lat': data['lat'], 'lon': data['lon'], 'land': 0,
        'vmax': data['wind'], 'pressure': data['pressure']
    }
    for n in ['TD', 'TS', 'STS', 'TY', 'STY']:
        base[f'nature_{n}'] = 1 if data['nature'] == n else 0
    df = pd.DataFrame([base])
    return df.reindex(columns=exp, fill_value=0).values[0]

# ===============================================================
#  KHỞI TẠO MÔ HÌNH (LOAD ONCE)
# ===============================================================
print("\n🔄 Đang khởi động hệ thống mô hình...")

# 1. Mô hình Xác suất
try:
    prob_model = StormPredictor("models/model_cnn_rnn.pth", "models/normalization_stats.json")
    print("✅ Mô hình Xác suất: OK")
except Exception as e:
    prob_model = None
    print(f"❌ Mô hình Xác suất Lỗi: {e}")

# 2. Mô hình Cường độ
try:
    cuongdo_model = joblib.load('models/forecast_6h_typhoon_model_ready.joblib')
    print("✅ Mô hình Cường độ: OK")
except Exception as e:
    cuongdo_model = None
    print(f"❌ Mô hình Cường độ Lỗi: {e}")

# 3. Mô hình Đường đi & GFS (TrackModel)
try:
    track_model = TrackPredictor(model_dir='models')
    print("✅ Mô hình Đường đi (TrackModel): OK")
except Exception as e:
    track_model = None
    print(f"❌ Mô hình Đường đi Lỗi: {e}")


# ===============================================================
#  FLASK ROUTES
# ===============================================================

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/status')
def status():
    return jsonify({
        'prob_model': prob_model is not None,
        'track_model': track_model.global_storm_data is not None if track_model else False,
        'intensity_model': cuongdo_model is not None
    })

# --- API MỚI: LẤY DỮ LIỆU GFS ---
@app.route('/get_gfs_data', methods=['POST'])
def get_gfs_data_api():
    try:
        data = request.json
        lat, lon = float(data['lat']), float(data['lon'])
        
        if not track_model:
            return jsonify({"success": False, "error": "TrackModel chưa tải xong"})

        # Gọi hàm từ track_model.py
        env_data, gfs_wind, gfs_pres = track_model.get_gfs_data(lat, lon)
        nature = get_storm_nature(gfs_wind)
        
        gfs_details = {
            "wind_speed": round(gfs_wind, 1),
            "pressure": round(gfs_pres, 1),
            "u_wind": round(env_data.get('u10', 0), 2),
            "v_wind": round(env_data.get('v10', 0), 2),
            "sst": round(env_data.get('sst', 273.15) - 273.15, 1),
            "tcwv": round(env_data.get('tcwv', 0), 1),
            "nature": nature
        }
        
        return jsonify({"success": True, "gfs": gfs_details})
    except Exception as e:
        print(f"Lỗi GFS API: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/predict_prob', methods=['POST'])
def predict_prob():
    try:
        data = request.json
        lat, lon = float(data['lat']), float(data['lon'])
        
        if prob_model is None:
             return jsonify({"success": False, "message": "Mô hình xác suất chưa tải."})

        prob = prob_model.predict_for_location(lon, lat)
        if prob < 0: return jsonify({"success": False, "message": "Lỗi dữ liệu GFS (Prob Model)"})
        
        # Gọi lại GFS để lấy info hiển thị (nếu chưa có)
        env_data, gfs_wind, gfs_pres = track_model.get_gfs_data(lat, lon)
        
        return jsonify({
            "success": True,
            "probability": round(prob * 100, 2),
            "alert": prob >= 0.5,
            "lat": lat, "lon": lon
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/predict_track', methods=['POST'])
def predict_track():
    try:
        d = request.json
        lat, lon = float(d['lat']), float(d['lon'])
        
        if not track_model or not track_model.global_storm_data is not None:
            return jsonify({'error': 'Hệ thống dự báo đường đi chưa sẵn sàng'}), 500

        # 1. Chạy mô hình đường đi
        pred_lats, pred_lons, gfs_wind, gfs_pres = track_model.predict_track(lat, lon, d['month'], d['hour'])
        
        # 2. Tổng hợp
        forecasts = []
        horizons = [6, 12, 18, 24]
        nature_gfs = get_storm_nature(gfs_wind)
        
        curr_data = {
            'lat': lat, 'lon': lon, 'wind': gfs_wind, 'pressure': gfs_pres,
            'month': d['month'], 'hour': d['hour'], 'nature': nature_gfs
        }

        for i, h in enumerate(horizons):
            p_lat = round(pred_lats[i], 4)
            p_lon = round(pred_lons[i], 4)
            
            # Dự báo cường độ
            if cuongdo_model:
                feats = prepare_cuongdo_features(curr_data, cuongdo_model)
                p_wind = cuongdo_model.predict([feats])[0]
            else:
                p_wind = gfs_wind
            
            p_wind = max(10, min(200, round(float(p_wind), 1)))
            p_pres = max(870, min(1020, round(gfs_pres - (p_wind - gfs_wind)*0.6, 1)))
            
            prev_lat = lat if i == 0 else pred_lats[i-1]
            prev_lon = lon if i == 0 else pred_lons[i-1]
            speed = calculate_movement_speed(prev_lat, prev_lon, p_lat, p_lon, 6)
            direction = get_direction_name(prev_lat, prev_lon, p_lat, p_lon)
            new_nature = get_storm_nature(p_wind)

            forecasts.append({
                'hours': h, 'lat': p_lat, 'lon': p_lon,
                'wind': p_wind, 'pressure': p_pres,
                'speed': round(speed, 2),
                'direction': direction,
                'nature': new_nature
            })
            
            curr_data.update({'lat': p_lat, 'lon': p_lon, 'wind': p_wind, 'pressure': p_pres, 'nature': new_nature})

        return jsonify({
            'current': {'lat': lat, 'lon': lon, 'wind': round(gfs_wind, 1), 'pressure': round(gfs_pres, 1), 'nature': nature_gfs},
            'forecasts': forecasts,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"❌ API Error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 SERVER READY (PORT 5000)")
    app.run(debug=DEBUG, port=5000, host='0.0.0.0')
