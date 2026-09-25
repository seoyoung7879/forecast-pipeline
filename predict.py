import os
import math
import joblib
import numpy as np
import pandas as pd
import requests
import pytz
import pvlib
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")

GRID_MAPPING_CSV = "격자예보_관측지점매핑.csv"
PLANT_DATA_CSV = "발전효율모델_최종데이터.csv"
INSOLATION_MODEL_PATH = "일사모델.joblib"
POWER_MODEL_PATH = "태양광_발전량_모델.joblib"
DB_PATH = "predictions.db"
API_URL = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

TARGET_PLANTS = {
    "무릉리":   {"1단계": "제주특별자치도", "2단계": "제주시", "3단계": "한경면"},
    "부산본부": {"1단계": "부산광역시",     "2단계": "중구",   "3단계": None},
}


def calculate_dew_point(temp_celsius, rh_percent):
    b, c = 17.62, 243.12
    rh_percent = max(rh_percent, 0.1)
    gamma = (b * temp_celsius / (c + temp_celsius)) + np.log(rh_percent / 100.0)
    return (c * gamma) / (b - gamma)


def get_latest_base_time():
    now = datetime.now()
    available_times = ['02', '05', '08', '11', '14', '17', '20', '23']
    current_hour = now.hour
    current_date = now.strftime('%Y%m%d')
    latest_time = None
    for t in reversed(available_times):
        if current_hour >= int(t):
            latest_time = t + '00'
            break
    if latest_time is None:
        yesterday = now - pd.Timedelta(days=1)
        current_date = yesterday.strftime('%Y%m%d')
        latest_time = '2300'
    return current_date, latest_time


def get_plant_location(grid_df, plant_name):
    info = TARGET_PLANTS[plant_name]
    q = grid_df[
        (grid_df['1단계'] == info['1단계']) &
        (grid_df['2단계'] == info['2단계'])
    ]
    if info['3단계']:
        q = q[q['3단계'] == info['3단계']]
    if q.empty:
        raise ValueError(f"[{plant_name}] 격자매핑에서 위치를 찾을 수 없습니다: {info}")
    row = q.iloc[0]
    return {
        'nx': int(row['격자 X']), 'ny': int(row['격자 Y']),
        '위도': float(row['위도']), '경도': float(row['경도']),
        '고도': float(row['노장해발고도(m)']),
    }


def get_plant_specs(plant_df, plant_name):
    row = plant_df[plant_df['발전구분'] == plant_name].iloc[-1]
    capacity_mw = float(row['설비용량(MW)']) if '설비용량(MW)' in row else None
    age_years = float(row['연식(년)']) if '연식(년)' in row else None
    return {'설비용량(MW)': capacity_mw, '연식(년)': age_years}


def get_weather_forecast(nx, ny, base_date, base_time):
    params = {
        'serviceKey': SOLAR_API_KEY, 'pageNo': '1', 'numOfRows': '1000',
        'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time,
        'nx': str(nx), 'ny': str(ny),
    }
    response = requests.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    if data['response']['header']['resultCode'] != '00':
        raise RuntimeError(f"API 오류: {data['response']['header']['resultMsg']}")
    return data['response']['body']['items'].get('item', [])


def process_weather_data(items, location_info):
    df = pd.DataFrame(items)
    df['fcstValue'] = pd.to_numeric(df['fcstValue'], errors='coerce')
    pivot_df = df.pivot_table(index=['fcstDate', 'fcstTime'], columns='category', values='fcstValue').reset_index()
    rename_dict = {'TMP': '기온(°C)', 'PCP': '강수량(mm)', 'REH': '습도(%)', 'WSD': '풍속(m/s)', 'SNO': '적설(cm)', 'SKY': '하늘상태'}
    pivot_df.rename(columns=rename_dict, inplace=True)
    pivot_df.fillna(0, inplace=True)

    seoul_tz = pytz.timezone('Asia/Seoul')
    pivot_df['datetime'] = pd.to_datetime(pivot_df['fcstDate'] + pivot_df['fcstTime'], format='%Y%m%d%H%M').dt.tz_localize(seoul_tz)

    pivot_df['month'] = pivot_df['datetime'].dt.month
    pivot_df['hour'] = pivot_df['datetime'].dt.hour
    pivot_df['hour_sin'] = np.sin(2 * np.pi * pivot_df['hour'] / 24)
    pivot_df['hour_cos'] = np.cos(2 * np.pi * pivot_df['hour'] / 24)
    pivot_df['days_in_month'] = pivot_df['datetime'].dt.days_in_month
    pivot_df['month_day'] = pivot_df['datetime'].dt.month + (pivot_df['datetime'].dt.day - 1) / pivot_df['days_in_month']
    pivot_df['month_day_sin'] = np.sin(2 * np.pi * pivot_df['month_day'] / 12)
    pivot_df['month_day_cos'] = np.cos(2 * np.pi * pivot_df['month_day'] / 12)
    pivot_df['이슬점'] = pivot_df.apply(lambda r: calculate_dew_point(r['기온(°C)'], r['습도(%)']), axis=1)
    pivot_df['T-Td'] = pivot_df['기온(°C)'] - pivot_df['이슬점']

    for k, v in location_info.items():
        pivot_df[k] = v
    return pivot_df


def add_solar_position_features(df):
    loc = pvlib.location.Location(latitude=df['위도'].iloc[0], longitude=df['경도'].iloc[0], tz='Asia/Seoul', altitude=df['고도'].iloc[0])
    solar_positions = loc.get_solarposition(times=df['datetime'])
    df['태양고도'] = solar_positions['apparent_elevation'].values
    df['방위각'] = solar_positions['azimuth'].values
    return df


def predict_insolation(df):
    model_dict = joblib.load(INSOLATION_MODEL_PATH)
    classifier, regressor = model_dict['classifier'], model_dict['regressor']

    feature_cols = [
        'month_day_sin', 'month_day_cos', 'hour_sin', 'hour_cos',
        '태양고도', '방위각', '기온(°C)', '풍속(m/s)', '습도(%)',
        '강수량(mm)', '하늘상태', 'T-Td'
    ]
    input_df = df[feature_cols].copy()
    input_df['하늘상태'] = input_df['하늘상태'].astype('category')

    is_zero = classifier.predict(input_df)
    final_pred = np.zeros(len(input_df))
    idx_nonzero = np.where(is_zero == 0)[0]
    if len(idx_nonzero) > 0:
        pred = regressor.predict(input_df.iloc[idx_nonzero])
        final_pred[idx_nonzero] = np.clip(pred, 0, None)

    df['일사(MJ/m2)'] = final_pred
    return df


def predict_power_generation(df, extra_features):
    model_dict = joblib.load(POWER_MODEL_PATH)
    classifier, regressor, feature_cols = model_dict['classifier'], model_dict['regressor'], model_dict['features']

    for k, v in extra_features.items():
        df[k] = v

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"발전량 모델 필요 피처 누락: {missing}")

    input_df = df[feature_cols].copy()
    input_df['하늘상태'] = input_df['하늘상태'].astype('int')

    is_zero = classifier.predict(input_df)
    final_eff = np.zeros(len(input_df))
    idx_nonzero = np.where(is_zero == 0)[0]
    if len(idx_nonzero) > 0:
        pred_log = regressor.predict(input_df.iloc[idx_nonzero])
        final_eff[idx_nonzero] = np.clip(np.expm1(pred_log), 0, None)
    return final_eff


def run_plant_prediction(plant_name, grid_df, plant_df, base_date, base_time):
    print(f"\n=== {plant_name} 예측 시작 ===")
    location_info = get_plant_location(grid_df, plant_name)
    specs = get_plant_specs(plant_df, plant_name)

    items = get_weather_forecast(location_info['nx'], location_info['ny'], base_date, base_time)
    if not items:
        print(f"[{plant_name}] 기상 데이터 없음, 스킵")
        return None

    df = process_weather_data(items, location_info)
    df = add_solar_position_features(df)
    df = predict_insolation(df)

    eff_pred = predict_power_generation(df, {'연식(년)': specs['연식(년)']})
    capacity_mw = specs['설비용량(MW)'] or 0

    df['예측효율'] = eff_pred
    df['예측발전량(kWh)'] = eff_pred * capacity_mw * 1000
    df['발전구분'] = plant_name
    df['예측기준일시'] = f"{base_date}{base_time}"
    df['datetime'] = df['datetime'].dt.tz_localize(None)

    return df[['발전구분', 'datetime', '예측기준일시', '기온(°C)', '습도(%)', '태양고도',
               '일사(MJ/m2)', '예측효율', '예측발전량(kWh)']]


def save_to_sqlite(df, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    df.to_sql("predictions", conn, if_exists="append", index=False)
    conn.close()


if __name__ == "__main__":
    grid_df = pd.read_csv(GRID_MAPPING_CSV)
    plant_df = pd.read_csv(PLANT_DATA_CSV, encoding='utf-8')

    base_date, base_time = get_latest_base_time()
    print(f"기준일시: {base_date} {base_time}")

    all_results = []
    for plant_name in TARGET_PLANTS.keys():
        try:
            result_df = run_plant_prediction(plant_name, grid_df, plant_df, base_date, base_time)
            if result_df is not None:
                all_results.append(result_df)
        except Exception as e:
            print(f"❌ [{plant_name}] 예측 실패: {e}")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        save_to_sqlite(final_df)
        print(f"\n✅ 총 {len(final_df)}행 저장 완료: {DB_PATH}")
    else:
        print("❌ 저장할 결과가 없습니다.")