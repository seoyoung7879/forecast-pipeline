import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

WIND_API_KEY = os.getenv("WIND_API_KEY")
SOLAR_API_KEY = os.getenv("SOLAR_API_KEY")

WIND_API_URL = "https://apis.data.go.kr/..."   # 실제 풍력 API 주소로 교체
SOLAR_API_URL = "https://apis.data.go.kr/..."  # 실제 태양광 API 주소로 교체


def fetch_solar():
    params = {
        "serviceKey": SOLAR_API_KEY,
        "numOfRows": 100,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": datetime.now().strftime("%Y%m%d"),
    }
    response = requests.get(SOLAR_API_URL, params=params)
    response.raise_for_status()
    return response.json()


def fetch_wind():
    params = {
        "serviceKey": WIND_API_KEY,
        "numOfRows": 100,
        "pageNo": 1,
        "dataType": "JSON",
        "base_date": datetime.now().strftime("%Y%m%d"),
    }
    response = requests.get(WIND_API_URL, params=params)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    today = datetime.now().strftime("%Y%m%d")
    os.makedirs("data", exist_ok=True)

    solar_data = fetch_solar()
    df_solar = pd.DataFrame(solar_data["response"]["body"]["items"])
    df_solar.to_csv(f"data/solar_{today}.csv", index=False)
    print(f"태양광 데이터 저장 완료: data/solar_{today}.csv")

    wind_data = fetch_wind()
    df_wind = pd.DataFrame(wind_data["response"]["body"]["items"])
    df_wind.to_csv(f"data/wind_{today}.csv", index=False)
    print(f"풍력 데이터 저장 완료: data/wind_{today}.csv")