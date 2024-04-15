import requests
import os

def get_weather_from_opendata(location):
    # 定義請求的 URL 和參數
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001"
    params = {
        "Authorization": os.getenv("API_KEY_CWA"),
        "format": "JSON",
        "StationId": location,
        "WeatherElement": "AirTemperature,RelativeHumidity"
    }

    # 發送 GET 請求
    response = requests.get(url, params=params, headers={"accept": "application/json"})

    # 檢查請求是否成功
    if response.status_code == 200:
        print("Request successful.")
        # 解析 JSON 響應內容
        data = response.json()

        # 提取站點氣象資料
        if data['success'] == 'true' and 'records' in data:
            station_data = data['records']['Station'][0]
            weather_elements = station_data['WeatherElement']

            # 提取氣溫和濕度
            air_temperature = weather_elements['AirTemperature']
            relative_humidity = weather_elements['RelativeHumidity']
            print(f"Air Temperature: {air_temperature}°C, Relative Humidity: {relative_humidity}%")
            return int(air_temperature), int(relative_humidity)
        else:
            print("No weather data available.")
            return None, None
    else:
        print("Request failed with status code:", response.status_code)
        return None, None

