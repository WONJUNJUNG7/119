import os
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
try:
    import folium
    FOLIUM_AVAILABLE = True
except ModuleNotFoundError:
    folium = None
    FOLIUM_AVAILABLE = False
import streamlit.components.v1 as components

# folium 출력을 위한 helper (streamlit-folium이 없을 경우를 대비한 fallback)
try:
    from streamlit_folium import st_folium
except ImportError:
    st_folium = None

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="전국 시·도 소방안전 취약도 및 소화전 인프라 최적화 시스템",
    page_icon="🚒",
    layout="wide"
)

# ==========================================
# 1. 시·도 기준 더미데이터 생성
# ==========================================
@st.cache_data
def load_sido_hydrant_data():
    """
    컬럼 정의
    - 시도명: 17개 광역 행정구역
    - 면적_B: 행정구역 면적 (km²)
    - 소화전개소_A: 소화전 수
    - 화재발생건수_C: 연평균 화재 건수
    - latitude / longitude: 지도 시각화용 중심 좌표
    """
    sido_raw = {
        "서울특별시":     {"lat": 37.5665, "lon": 126.9780, "area":   605.2, "hydrants": 61200, "fires": 5400},
        "부산광역시":     {"lat": 35.1796, "lon": 129.0756, "area":   770.1, "hydrants": 23500, "fires": 2500},
        "대구광역시":     {"lat": 35.8714, "lon": 128.6014, "area":   883.7, "hydrants": 18200, "fires": 1200},
        "인천광역시":     {"lat": 37.4563, "lon": 126.7052, "area":  1066.4, "hydrants": 19500, "fires": 1400},
        "광주광역시":     {"lat": 35.1595, "lon": 126.8526, "area":   501.2, "hydrants":  9800, "fires":  850},
        "대전광역시":     {"lat": 36.3504, "lon": 127.3845, "area":   539.8, "hydrants": 10500, "fires":  900},
        "울산광역시":     {"lat": 35.5389, "lon": 129.3114, "area":  1062.0, "hydrants":  7200, "fires":  800},
        "세종특별자치시": {"lat": 36.4801, "lon": 127.2890, "area":   464.9, "hydrants":  3200, "fires":  250},
        "경기도":         {"lat": 37.4138, "lon": 127.5183, "area": 10195.0, "hydrants": 98000, "fires": 8600},
        "강원특별자치도": {"lat": 37.8228, "lon": 128.1555, "area": 16828.1, "hydrants": 18500, "fires": 1900},
        "충청북도":       {"lat": 36.8836, "lon": 127.7915, "area":  7407.9, "hydrants": 14200, "fires": 1400},
        "충청남도":       {"lat": 36.7171, "lon": 126.8018, "area":  8246.8, "hydrants": 21000, "fires": 2100},
        "전라북도":       {"lat": 35.7175, "lon": 127.1530, "area":  8072.1, "hydrants": 16500, "fires": 1900},
        "전라남도":       {"lat": 34.8679, "lon": 126.9910, "area": 12359.0, "hydrants": 19000, "fires": 2600},
        "경상북도":       {"lat": 36.3308, "lon": 128.7001, "area": 19034.0, "hydrants": 28000, "fires": 2900},
        "경상남도":       {"lat": 35.3401, "lon": 128.2618, "area": 10541.1, "hydrants": 29500, "fires": 2800},
        "제주특별자치도": {"lat": 33.4996, "lon": 126.5312, "area":  1850.2, "hydrants":  6800, "fires":  600},
    }

    rows = []
    for sido, info in sido_raw.items():
        rows.append({
            "시도명":        sido,
            "latitude":      info["lat"],
            "longitude":     info["lon"],
            "면적_B":        info["area"],
            "소화전개소_A":  info["hydrants"],
            "화재발생건수_C": info["fires"],
        })

    df = pd.DataFrame(rows)

    # 소화전 밀도 (개/km²)
    df["소화전_밀도_D"] = (df["소화전개소_A"] / df["면적_B"]).round(2)

    # 화재 밀도 (건/km²)
    df["화재_발생_밀도_E"] = (df["화재발생건수_C"] / df["면적_B"]).round(3)

    # NFDS 기반 화재 대응 취약 지수 (가중치 모델)
    # 가중치: 화재 위험도(0.6) + 인프라 결핍도(0.4)
    df["취약_지수"] = (
        (df["화재_발생_밀도_E"] * 0.6) + 
        ((1 / df["소화전_밀도_D"].replace(0, np.nan)).fillna(0) * 0.4)
    ).round(2)  # 소수점 2자리로 계산

    # 법적 기준 미달율 계산: 면적 대비 소화전 수가 적을수록 값이 커짐
    # 시각화 안정성을 위해 최소 8.5%, 최대 89.2%로 범위를 제한(clip)함
    df["법적기준_미달율"] = (
        (df["면적_B"] / df["소화전개소_A"].replace(0, np.nan)) * 300 + 12
    ).fillna(89.2).clip(lower=8.5, upper=89.2).round(1) 

    df["시도명_full"] = df["시도명"]

    return df


def standardize_sido_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 기존 유연한 매핑 로직 유지
    mapping = {
        "면적": "면적_B", "총면적_km2": "면적_B", "총면적_m2": "면적_B_m2",
        "화재건수": "화재발생건수_C", "소화전_개수": "소화전개소_A", "소화전 개수": "소화전개소_A"
    }
    for src, dst in mapping.items():
        if src in df.columns and dst not in df.columns:
            if dst == "면적_B_m2":
                df["면적_B"] = df[src] / 1_000_000
            else:
                df[dst] = df[src]

    if "면적_B" in df.columns and "소화전개소_A" in df.columns:
        df["소화전_밀도_D"] = (df["소화전개소_A"] / df["면적_B"]).astype(float)
    if "면적_B" in df.columns and "화재발생건수_C" in df.columns:
        df["화재_발생_밀도_E"] = (df["화재발생건수_C"] / df["면적_B"]).astype(float)
    
    return df


def normalize_coord_columns(df):
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"latitude", "lat", "위도", "y", "위도(도)"}:
            rename[col] = "latitude"
        if key in {"longitude", "lon", "lng", "경도", "경도(도)"}:
            rename[col] = "longitude"
    return df.rename(columns=rename)


@st.cache_data
def load_and_merge_new_data():
    """제공된 3개의 CSV 파일을 로드하여 통합 데이터프레임을 생성"""
    try:
        # 1. 데이터 로드 (BOM 대응을 위해 utf-8-sig 사용)
        df_area = pd.read_csv('전국면적.csv', encoding='utf-8-sig')
        df_fire = pd.read_csv('화재발생.csv', encoding='utf-8-sig')
        df_hydrants_raw = pd.read_csv('소화전.csv', encoding='utf-8-sig') if os.path.exists('소화전.csv') else None
        
        # 2. 행정구역 데이터 전처리 (시·도 단위로 집계)
        df_area.columns = [c.replace('\ufeff', '') for c in df_area.columns]
        df_area.rename(columns={'소재지(시군구)별': '지역명_raw', '2025': '면적_B'}, inplace=True)

        sidos = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", 
                 "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도", 
                 "전라남도", "경상북도", "경상남도", "제주특별자치도"]
        
        # 면적 데이터를 시도별로 집계
        area_list = []
        curr_sido = None
        for _, row in df_area.iterrows():
            name = str(row['지역명_raw']).strip()
            if name in sidos:
                curr_sido = name
                area_list.append({'시도명': curr_sido, '면적_B': row['면적_B']})
        df_area_processed = pd.DataFrame(area_list)

        # 화재 데이터 전처리 (시도별로 집계)
        df_fire.rename(columns={'행정구역별(1)': '시도명', '행정구역별(2)': '시군구명', '2025': '화재발생건수_C'}, inplace=True)
        # 소계 행만 사용하여 시도별 합계 가져오기
        df_fire_sido = df_fire[df_fire['시군구명'] == '소계'].copy()
        df_fire_sido = df_fire_sido[['시도명', '화재발생건수_C']].copy()
        
        # 3. 소화전 데이터 집계 (주소 분석을 통한 시도 단위 카운트)
        if df_hydrants_raw is not None:
            addr_cols = ['시도명', '시도', '지역', '주소', 'location', '소재지', '설치위치', '설치장소', '도로명주소', '지번주소', '소재지도로명주소', '설치장소']
            target_col = next((c for c in addr_cols if c in df_hydrants_raw.columns), None)
            
            if target_col:
                sido_map = {
                    "서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시",
                    "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
                    "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
                    "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도", "제주": "제주특별자치도"
                }
                def parse_addr(val):
                    if not isinstance(val, str) or not val.strip(): return "기타"
                    val = val.strip()
                    
                    # 시도명 찾기 (긴 명칭 우선 매칭으로 정확도 향상)
                    sido = "기타"
                    matched_prefix = ""
                    for s in sidos:
                        if val.startswith(s):
                            sido = s
                            matched_prefix = s
                            break
                    if sido == "기타":
                        for short, full in sido_map.items():
                            if val.startswith(short):
                                sido = full
                                matched_prefix = short
                                break
                    return sido

                # 성능 최적화: 시도별로만 파싱
                parsed_results = df_hydrants_raw[target_col].apply(parse_addr)
                df_hydrants_raw['시도명_std'] = parsed_results
                df_h_count = df_hydrants_raw.groupby('시도명_std').size().reset_index(name='소화전개소_A')
                
                # 위경도 데이터가 있다면 시도별 평균 좌표 계산
                hydrants_mapped = normalize_coord_columns(df_hydrants_raw)
                if 'latitude' in hydrants_mapped.columns and 'longitude' in hydrants_mapped.columns:
                    coords_agg = hydrants_mapped.groupby('시도명_std')[['latitude', 'longitude']].mean().reset_index()
                    df_h_count = pd.merge(df_h_count, coords_agg, on='시도명_std', how='left')
                
                df_h_count.rename(columns={'시도명_std': '시도명'}, inplace=True)
            else:
                df_h_count = pd.DataFrame(columns=['시도명', '소화전개소_A'])
        else:
            df_h_count = pd.DataFrame(columns=['시도명', '소화전개소_A'])

        # 4. 데이터 병합 (시도명 기준)
        merged = pd.merge(df_area_processed, df_fire_sido, on='시도명', how='inner')
        merged = pd.merge(merged, df_h_count, on='시도명', how='left')
        
        # 데이터 매칭 확인을 위한 알림
        zero_count = merged['소화전개소_A'].isna().sum()
        if zero_count > 0:
            st.sidebar.info(f"ℹ️ {zero_count}개 지역은 소화전 데이터가 없거나 주소 형식이 다릅니다.")

        merged['소화전개소_A'] = merged['소화전개소_A'].fillna(0)
        
        # 표시용 지역명 생성
        merged['시도명_full'] = merged['시도명']
        
        return merged, df_hydrants_raw
    except Exception as e:
        st.error(f"데이터 파일 로드 중 오류 발생: {e}")
        return None, None

# 신규 데이터 로드 시도
if os.path.exists('전국면적.csv') and os.path.exists('화재발생.csv') and os.path.exists('소화전.csv'):
    df_sido, hydrant_points_raw = load_and_merge_new_data()
    if df_sido is not None:
        skip_processing = False
        df_sido = standardize_sido_columns(df_sido)
        # 개 개별 좌표 데이터 정규화
        hydrant_points = normalize_coord_columns(hydrant_points_raw) if hydrant_points_raw is not None else None
    else:
        st.info("신규 데이터를 통합하는 데 실패하여 시뮬레이션 데이터를 사용합니다.")
        df_sido = load_sido_hydrant_data()
        skip_processing = True
        hydrant_points = None
else:
    st.info("제공된 3개의 데이터 파일(전국면적, 소화전, 화재발생) 중 일부를 찾을 수 없어 시뮬레이션 데이터를 로드합니다.")
    df_sido = load_sido_hydrant_data()
    skip_processing = True
    hydrant_points = None

# CSV 컬럼을 앱에서 사용하는 컬럼명으로 맞춥니다.
column_map = {
    "화재건수": "화재발생건수_C",
    "소화전 개수": "소화전개소_A",
    "면적": "면적_B",
    "소화전 밀도 (개/면적)": "소화전_밀도_D",
    "화재 발생률 (건/면적)": "화재_발생_밀도_E",
    "행정구역": "시도명",
}
if not skip_processing:
    for src, dst in column_map.items():
        if src in df_sido.columns and dst not in df_sido.columns:
            df_sido[dst] = df_sido[src]

    # 필요 시 밀도 값을 재계산합니다.
    if "소화전_밀도_D" not in df_sido.columns and "소화전 개수" in df_sido.columns and "면적_B" in df_sido.columns:
        df_sido["소화전_밀도_D"] = (df_sido["소화전개소_A"] / df_sido["면적_B"]).round(2)
    if "화재_발생_밀도_E" not in df_sido.columns and "화재건수" in df_sido.columns and "면적_B" in df_sido.columns:
        df_sido["화재_발생_밀도_E"] = (df_sido["화재건수"] / df_sido["면적_B"]).round(9)

    # 평가 지표가 없으면 생성합니다.
    if "취약_지수" not in df_sido.columns:
        if "화재_발생_밀도_E" in df_sido.columns and "소화전_밀도_D" in df_sido.columns:
            df_sido["취약_지수"] = (
                (df_sido["화재_발생_밀도_E"] * 0.6) + 
                ((1 / df_sido["소화전_밀도_D"].replace(0, np.nan)).fillna(0) * 0.4)
            ).round(2)
        else:
            df_sido["취약_지수"] = 50.0

    # 유한하지 않은 값을 처리합니다.
    df_sido["취약_지수"] = df_sido["취약_지수"].replace([np.inf, -np.inf], np.nan)
    if df_sido["취약_지수"].isna().any():
        df_sido["취약_지수"] = df_sido["취약_지수"].fillna(50.0)

    if "법적기준_미달율" not in df_sido.columns:
        df_sido["법적기준_미달율"] = (
            (df_sido["면적_B"] / df_sido["소화전개소_A"].replace(0, np.nan)) * 300 + 12
        ).fillna(89.2).clip(lower=8.5, upper=89.2).round(1)

# 지도 표시를 위해 광역 단위 좌표를 부여합니다.
province_coords = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5389, 129.3114),
    "세종": (36.4801, 127.2890),
    "경기": (37.4138, 127.5183),
    "강원": (37.8228, 128.1555),
    "충북": (36.8836, 127.7915),
    "충남": (36.7171, 126.8018),
    "전북": (35.7175, 127.1530),
    "전남": (34.8679, 126.9910),
    "경북": (36.3308, 128.7001),
    "경남": (35.3401, 128.2618),
    "제주": (33.4996, 126.5312),
    "전라북도": (35.7175, 127.1530),
    "전라남도": (34.8679, 126.9910),
    "경기도": (37.4138, 127.5183),
    "충청북도": (36.8836, 127.7915),
    "충청남도": (36.7171, 126.8018),
    "경상북도": (36.3308, 128.7001),
    "경상남도": (35.3401, 128.2618),
    "제주특별자치도": (33.4996, 126.5312),
    "세종특별자치시": (36.4801, 127.2890),
    "강원특별자치도": (37.8228, 128.1555),
    "울산광역시": (35.5389, 129.3114),
    "광주광역시": (35.1595, 126.8526),
    "대전광역시": (36.3504, 127.3845),
    "부산광역시": (35.1796, 129.0756),
    "대구광역시": (35.8714, 128.6014),
    "인천광역시": (37.4563, 126.7052),
    "서울특별시": (37.5665, 126.9780),
}

def normalize_region_name(name: str):
    if not isinstance(name, str):
        return ""
    text = name.strip().replace("\u3000", " ")
    text = " ".join(text.split())
    # 시군구 단위에서는 "경기도 수원시" 형태로 들어옴
    if " " in text:
        parts = text.split()
        sido_prefix = parts[0][:2]
        # 시도명 매핑 찾기
        for key, val in province_coords.items():
            if key.startswith(sido_prefix):
                parts[0] = key
                break
        return " ".join(parts)
    return text

if "latitude" not in df_sido.columns or "longitude" not in df_sido.columns:
    df_sido["시도명"] = df_sido["시도명"].astype(str).apply(normalize_region_name)
    df_sido["province_prefix"] = df_sido["시도명"].apply(
        lambda name: next((key for key in province_coords if name.startswith(key) or key in name), None)
    )
    df_sido["latitude"] = df_sido["province_prefix"].map(lambda p: province_coords.get(p, (None, None))[0])
    df_sido["longitude"] = df_sido["province_prefix"].map(lambda p: province_coords.get(p, (None, None))[1])
    if df_sido["latitude"].isna().any() or df_sido["longitude"].isna().any():
        st.warning("일부 행정구역에 대해 좌표를 찾지 못했습니다. 지도 시각화는 제한될 수 있습니다.")

# --- 시각화 함수 정의 ---
def render_vulnerability_map(data):
    """취약 지수에 따른 시각화 지도 렌더링"""
    # 취약 지수에 따른 색상 지정 (높을수록 붉은색)
    data['color_r'] = data['취약_지수'].apply(lambda x: min(255, int(x * 5)))
    data['color_g'] = data['취약_지수'].apply(lambda x: max(0, 255 - int(x * 5)))
    
    view_state = pdk.ViewState(
        latitude=36.5, longitude=127.5, zoom=6, pitch=45
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position=["longitude", "latitude"],
        get_color="[color_r, color_g, 100, 160]",
        get_radius="취약_지수 * 500",
        pickable=True,
    )
    
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "{시도명}\n취약 지수: {취약_지수}\n소화전 밀도: {소화전_밀도_D}"}
    )
    st.pydeck_chart(r)


def haversine_distance(lat1, lon1, lat2, lon2):
    """두 지점 간 대략적 거리(km)를 계산합니다."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def classify_operational_area(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "취약_지수" not in df.columns:
        return df

    high_vuln = np.percentile(df["취약_지수"].dropna(), 75)
    high_fire = np.percentile(df["화재_발생_밀도_E"].dropna(), 70)
    low_hydrant = np.percentile(df["소화전_밀도_D"].dropna(), 30)

    def _label(row):
        if row["취약_지수"] >= high_vuln or (
            row["화재_발생_밀도_E"] >= high_fire and row["소화전_밀도_D"] <= low_hydrant
        ):
            return "A. 핵심 출동권역"
        if row["취약_지수"] >= np.percentile(df["취약_지수"].dropna(), 50):
            return "B. 집중 감시권역"
        return "C. 일반 운영권역"

    df["출동_우선등급"] = df.apply(_label, axis=1)
    return df


def compute_nearest_hydrant_info(df_regions: pd.DataFrame, hydrants: pd.DataFrame) -> pd.DataFrame:
    if hydrants is None or hydrants.empty:
        return pd.DataFrame()

    hydrants = hydrants.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if hydrants.empty:
        return pd.DataFrame()

    rows = []
    for _, region in df_regions.dropna(subset=["latitude", "longitude"]).iterrows():
        target_lat = region["latitude"]
        target_lon = region["longitude"]

        distances = haversine_distance(
            target_lat,
            target_lon,
            hydrants["latitude"].to_numpy(dtype=float),
            hydrants["longitude"].to_numpy(dtype=float),
        )
        closest_idx = int(np.argmin(distances))
        rows.append({
            "시도명": region["시도명"],
            "시도명_full": region.get("시도명_full", region["시도명"]),
            "latitude": target_lat,
            "longitude": target_lon,
            "취약_지수": region.get("취약_지수", np.nan),
            "출동_우선등급": region.get("출동_우선등급", ""),
            "nearest_hydrant_km": float(distances[closest_idx].round(3)),
            "nearest_hydrant_lat": float(hydrants.loc[closest_idx, "latitude"]),
            "nearest_hydrant_lon": float(hydrants.loc[closest_idx, "longitude"]),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Helper functions for patrol route planning and region coordinates
# ---------------------------------------------------------------------------

def convert_numeric_route_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["취약_지수"] = pd.to_numeric(df["취약_지수"], errors="coerce")
    if "nearest_hydrant_km" not in df.columns:
        df["nearest_hydrant_km"] = np.nan
    else:
        df["nearest_hydrant_km"] = pd.to_numeric(df["nearest_hydrant_km"], errors="coerce")
    return df


def get_region_coordinates(region_name: str):
    if not isinstance(region_name, str):
        return None

    fallback = {
        "서울": (37.5665, 126.9780),
        "부산": (35.1796, 129.0756),
        "대구": (35.8714, 128.6014),
        "인천": (37.4563, 126.7052),
        "광주": (35.1595, 126.8526),
        "대전": (36.3504, 127.3845),
        "울산": (35.5389, 129.3114),
        "세종": (36.4801, 127.2890),
        "경기": (37.4138, 127.5183),
        "강원": (37.8228, 128.1555),
        "충북": (36.8836, 127.7915),
        "충남": (36.7171, 126.8018),
        "전북": (35.7175, 127.1530),
        "전남": (34.8679, 126.9910),
        "경북": (36.3308, 128.7001),
        "경남": (35.3401, 128.2618),
        "제주": (33.4996, 126.5312),
    }
    for key, coords in fallback.items():
        if key in region_name:
            return coords
    return None


def build_fire_patrol_route(df: pd.DataFrame):
    if not FOLIUM_AVAILABLE:
        return None, df["취약_지수"].mean() if "취약_지수" in df.columns else 0.0, df

    df = convert_numeric_route_columns(df)
    mean_vuln = df["취약_지수"].mean()

    route_df = df[df["취약_지수"] > mean_vuln].copy()
    if route_df.empty:
        return None, mean_vuln, route_df

    route_df = route_df.sort_values(by="nearest_hydrant_km", ascending=False).reset_index(drop=True)

    route_points = []
    for _, row in route_df.iterrows():
        # Prefer explicit latitude/longitude present in the navigation dataframe
        lat = None
        lon = None
        if "latitude" in row.index and not pd.isna(row.get("latitude")):
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
            except Exception:
                lat = None
                lon = None
        # fall back to nearest_hydrant coords if available
        if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and not pd.isna(row.get("nearest_hydrant_lat")):
            try:
                lat = float(row.get("nearest_hydrant_lat"))
                lon = float(row.get("nearest_hydrant_lon"))
            except Exception:
                lat = None
                lon = None

        # lastly, attempt to resolve using province-level fallback mapping
        if lat is None or lon is None:
            coords = get_region_coordinates(row["시도명"]) if "시도명" in row.index else None
            if coords is not None:
                lat, lon = coords[0], coords[1]

        if lat is None or lon is None:
            # skip entries we cannot geolocate
            continue

        route_points.append({
            "시도명": row["시도명"],
            "취약_지수": row["취약_지수"],
            "nearest_hydrant_km": row["nearest_hydrant_km"],
            "latitude": lat,
            "longitude": lon,
        })

    if not route_points:
        return None, mean_vuln, route_df

    first_point = route_points[0]
    route_map = folium.Map(location=[first_point["latitude"], first_point["longitude"]], zoom_start=7)

    path = []
    for idx, point in enumerate(route_points, start=1):
        location = [point["latitude"], point["longitude"]]
        popup_html = (
            f"<strong>{point['시도명']}</strong><br>"
            f"취약 지수: {point['취약_지수']:.2f}<br>"
            f"nearest_hydrant_km: {point['nearest_hydrant_km']:.3f} km"
        )
        folium.Marker(
            location=location,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{idx}. {point['시도명']}",
            icon=folium.Icon(color="red" if idx == 1 else "blue"),
        ).add_to(route_map)
        path.append(location)

    folium.PolyLine(path, color="red", weight=5, opacity=0.8).add_to(route_map)
    return route_map, mean_vuln, route_df


def build_phase3_route_map(route_points_df: pd.DataFrame):
    if not FOLIUM_AVAILABLE:
        return None, route_points_df

    route_points_df = route_points_df.copy()
    route_points_df = convert_numeric_route_columns(route_points_df)

    path = []
    for _, row in route_points_df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            lat = row.get("nearest_hydrant_lat")
            lon = row.get("nearest_hydrant_lon")

        if pd.isna(lat) or pd.isna(lon):
            coords = get_region_coordinates(str(row.get("시도명", "")))
            if coords is not None:
                lat, lon = coords

        if pd.isna(lat) or pd.isna(lon):
            continue

        path.append((float(lat), float(lon)))

    if not path:
        return None, route_points_df

    route_map = folium.Map(location=path[0], zoom_start=7)
    for idx, (lat, lon) in enumerate(path, start=1):
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                html=f"""
                    <div style=\"width:32px;height:32px;border-radius:50%;background:#d62728;color:white;font-weight:bold;text-align:center;line-height:32px;border:2px solid white;box-shadow:0 0 4px rgba(0,0,0,0.25);\">{idx}</div>
                """
            ),
        ).add_to(route_map)

    folium.PolyLine(path, color="red", weight=4, opacity=0.8).add_to(route_map)
    return route_map, route_points_df


# ---------------------------------------------------------------------------
# 시도 내 시군구 단위 집계 및 순찰 경로 생성
# ---------------------------------------------------------------------------

# 시군구별 대표 좌표 (주요 경기도 시군구 포함 전국)
_SIGUNGU_COORDS = {
    # 경기도
    "수원시": (37.2636, 127.0286), "성남시": (37.4449, 127.1388), "의정부시": (37.7382, 127.0337),
    "안양시": (37.3943, 126.9568), "부천시": (37.5034, 126.7660), "광명시": (37.4784, 126.8644),
    "평택시": (36.9921, 127.1128), "동두천시": (37.9037, 127.0607), "안산시": (37.3236, 126.8219),
    "고양시": (37.6584, 126.8320), "과천시": (37.4292, 126.9879), "구리시": (37.5942, 127.1296),
    "남양주시": (37.6360, 127.2162), "오산시": (37.1498, 127.0775), "시흥시": (37.3800, 126.8029),
    "군포시": (37.3614, 126.9350), "의왕시": (37.3448, 126.9688), "하남시": (37.5395, 127.2147),
    "용인시": (37.2411, 127.1776), "파주시": (37.7600, 126.7797), "이천시": (37.2792, 127.4428),
    "안성시": (37.0078, 127.2797), "김포시": (37.6154, 126.7157), "화성시": (37.1994, 126.8313),
    "광주시": (37.4296, 127.2553), "양주시": (37.7854, 127.0459), "포천시": (37.8947, 127.2003),
    "여주시": (37.2983, 127.6375), "연천군": (38.0961, 127.0749), "가평군": (37.8316, 127.5112),
    "양평군": (37.4916, 127.4875),
    # 서울
    "종로구": (37.5730, 126.9794), "중구": (37.5638, 126.9976), "용산구": (37.5326, 126.9903),
    "성동구": (37.5636, 127.0364), "광진구": (37.5388, 127.0824), "동대문구": (37.5744, 127.0396),
    "중랑구": (37.6065, 127.0927), "성북구": (37.5894, 127.0167), "강북구": (37.6396, 127.0255),
    "도봉구": (37.6688, 127.0471), "노원구": (37.6543, 127.0568), "은평구": (37.6176, 126.9227),
    "서대문구": (37.5791, 126.9368), "마포구": (37.5638, 126.9084), "양천구": (37.5170, 126.8665),
    "강서구": (37.5509, 126.8496), "구로구": (37.4954, 126.8876), "금천구": (37.4569, 126.8955),
    "영등포구": (37.5263, 126.8963), "동작구": (37.5124, 126.9393), "관악구": (37.4784, 126.9516),
    "서초구": (37.4837, 127.0325), "강남구": (37.5172, 127.0474), "송파구": (37.5145, 127.1059),
    "강동구": (37.5301, 127.1238),
    # 부산
    "중구": (35.1032, 129.0326), "서구": (35.0987, 129.0253), "동구": (35.1358, 129.0589),
    "영도구": (35.0876, 129.0680), "부산진구": (35.1636, 129.0531), "동래구": (35.2057, 129.0842),
    "남구": (35.1361, 129.0843), "북구": (35.2031, 128.9909), "해운대구": (35.1631, 129.1635),
    "사하구": (35.1041, 128.9741), "금정구": (35.2430, 129.0900), "강서구": (35.2112, 128.9801),
    "연제구": (35.1766, 129.0802), "수영구": (35.1456, 129.1134), "사상구": (35.1498, 128.9924),
    "기장군": (35.2445, 129.2218),
}


@st.cache_data
def build_sigungu_route_for_sido(sido_name: str, hydrant_raw_df):
    """
    선택된 시도(sido_name) 내의 시/군/구 단위로 소화전 데이터를 집계하여
    취약도 기반 순찰 경로 DataFrame을 반환합니다.
    """
    if hydrant_raw_df is None or hydrant_raw_df.empty:
        return pd.DataFrame()

    addr_col = hydrant_raw_df.columns[0]  # 첫 번째 컬럼 = 도로명주소
    lat_col = next((c for c in hydrant_raw_df.columns if str(c).lower() in {"latitude","lat","위도"}), None)
    lon_col = next((c for c in hydrant_raw_df.columns if str(c).lower() in {"longitude","lon","lng","경도"}), None)

    # 시도 접두어로 필터링 (예: "경기" → "경기도 ..." 행)
    sido_prefix = sido_name[:2]
    mask = hydrant_raw_df[addr_col].astype(str).str.startswith(sido_prefix)
    sido_df = hydrant_raw_df[mask].copy()

    if sido_df.empty:
        return pd.DataFrame()

    # 알려진 시군구 명칭 목록 (오타·공백 누락 정제에 사용)
    _KNOWN_SIGUNGU = [
        # 경기도
        "수원시","성남시","의정부시","안양시","부천시","광명시","평택시","동두천시","안산시",
        "고양시","과천시","구리시","남양주시","오산시","시흥시","군포시","의왕시","하남시",
        "용인시","파주시","이천시","안성시","김포시","화성시","광주시","양주시","포천시",
        "여주시","연천군","가평군","양평군",
        # 서울
        "종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구","강북구",
        "도봉구","노원구","은평구","서대문구","마포구","양천구","강서구","구로구","금천구",
        "영등포구","동작구","관악구","서초구","강남구","송파구","강동구",
        # 부산
        "영도구","부산진구","동래구","해운대구","사하구","금정구","연제구","수영구","사상구","기장군",
        # 그 외 주요 시군구
        "남구","북구","동구","서구","달서구","수성구","달성군","중구",
        "계양구","부평구","남동구","연수구","미추홀구","서구","강화군","옹진군",
        "완산구","덕진구","익산시","군산시","정읍시","남원시","김제시",
        "순천시","목포시","여수시","나주시","광양시",
        "포항시","경주시","김천시","안동시","구미시","영주시","영천시","상주시","문경시","경산시",
        "창원시","진주시","통영시","사천시","김해시","밀양시","거제시","양산시",
        "제주시","서귀포시",
        "천안시","공주시","보령시","아산시","서산시","논산시","계룡시","당진시",
        "청주시","충주시","제천시",
        "춘천시","원주시","강릉시","동해시","태백시","속초시","삼척시",
        "세종시",
    ]

    def extract_sigungu(addr):
        """주소에서 시군구를 추출하되, 오타·공백 누락 등을 정제합니다."""
        if not isinstance(addr, str):
            return "기타"
        addr = addr.strip()
        parts = addr.split()
        if len(parts) < 2:
            return "기타"

        raw_token = parts[1]  # 두 번째 토큰 (정상이면 시군구명)

        # ① 정상 케이스: 끝이 시/군/구로 끝나는 경우
        if raw_token.endswith(("시", "군", "구")):
            # 오타 교정 (예: 냠양주시 → 남양주시)
            for known in _KNOWN_SIGUNGU:
                if raw_token == known:
                    return known
                # 2글자 이상 일치하는 알려진 이름으로 교정
                if len(raw_token) >= 3 and len(known) >= 3:
                    if raw_token[-1] == known[-1] and sum(a == b for a, b in zip(raw_token, known)) >= len(known) - 1:
                        return known
            return raw_token  # 교정 불가시 원본 반환

        # ② 공백 누락 케이스: 두 번째 토큰이 "시흥시경기도과기대로"처럼 붙어 있는 경우
        for known in _KNOWN_SIGUNGU:
            if raw_token.startswith(known):
                return known

        return raw_token

    sido_df["시군구"] = sido_df[addr_col].apply(extract_sigungu)

    # 시군구별 소화전 수 집계
    count_df = sido_df.groupby("시군구").size().reset_index(name="소화전수")

    # 시군구별 평균 좌표 집계 (CSV에 좌표가 있으면)
    if lat_col and lon_col:
        sido_df[lat_col] = pd.to_numeric(sido_df[lat_col], errors="coerce")
        sido_df[lon_col] = pd.to_numeric(sido_df[lon_col], errors="coerce")
        coords_df = sido_df.dropna(subset=[lat_col, lon_col]).groupby("시군구")[[lat_col, lon_col]].mean().reset_index()
        coords_df.rename(columns={lat_col: "latitude", lon_col: "longitude"}, inplace=True)
        count_df = pd.merge(count_df, coords_df, on="시군구", how="left")
    else:
        count_df["latitude"] = None
        count_df["longitude"] = None

    # 좌표가 없는 시군구는 _SIGUNGU_COORDS에서 보완
    for i, row in count_df.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            sg = row["시군구"]
            fallback = next(
                (v for k, v in _SIGUNGU_COORDS.items() if sg.startswith(k[:2]) or k.startswith(sg[:2])),
                None,
            )
            if fallback:
                count_df.at[i, "latitude"] = fallback[0]
                count_df.at[i, "longitude"] = fallback[1]

    count_df = count_df.dropna(subset=["latitude", "longitude"]).copy()

    # 취약 지수 = 소화전 수의 역수 기반 (Phase 1 방식과 통일)
    # 소화전이 적을수록 취약도가 높으며, 역수 min-max 정규화로 0~49.9 범위 유지
    if count_df.empty:
        return pd.DataFrame()

    inv = (1.0 / count_df["소화전수"].replace(0, np.nan)).fillna(0)
    inv_min, inv_max = inv.min(), inv.max()
    if inv_max > inv_min:
        count_df["취약_지수"] = ((inv - inv_min) / (inv_max - inv_min) * 49.9).round(2)
    else:
        count_df["취약_지수"] = 25.0
    count_df["시도명"] = sido_name
    count_df["시도명_full"] = sido_name + " " + count_df["시군구"]
    count_df["소화전_밀도_D"] = count_df["소화전수"]

    return count_df.sort_values(by="취약_지수", ascending=False).reset_index(drop=True)


def build_sigungu_folium_map(sigungu_df: pd.DataFrame, sido_name: str, top_n: int = 10):
    """시군구 단위 순찰 경로 Folium 지도를 생성합니다."""
    if not FOLIUM_AVAILABLE or sigungu_df.empty:
        return None, sigungu_df

    route_df = sigungu_df.head(top_n).copy()

    path = []
    for _, row in route_df.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        path.append((float(lat), float(lon)))

    if not path:
        return None, route_df

    center_lat = sum(p[0] for p in path) / len(path)
    center_lon = sum(p[1] for p in path) / len(path)

    route_map = folium.Map(location=[center_lat, center_lon], zoom_start=9)

    color_map = {1: "red", 2: "orange", 3: "orange"}
    for idx, (lat, lon) in enumerate(path, start=1):
        row = route_df.iloc[idx - 1]
        sg_name = row.get("시군구", "")
        vuln = row.get("취약_지수", 0)
        cnt = int(row.get("소화전수", 0))
        color = color_map.get(idx, "blue")
        popup_html = (
            f"<strong>{idx}순위: {sg_name}</strong><br>"
            f"취약 지수: {vuln:.1f}<br>"
            f"소화전 수: {cnt}개"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{idx}. {sg_name} (소화전 {cnt}개)",
            icon=folium.Icon(color=color, icon="fire" if idx == 1 else "info-sign", prefix="glyphicon"),
        ).add_to(route_map)

    if len(path) > 1:
        folium.PolyLine(path, color="red", weight=4, opacity=0.8, dash_array="8").add_to(route_map)

    # 범례 추가
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:white;
                padding:12px 16px;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
                font-size:13px;line-height:1.8;">
        <b>🚒 순찰 우선순위</b><br>
        <span style="color:#d62728;">●</span> 1순위 (최고 취약)<br>
        <span style="color:#ff7f0e;">●</span> 2~3순위<br>
        <span style="color:#1f77b4;">●</span> 4순위 이하<br>
        <span style="color:#e00;">——</span> 순찰 경로
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(legend_html))

    return route_map, route_df

# 출동 우선등급(Top20%) 및 네비게이션 후보 계산
#  - 출동_우선등급: 취약지수 상위 20% -> A, 나머지 -> B
threshold_80 = df_sido["취약_지수"].quantile(0.8)
df_sido["출동_우선등급"] = np.where(
    df_sido["취약_지수"] >= threshold_80, "A. 핵심 출동권역", "B. 일반 순찰권역"
)

# 네비게이션 후보 계산 (개별 소화전 좌표가 존재하면 사용)
navigation_df = compute_nearest_hydrant_info(df_sido, hydrant_points)

# 출력 포맷 정리: 소수점 2자리 고정
if "소화전_밀도_D" in df_sido.columns:
    df_sido["소화전_밀도_D"] = df_sido["소화전_밀도_D"].round(2)
if "취약_지수" in df_sido.columns:
    df_sido["취약_지수"] = df_sido["취약_지수"].round(2)

# ==========================================
# 2. 메인 화면
# ==========================================
st.title("🚒 전국 소방 취약도 분석 & 출동 네비게이션 제안")
st.subheader("1페이즈: 핵심 지표 도출 → 2페이즈: 시각화 해석 → 3페이즈: 대안 및 출동 전략")
st.divider()

# (프로젝트 개요는 Phase 1에서만 표시됩니다.)

# ==========================================
# 3. 탭 구성
# ==========================================
st.sidebar.title("🔍 분석 페이즈 선택")
menu = st.sidebar.radio(
    "이동할 단계를 선택하세요:",
    [
        "1️⃣ Phase 1 | 핵심 분석 지표 도출",
        "2️⃣ Phase 2 | 시각화 기반 분석 심층화",
        "3️⃣ Phase 3 | 해결 전략 & 우선순위 제언",
    ]
)

# 고정 출처 표기 (우측 사이드바 하단)
st.sidebar.markdown("---")
st.sidebar.markdown("**출처:** NFDS(국가화재정보시스템), KOSIS(국가통계포털)")

# ------------------------------------------
# Tab 1 — 공간 지리 맵핑
# ------------------------------------------
if menu == "1️⃣ Phase 1 | 핵심 분석 지표 도출":
    st.header("🗺️ Phase 1: 화재 취약도 지표 분석")
    
    st.subheader("📌 개요")
    st.write("전국 광역/기초 자치단체별 화재 발생 현황과 소화전 인프라를 비교하여 화재 위험 지역을 식별합니다.")

    col_goal, col_problem = st.columns(2)
    with col_goal:
        st.subheader("🎯 목적")
        st.markdown("- 취약 지역 조기 식별 및 자원 우선 배분 기준 정립\n- 데이터 기반의 객관적인 소방 행정 정책 수립 지원")
    with col_problem:
        st.subheader("🚩 문제정의")
        st.write("인구 밀집 및 화재 다발 지역임에도 불구하고 소화전 등 기초 소방 시설이 부족한 '안전 사각지대'를 정량화합니다.")

    st.divider()

    # 핵심 공식 및 법적 근거 (컴팩트 배치)
    st.subheader("🧮 핵심 지표 및 산출 근거")
    
    col_formula_imgs = st.columns(2)
    with col_formula_imgs[0]:
        # 취약지수 공식 이미지 추가 (크기 조절)
        formula_img_path = "취약지수.png"
        if os.path.exists(formula_img_path):
            st.image(formula_img_path, caption="화재 취약 지수 산출 공식", width=600)
    with col_formula_imgs[1]:
        # 연구 배경 이미지 이동 및 크기 조절
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f != "취약지수.png"]
        if image_files:
            st.image(image_files[0], caption="소방 방재 연구 배경 및 도메인 지식", width=550)

    col_info_1, col_info_2 = st.columns(2)
    with col_info_1:
        st.markdown("**⚖️ 법적 근거**")
        st.info("소방시설 설치 및 관리에 관한 법률 시행령 (소화전 설치 기준 및 거리 준수)")
    with col_info_2:
        st.markdown("**📚 데이터 출처**")
        st.info("국가화재정보시스템(NFDS) 전문가 설문 및 가중치 분석 연구")

    st.divider()

    # 전국 소화전 분포 지도
    st.subheader("📍 전국 소화전 분포 지도")
    
    valid_scores = df_sido["취약_지수"].dropna()
    score_cutoff = st.slider(
        "취약 지수 필터 (높을수록 위험)",
        min_value=float(valid_scores.min()),
        max_value=float(valid_scores.max()),
        value=float(valid_scores.min()),
        step=0.5,
    )
    filtered_df = df_sido[df_sido["취약_지수"] >= score_cutoff]
    
    col_map_view, col_map_table = st.columns([3, 2])

    with col_map_view:
        if hydrant_points is not None:
            map_df = hydrant_points.dropna(subset=["latitude", "longitude"])
            if map_df.empty:
                st.error("업로드된 개별 소화전 좌표 데이터에 유효한 위치 정보가 없습니다.")
            else:
                st.map(map_df, size=5)
        else:
            map_df = filtered_df.dropna(subset=["latitude", "longitude"])
            if map_df.empty:
                st.error("지도에 표시할 수 있는 유효한 좌표 데이터가 없습니다.")
            else:
                render_vulnerability_map(map_df)

    with col_map_table:
        st.markdown("**주요 지표 요약**")
        styled_df = (
            filtered_df[[
                "시도명", "면적_B", "소화전개소_A",
                "소화전_밀도_D", "화재발생건수_C",
                "취약_지수", "법적기준_미달율",
            ]]
            .sort_values(by="취약_지수", ascending=False)
        )
        st.dataframe(styled_df, width="stretch")

# ------------------------------------------
# Tab 2 — 4분면 매트릭스 진단
# ------------------------------------------
elif menu == "2️⃣ Phase 2 | 시각화 기반 분석 심층화":
    st.header("📊 Phase 2: 취약지수에 따른 데이터 분석")
    
    st.sidebar.subheader("📍 지역 필터")
    selected_sidos = st.sidebar.multiselect(
        "분석할 시도를 선택하세요", 
        options=sorted(df_sido["시도명_full"].unique()),
        default=sorted(df_sido["시도명_full"].unique())[:3]
    )
    
    display_df = df_sido[df_sido["시도명_full"].isin(selected_sidos)]

    # (차트) 상위 취약 지역
    median_fire = display_df["화재발생건수_C"].median()
    median_density = display_df["소화전_밀도_D"].median()

    def get_zone(f, d):
        if f >= median_fire and d < median_density:
            return "🚨 최우선 관리"
        elif f < median_fire and d >= median_density:
            return "✅ 안전 지대"
        elif f >= median_fire and d >= median_density:
            return "📊 관리 요망"
        else:
            return "⚠️ 예방 강화"

    display_df["분석_구역"] = display_df.apply(lambda x: get_zone(x["화재발생건수_C"], x["소화전_밀도_D"]), axis=1)

    st.subheader("📈 화재 빈도 vs 소화전 밀도 (4분면 매트릭스)")
    fig_quad = px.scatter(
        display_df, x="화재발생건수_C",
        y="소화전_밀도_D",
        color="분석_구역",
        size="취약_지수",
        hover_name="시도명",
        title="4분면 분석: 화재 건수 vs 소화전 밀도",
        color_discrete_map={
            "🚨 최우선 관리": "#D62728",
            "⚠️ 예방 강화": "#FF7F0E",
            "📊 관리 요망": "#1F77B4",
            "✅ 안전 지대": "#2CA02C",
        },
        labels={
            "화재발생건수_C": "연간 화재 건수",
            "소화전_밀도_D": "소화전 밀도 (개/km²)",
            "취약_지수": "취약 지수",
            "분석_구역": "분류",
        },
        height=620,
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # 데이터 분석 결과 텍스트 요약
    st.subheader("💡 데이터 분석 결과 요약")
    highest_v = display_df.loc[display_df["취약_지수"].idxmax()]
    lowest_v = display_df.loc[display_df["취약_지수"].idxmin()]
    
    analysis_text = f"""
    - **전체 요약:** 선택된 {len(selected_sidos)}개 시도 중 **{highest_v['시도명']}**이(가) 취약 지수 {highest_v['취약_지수']:.2f}로 가장 위험도가 높은 것으로 분석되었습니다.
    - **최우선 관리 대상:** 화재 빈도가 중위값({median_fire:.0f}건) 이상이면서 소화전 밀도가 중위값({median_density:.2f}개/km²) 이하인 지역은 집중 관리가 필요합니다.
    - **인프라 결핍:** 취약 지수가 높은 지역은 주로 화재 발생 밀도가 급증함에도 불구하고 소화전 확충 속도가 따라가지 못하는 경향을 보입니다.
    """
    st.info(analysis_text)

# ------------------------------------------
# Tab 3 — 우선순위 정책 제언
# ------------------------------------------
elif menu == "3️⃣ Phase 3 | 해결 전략 & 우선순위 제언":
    st.header("💡 Phase 3: 데이터 기반 결론 및 정책 제언")
    st.subheader("🎯 지역별 시/군/구 순찰 내비게이션 전략")
    st.write("시/도를 선택하면 해당 지역 내 시/군/구별 소화전 데이터를 분석하여 최적 순찰 경로를 제안합니다.")

    # ── 지역 선택 UI ──────────────────────────────────────────
    sido_options = sorted(df_sido["시도명_full"].unique())
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        nav_sido_val = st.selectbox(
            "🗺️ 순찰 지역 선택 (시/도)",
            options=sido_options,
            index=sido_options.index("경기도") if "경기도" in sido_options else 0,
            help="선택한 시/도 내의 시군구 단위로 순찰 내비게이션이 생성됩니다."
        )
    with col_sel2:
        top_n_routes = st.slider(
            "🔢 순찰 경유지 수 (상위 N개 시군구)",
            min_value=3, max_value=15, value=7, step=1,
            help="소화전이 부족한 상위 N개 시군구를 경유하는 경로를 생성합니다."
        )
    display_name = nav_sido_val

    st.markdown("<style>\n"
                ".phase3-step {padding:14px;border-radius:10px;margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.08);}\n"
                ".phase3-step.rank1 {border-left:5px solid #d62728;background:#fff5f5;}\n"
                ".phase3-step.rank2 {border-left:5px solid #ff7f0e;background:#fff7ed;}\n"
                ".phase3-step.rank3 {border-left:5px solid #ff7f0e;background:#fff7ed;}\n"
                ".phase3-step.rankN {border-left:5px solid #1f77b4;background:#edf6ff;}\n"
                ".phase3-step.rankLast {border-left:5px solid #2ca02c;background:#effaf1;}\n"
                ".phase3-step-title {font-size:1rem;font-weight:700;margin-bottom:4px;}\n"
                ".phase3-step-sub {font-size:0.94rem;color:#333;}\n"
                "</style>", unsafe_allow_html=True)

    # ── 시군구 단위 데이터 집계 ────────────────────────────────
    hydrant_raw_for_route = hydrant_points_raw if 'hydrant_points_raw' in dir() else None
    sigungu_df = build_sigungu_route_for_sido(nav_sido_val, hydrant_raw_for_route)

    col_nav, col_detail = st.columns([3, 2])

    with col_nav:
        st.markdown(f"**🚒 {display_name} 내 시/군/구 순찰 경로 (소화전 부족 상위 {top_n_routes}개)**")

        if not sigungu_df.empty:
            route_map_sg, route_top_df = build_sigungu_folium_map(sigungu_df, nav_sido_val, top_n=top_n_routes)
            if route_map_sg is not None:
                if st_folium:
                    st_folium(route_map_sg, width=750, height=520)
                else:
                    # streamlit-folium 미설치 시 HTML 렌더링으로 대체
                    map_html = route_map_sg._repr_html_()
                    components.html(map_html, height=520, scrolling=False)
            else:
                st.warning("경로 지도를 생성할 수 없습니다. 소화전 데이터를 확인해 주세요.")
        else:
            # 소화전 개별 좌표 없음 → 시도 단위 기본 경로로 fallback
            st.info(f"📌 {nav_sido_val}의 개별 소화전 좌표 데이터가 없어 시도 단위 경로로 표시합니다.")
            route_candidates = df_sido.sort_values(by="취약_지수", ascending=False).reset_index(drop=True)
            phase3_route_df = route_candidates.head(top_n_routes)
            route_map_fb, phase3_route_df = build_phase3_route_map(phase3_route_df)
            if route_map_fb is not None:
                if st_folium:
                    st_folium(route_map_fb, width=750, height=520)
                else:
                    components.html(route_map_fb._repr_html_(), height=520, scrolling=False)
            else:
                st.warning("지도를 표시할 수 없습니다.")
            route_top_df = phase3_route_df

    with col_detail:
        st.markdown("**🧭 단계별 순찰 경로 가이드**")

        if not sigungu_df.empty:
            guide_df = sigungu_df.head(top_n_routes).reset_index(drop=True)
            total = len(guide_df)
            for idx, row in guide_df.iterrows():
                rank = idx + 1
                sg_name = row.get("시군구", row.get("시도명_full", ""))
                vuln = row.get("취약_지수", 0)
                cnt = int(row.get("소화전수", row.get("소화전개소_A", 0)))
                if rank == 1:
                    label, css, icon = "[1순위 / 출발점]", "rank1", "🚒"
                elif rank == total:
                    label, css, icon = f"[{rank}순위 / 도착점]", "rankLast", "🏁"
                elif rank <= 3:
                    label, css, icon = f"[{rank}순위 / 경유지]", "rank2", "🔴"
                else:
                    label, css, icon = f"[{rank}순위 / 경유지]", "rankN", "📍"

                st.markdown(
                    f"<div class='phase3-step {css}'>"
                    f"<div class='phase3-step-title'>{icon} {label} {sg_name}</div>"
                    f"<div class='phase3-step-sub'>"
                    f"취약 지수: <b>{vuln:.1f}</b> | 소화전 수: <b>{cnt}개</b>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("시군구 단위 데이터가 없습니다.")

        st.markdown("**📌 순찰 경로 활용 방향**")
        st.markdown("1. **소화전 부족 지역 우선 점검:** 소화전 수가 적은 시군구는 화재 시 대응 능력이 낮으므로 순찰을 집중합니다.")
        st.markdown("2. **경유지 중심 자원 배치:** 순찰 경로 상 경유지에 예비 소방차·급수차를 대기 배치하여 신속 대응합니다.")
        st.markdown("3. **실시간 관제 연동:** 소방 상황실에서 해당 지역 취약 시군구를 상시 모니터링합니다.")

    st.divider()

    # 개선 방향 및 대안
    st.subheader("🛡️ 개선 방향 및 정책 제언")
    col_strat_a, col_strat_b = st.columns(2)
    with col_strat_a:
        st.markdown("#### [단기] 인프라 보강 및 대응")
        st.error("""
        - **급수차 우선 진입로 확보:** 소화전 밀도가 낮은 지역은 용수 공급이 어려우므로 대형 급수차 전용 진입로 우선 설정.
        - **추가 소화전 집중 설치:** 법적 기준 미달율이 높은 **{display_name}** 내 고위험 구역에 예산 우선 투입.
        - **비상 소화장치 보급:** 소방차 진입 곤란 지역(전통시장, 노후 주거지)에 주민 자율 소화장치 설치 확대.
        """)
    with col_strat_b:
        st.markdown("#### [장기] 예방 행정 및 정책")
        st.success(f"""
        - **지자체 협력(시청/도청):** 도시 개발 계획 단계에서 화재 취약 지수를 반영한 '소방 안전성 검토' 의무화 제안.
        - **소방안전지도 고도화:** 본 데이터를 시청 관제 센터와 공유하여 지역별 맞춤형 소방 행정 서비스 제공.
        - **법적 기준 강화:** 화재 발생 밀도가 높은 지역에 대해 소화전 설치 간격을 법적 기준보다 강화하여 적용.
        """)

    st.divider()
    st.subheader("💡 결론")
    st.markdown(f"""
    본 분석을 통해 **{display_name}** 지역의 실질적인 인프라 결핍과 화재 취약성을 확인하였습니다. 
    단순히 소화전을 늘리는 것을 넘어, **데이터 기반의 우선순위 설정**과 **내비게이션을 활용한 전략적 대응**이 결합될 때 
    지역 사회의 화재 안전망이 비로소 완성될 수 있습니다.
    """)
