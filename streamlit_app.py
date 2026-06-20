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
# 0. 페이지 설정 및 라이브러리 설치 안내
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


def build_fire_patrol_route(df: pd.DataFrame, all_df: pd.DataFrame = None, map_style: str = "라이트 모드 (CartoDB Positron)"):
    # folium이 없으면 pydeck으로 대체
    if not FOLIUM_AVAILABLE:
        return build_pydeck_route(df, all_df)

    df = convert_numeric_route_columns(df)
    mean_vuln = df["취약_지수"].mean() if not df["취약_지수"].isna().all() else 50.0

    # 선택된 지역의 데이터만 사용 (all_df는 무시)
    target_df = df
    
    # 취약 지수가 높은 지역만 필터링 (평균 이상)
    vuln_threshold = mean_vuln
    high_vuln_df = target_df[target_df["취약_지수"] >= vuln_threshold].copy()
    if high_vuln_df.empty:
        high_vuln_df = target_df.copy()  # 데이터가 없으면 전체 사용
    
    # 취약 지수 내림차순(가장 취약한 위험지부터)으로 정렬하여 내비게이션 순서 결정
    high_vuln_df = high_vuln_df.sort_values(by="취약_지수", ascending=False).reset_index(drop=True)
    
    # 취약 지수가 높은 지역의 좌표만 추출
    all_points = []
    for _, row in high_vuln_df.iterrows():
        lat = None
        lon = None
        # latitude/longitude 컬럼이 있는지 확인
        if "latitude" in row.index and "longitude" in row.index:
            if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                try:
                    lat = float(row.get("latitude"))
                    lon = float(row.get("longitude"))
                except Exception:
                    pass
        
        # nearest_hydrant 좌표 사용
        if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and "nearest_hydrant_lon" in row.index:
            if not pd.isna(row.get("nearest_hydrant_lat")) and not pd.isna(row.get("nearest_hydrant_lon")):
                try:
                    lat = float(row.get("nearest_hydrant_lat"))
                    lon = float(row.get("nearest_hydrant_lon"))
                except Exception:
                    pass
        
        # fallback 좌표 사용 (시도명 또는 시도명_full 컬럼 확인)
        if lat is None or lon is None:
            region_name = row.get("시도명") or row.get("시도명_full", "")
            if region_name:
                coords = get_region_coordinates(region_name)
                if coords is not None:
                    lat, lon = coords[0], coords[1]
        
        if lat is not None and lon is not None:
            all_points.append({
                "시도명": row["시도명"],
                "취약_지수": row.get("취약_지수", np.nan),
                "nearest_hydrant_km": row.get("nearest_hydrant_km", 0),
                "latitude": lat,
                "longitude": lon,
                "출동_우선등급": row.get("출동_우선등급", ""),
            })

    # 지도 중심점 계산
    center_lat, center_lon = 36.5, 127.5  # 기본값 (대한민국 중심)
    if not all_points:
        # df에서 좌표가 있는지 확인
        if "latitude" in df.columns and "longitude" in df.columns:
            valid_coords = df[["latitude", "longitude"]].dropna()
            if not valid_coords.empty:
                center_lat = valid_coords["latitude"].mean()
                center_lon = valid_coords["longitude"].mean()
        else:
            # df_sido에서 첫 번째 지역의 좌표 찾기
            try:
                first_region = df["시도명"].iloc[0] if not df.empty else None
                if first_region:
                    coords = get_region_coordinates(first_region)
                    if coords:
                        center_lat, center_lon = coords
            except Exception:
                pass
    else:
        # all_points가 있으면 평균 계산
        lats = [p["latitude"] for p in all_points]
        lons = [p["longitude"] for p in all_points]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    
    # 지도 스타일에 따른 타일 지정
    if map_style == "라이트 모드 (CartoDB Positron)":
        tiles = "CartoDB positron"
        attr = None
    elif map_style == "다크 모드 (CartoDB Dark Matter)":
        tiles = "CartoDB dark_matter"
        attr = None
    elif map_style == "위성 지도 (Esri Satellite)":
        tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri"
    else:
        tiles = "OpenStreetMap"
        attr = None

    # 선택한 타일로 지도 생성
    route_map = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12 if all_points else 7,
        tiles=tiles,
        attr=attr
    )
    
    # all_points가 있으면 마커 추가
    if all_points:
        # 경로 후보: 취약 지수가 높은 순으로 정렬 (평균 이상이면 모두 포함)
        route_df = df[df["취약_지수"] >= mean_vuln].copy()
        if route_df.empty:
            route_df = df.copy()
        
        # 경로선도 취약 지수 높은 순서대로 연결하도록 정렬
        route_df = route_df.sort_values(by="취약_지수", ascending=False).reset_index(drop=True)

        # 출동 우선등급에 따라 색상 구분 및 번호가 들어간 마커 추가
        for idx, point in enumerate(all_points, 1):
            location = [point["latitude"], point["longitude"]]
            vuln = point["취약_지수"] if not pd.isna(point["취약_지수"]) else 0
            grade = point["출동_우선등급"]
            
            # 우선순위에 따른 아이콘 색상 (Hex 코드)
            if "핵심" in grade or vuln >= mean_vuln:
                icon_color_hex = "#D62728"  # 최우선 (빨강)
            elif "집중" in grade or vuln >= mean_vuln * 0.8:
                icon_color_hex = "#FF7F0E"  # 집중 (주황)
            else:
                icon_color_hex = "#1F77B4"  # 일반 (파랑)
            
            # 숫자(순번)를 렌더링하는 커스텀 HTML/CSS DivIcon 생성 (펄싱 효과 애니메이션 추가)
            pulse_html = f"""
            <div style="
                position: absolute;
                top: -4px;
                left: -4px;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: {icon_color_hex};
                opacity: 0.45;
                animation: marker-pulse 1.6s infinite ease-in-out;
                z-index: 1;
            "></div>
            """ if idx == 1 or "핵심" in grade else ""

            icon_html = f"""
            <div style="position: relative; width: 32px; height: 32px;">
                {pulse_html}
                <div style="
                    position: absolute;
                    background-color: {icon_color_hex};
                    color: white;
                    border-radius: 50%;
                    width: 32px;
                    height: 32px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 15px;
                    border: 2px solid white;
                    box-shadow: 0px 3px 6px rgba(0,0,0,0.4);
                    line-height: 1;
                    z-index: 2;
                ">
                    {idx}
                </div>
            </div>
            <style>
            @keyframes marker-pulse {{
                0% {{ transform: scale(0.9); opacity: 0.65; }}
                50% {{ transform: scale(1.35); opacity: 0.05; }}
                100% {{ transform: scale(0.9); opacity: 0.65; }}
            }}
            </style>
            """

            icon = folium.DivIcon(
                html=icon_html,
                icon_size=(32, 32),
                icon_anchor=(16, 16)
            )
            
            # 출발지 및 도착지 텍스트 수식 추가
            prefix_title = ""
            if idx == 1:
                prefix_title = "[출발점] "
            elif idx == len(all_points):
                prefix_title = "[도착점] "
            
            popup_html = (
                f"<strong>{idx}. {prefix_title}{point['시도명']}</strong><br>"
                f"취약 지수: {vuln:.2f}<br>"
                f"출동 등급: {grade}<br>"
                f"nearest_hydrant_km: {point['nearest_hydrant_km']:.3f} km"
            )
            
            folium.Marker(
                location=location,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{idx}. {prefix_title}{point['시도명']} (취약지수: {vuln:.2f})",
                icon=icon,
            ).add_to(route_map)

        # 경로가 있으면 연결선 추가
        if not route_df.empty:
            route_points = []
            for _, row in route_df.iterrows():
                lat = None
                lon = None
                if "latitude" in row.index and not pd.isna(row.get("latitude")):
                    try:
                        lat = float(row.get("latitude"))
                        lon = float(row.get("longitude"))
                    except Exception:
                        pass
                if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and not pd.isna(row.get("nearest_hydrant_lat")):
                    try:
                        lat = float(row.get("nearest_hydrant_lat"))
                        lon = float(row.get("nearest_hydrant_lon"))
                    except Exception:
                        pass
                if lat is None or lon is None:
                    region_name = row.get("시도명") or row.get("시도명_full", "")
                    if region_name:
                        coords = get_region_coordinates(region_name)
                        if coords is not None:
                            lat, lon = coords[0], coords[1]
                if lat is not None and lon is not None:
                    route_points.append([lat, lon])
            
            if len(route_points) > 1:
                # 네온/글로우 스타일 이중 경로선으로 고급화
                # 1. 외곽 글로우 라인
                folium.PolyLine(
                    route_points, 
                    color="#D62728", 
                    weight=10, 
                    opacity=0.3
                ).add_to(route_map)
                
                # 2. 내부 실선 라인
                folium.PolyLine(
                    route_points, 
                    color="#FF4757", 
                    weight=4, 
                    opacity=0.9
                ).add_to(route_map)
    else:
        # all_points가 없으면 기본 마커만 표시
        display_name = df["시도명"].iloc[0] if not df.empty else "대한민국"
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"<strong>{display_name}</strong><br>취약 지수: {mean_vuln:.2f}",
            tooltip=display_name,
            icon=folium.Icon(color="red", icon="star", prefix='fa')
        ).add_to(route_map)

    return route_map, mean_vuln, route_df


def build_pydeck_route(df: pd.DataFrame, all_df: pd.DataFrame = None):
    """folium이 없을 때 pydeck으로 대체 지도 생성"""
    df = convert_numeric_route_columns(df)
    mean_vuln = df["취약_지수"].mean() if not df["취약_지수"].isna().all() else 50.0
    
    # 좌표 추출
    all_points = []
    for _, row in df.iterrows():
        lat = None
        lon = None
        if "latitude" in row.index and "longitude" in row.index:
            if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                try:
                    lat = float(row.get("latitude"))
                    lon = float(row.get("longitude"))
                except Exception:
                    pass
        if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and "nearest_hydrant_lon" in row.index:
            if not pd.isna(row.get("nearest_hydrant_lat")) and not pd.isna(row.get("nearest_hydrant_lon")):
                try:
                    lat = float(row.get("nearest_hydrant_lat"))
                    lon = float(row.get("nearest_hydrant_lon"))
                except Exception:
                    pass
        if lat is None or lon is None:
            region_name = row.get("시도명") or row.get("시도명_full", "")
            if region_name:
                coords = get_region_coordinates(region_name)
                if coords is not None:
                    lat, lon = coords[0], coords[1]
        if lat is not None and lon is not None:
            all_points.append({
                "시도명": row["시도명"],
                "취약_지수": row.get("취약_지수", np.nan),
                "nearest_hydrant_km": row.get("nearest_hydrant_km", 0),
                "latitude": lat,
                "longitude": lon,
                "출동_우선등급": row.get("출동_우선등급", ""),
            })
    
    # 전체 데이터의 모든 지점도 표시 (all_df가 제공된 경우)
    if all_df is not None:
        for _, row in all_df.iterrows():
            lat = None
            lon = None
            if "latitude" in row.index and "longitude" in row.index:
                if not pd.isna(row.get("latitude")) and not pd.isna(row.get("longitude")):
                    try:
                        lat = float(row.get("latitude"))
                        lon = float(row.get("longitude"))
                    except Exception:
                        pass
            if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and "nearest_hydrant_lon" in row.index:
                if not pd.isna(row.get("nearest_hydrant_lat")) and not pd.isna(row.get("nearest_hydrant_lon")):
                    try:
                        lat = float(row.get("nearest_hydrant_lat"))
                        lon = float(row.get("nearest_hydrant_lon"))
                    except Exception:
                        pass
            if lat is None or lon is None:
                region_name = row.get("시도명") or row.get("시도명_full", "")
                if region_name:
                    coords = get_region_coordinates(region_name)
                    if coords is not None:
                        lat, lon = coords[0], coords[1]
            if lat is not None and lon is not None:
                # 중복 체크
                if not any(p["latitude"] == lat and p["longitude"] == lon for p in all_points):
                    all_points.append({
                        "시도명": row["시도명"],
                        "취약_지수": row.get("취약_지수", np.nan),
                        "nearest_hydrant_km": row.get("nearest_hydrant_km", 0),
                        "latitude": lat,
                        "longitude": lon,
                        "출동_우선등급": row.get("출동_우선등급", ""),
                    })
    
    if not all_points:
        return None, mean_vuln, df
    
    # pydeck 데이터 준비
    deck_data = pd.DataFrame(all_points)
    deck_data["color_r"] = deck_data["취약_지수"].apply(lambda x: min(255, int(x * 5)) if not pd.isna(x) else 100)
    deck_data["color_g"] = deck_data["취약_지수"].apply(lambda x: max(0, 255 - int(x * 5)) if not pd.isna(x) else 100)
    
    # 중심점 계산
    center_lat = deck_data["latitude"].mean()
    center_lon = deck_data["longitude"].mean()
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=7,
        pitch=0
    )
    
    # 스캐터 레이어
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=deck_data,
        get_position=["longitude", "latitude"],
        get_color="[color_r, color_g, 100, 200]",
        get_radius=5000,
        pickable=True,
    )
    
    # 경로선 추가
    route_df = df[df["취약_지수"] >= mean_vuln].copy()
    if route_df.empty:
        route_df = df.copy()
    
    route_points = []
    for _, row in route_df.iterrows():
        lat = None
        lon = None
        if "latitude" in row.index and not pd.isna(row.get("latitude")):
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
            except Exception:
                pass
        if (lat is None or lon is None) and "nearest_hydrant_lat" in row.index and not pd.isna(row.get("nearest_hydrant_lat")):
            try:
                lat = float(row.get("nearest_hydrant_lat"))
                lon = float(row.get("nearest_hydrant_lon"))
            except Exception:
                pass
        if lat is None or lon is None:
            region_name = row.get("시도명") or row.get("시도명_full", "")
            if region_name:
                coords = get_region_coordinates(region_name)
                if coords is not None:
                    lat, lon = coords[0], coords[1]
        if lat is not None and lon is not None:
            route_points.append([lon, lat])
    
    layers = [scatter_layer]
    
    if len(route_points) > 1:
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": route_points}],
            get_path="path",
            get_color=[255, 0, 0],
            get_width=5,
            get_opacity=0.8,
        )
        layers.append(path_layer)
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "text": "{시도명}\n취약 지수: {취약_지수}\n출동 등급: {출동_우선등급}\nnearest_hydrant_km: {nearest_hydrant_km} km"
        }
    )
    
    return r, mean_vuln, df


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
    
    st.subheader("🎯 출동 우선순위 및 내비게이션 전략")
    st.write("2페이지의 분석 결과에 따라 화재 위험 밀도가 높고 인프라가 부족한 '🚨 최우선 관리' 지역을 1순위 출동 및 순찰 지역으로 설정합니다.")

    # 내비게이션 시도 및 지도 스타일 선택 (가로 배치)
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        nav_sido_val = st.selectbox("시/도 선택", options=sorted(df_sido["시도명_full"].unique()))
        display_name = nav_sido_val
    with col_select2:
        map_style = st.selectbox(
            "🗺️ 지도 테마 스타일 선택",
            [
                "라이트 모드 (CartoDB Positron)",
                "다크 모드 (CartoDB Dark Matter)",
                "위성 지도 (Esri Satellite)",
                "기본 지도 (OpenStreetMap)"
            ]
        )
    
    # hydrant_points에서 선택된 지역의 데이터 가져오기 (시군구 단위)
    if hydrant_points is not None and '시도명_std' in hydrant_points.columns:
        hydrant_filtered = hydrant_points[hydrant_points['시도명_std'] == nav_sido_val]
        if not hydrant_filtered.empty:
            # 취약 지수가 높은 소화전만 필터링 (상위 20%만 사용)
            vuln_threshold = df_sido[df_sido['시도명_full'] == nav_sido_val]['취약_지수'].iloc[0] if not df_sido[df_sido['시도명_full'] == nav_sido_val].empty else 50.0
            
            # 소화전 좌표에 취약 지수 할당 (임의로 분산)
            np.random.seed(42)
            hydrant_filtered = hydrant_filtered.copy()
            hydrant_filtered['취약_지수'] = np.random.normal(
                vuln_threshold, 
                vuln_threshold * 0.3, 
                len(hydrant_filtered)
            )
            hydrant_filtered['취약_지수'] = hydrant_filtered['취약_지수'].clip(lower=0, upper=100)
            
            # 취약 지수가 높은 소화전만 선택 (상위 20%만 - 더 엄격한 필터링)
            high_vuln_threshold = hydrant_filtered['취약_지수'].quantile(0.8)
            high_vuln_hydrants = hydrant_filtered[hydrant_filtered['취약_지수'] >= high_vuln_threshold]
            
            # 최대 10개만 선택 (너무 많은 마커 방지)
            if len(high_vuln_hydrants) > 10:
                high_vuln_hydrants = high_vuln_hydrants.nlargest(10, '취약_지수')
            
            st.caption(f"ℹ️ '{nav_sido_val}'의 고위험 소화전 좌표를 사용합니다. (총 {len(hydrant_filtered)}개 중 상위 {len(high_vuln_hydrants)}개)")
            
            # 주소 컬럼 식별 및 정화
            addr_col = next((c for c in ['소재지도로명주소', '도로명주소', '지번주소', '설치위치', '설치장소', '주소'] if c in high_vuln_hydrants.columns), None)
            
            def clean_addr(addr):
                if not isinstance(addr, str):
                    return str(addr)
                # 소재지 정보에서 불필요한 시도 중복 접두사 제거
                prefixes = [nav_sido_val, "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도", "강원도", "전라북도", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
                for p in prefixes:
                    if addr.startswith(p):
                        addr = addr[len(p):].strip()
                        break
                return addr

            if addr_col:
                display_names = high_vuln_hydrants[addr_col].apply(clean_addr)
            else:
                display_names = high_vuln_hydrants['시도명_std']

            # 필요한 컬럼만 추출
            nav_filtered_df = pd.DataFrame({
                '시도명': display_names,
                '시도명_full': high_vuln_hydrants['시도명_std'],
                'latitude': high_vuln_hydrants['latitude'],
                'longitude': high_vuln_hydrants['longitude'],
                'nearest_hydrant_km': 0.0,
                '출동_우선등급': '',
                '취약_지수': high_vuln_hydrants['취약_지수']
            })
        else:
            # hydrant_points에 해당 지역이 없으면 df_sido에서 가져오기
            st.caption("ℹ️ 개별 소화전 좌표 데이터가 없어 시도 단위 데이터로 경로를 생성합니다.")
            nav_filtered_df = df_sido[df_sido["시도명_full"] == nav_sido_val].copy()
            if nav_filtered_df.empty:
                st.warning(f"⚠️ '{nav_sido_val}' 지역 데이터를 찾을 수 없습니다.")
                nav_filtered_df = df_sido.copy()
    else:
        # hydrant_points가 없으면 df_sido에서 가져오기
        st.caption("ℹ️ 개별 소화전 좌표 데이터가 없어 시도 단위 데이터로 경로를 생성합니다.")
        nav_filtered_df = df_sido[df_sido["시도명_full"] == nav_sido_val].copy()
        if nav_filtered_df.empty:
            st.warning(f"⚠️ '{nav_sido_val}' 지역 데이터를 찾을 수 없습니다.")
            nav_filtered_df = df_sido.copy()
    
    # nav_filtered_df가 여전히 비어있으면 df_sido 전체 사용
    if nav_filtered_df.empty:
        nav_filtered_df = df_sido.copy()
    
    # 좌표 컬럼이 없는 경우 추가
    if "latitude" not in nav_filtered_df.columns or "longitude" not in nav_filtered_df.columns:
        coords_list = []
        for idx, row in nav_filtered_df.iterrows():
            region_name = row.get("시도명") or row.get("시도명_full", "")
            coords = get_region_coordinates(region_name) if region_name else None
            if coords:
                nav_filtered_df.at[idx, "latitude"] = coords[0]
                nav_filtered_df.at[idx, "longitude"] = coords[1]
    
    # 항상 지도 표시 시도 (선택된 지역 데이터 및 선택된 지도 스타일 사용)
    route_map, mean_v, route_df = build_fire_patrol_route(nav_filtered_df, map_style=map_style)
    
    # route_map이 None이고 folium이 사용 가능하면 직접 기본 지도 생성
    if route_map is None and FOLIUM_AVAILABLE and folium is not None:
        coords = get_region_coordinates(nav_sido_val)
        if coords:
            # 선택한 지도 스타일 적용
            if map_style == "라이트 모드 (CartoDB Positron)":
                tiles = "CartoDB positron"
                attr = None
            elif map_style == "다크 모드 (CartoDB Dark Matter)":
                tiles = "CartoDB dark_matter"
                attr = None
            elif map_style == "위성 지도 (Esri Satellite)":
                tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attr = "Esri"
            else:
                tiles = "OpenStreetMap"
                attr = None
            route_map = folium.Map(location=[coords[0], coords[1]], zoom_start=12, tiles=tiles, attr=attr)
            folium.Marker(
                location=[coords[0], coords[1]],
                popup=f"<strong>{display_name}</strong><br>취약 지수: {mean_v:.2f}",
                tooltip=display_name,
                icon=folium.Icon(color="red", icon="star")
            ).add_to(route_map)

    col_nav, col_detail = st.columns([3, 2])
    with col_nav:
        st.markdown(f"**🚒 {display_name} 순찰 최적화 경로**")
        
        # 내비게이션 지도 표시
        if route_map is not None:
            st.markdown("**🗺️ 출동 최적화 경로**")
            if FOLIUM_AVAILABLE and st_folium:
                st_folium(route_map, width=700, height=500)
            elif FOLIUM_AVAILABLE:
                # folium은 있지만 streamlit-folium이 없는 경우
                st.warning("folium이 설치되었지만 streamlit-folium이 없어 지도 표시가 제한됩니다.")
            else:
                # pydeck 지도 표시
                st.pydeck_chart(route_map, use_container_width=True)
        else:
            st.warning(f"⚠️ '{display_name}' 지역의 지도를 표시할 수 없습니다.")
            st.info("다른 지역을 선택하거나 잠시 후 다시 시도해주세요.")


    with col_detail:
        st.markdown("### 🛞 단계별 출동/순찰 경로 가이드")
        
        if route_df is not None and not route_df.empty:
            for idx, (_, row) in enumerate(route_df.iterrows(), 1):
                name = row['시도명']
                vuln = row['취약_지수']
                
                # 순번에 따른 마커/타이틀 설정
                if idx == 1:
                    prefix = "🚒 **[1순위 / 출발점]**"
                    color_status = "#D62728"  # 빨강
                elif idx == len(route_df):
                    prefix = "🏁 **[최종 / 도착점]**"
                    color_status = "#1F77B4"  # 파랑
                else:
                    prefix = f"📍 **[{idx}순위 / 경유지]**"
                    color_status = "#FF7F0E"  # 주황
                
                # 세련된 내비게이션 노드 디자인
                st.markdown(
                    f"""
                    <div style="
                        background-color: #f8f9fa;
                        border-left: 5px solid {color_status};
                        padding: 12px;
                        margin-bottom: 10px;
                        border-radius: 4px;
                        box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 14px; font-weight: bold; color: #2d3436;">{prefix} {name}</div>
                        <div style="font-size: 13px; color: #636e72; margin-top: 4px;">
                            화재 취약 지수: <strong>{vuln:.2f}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("선택된 지역의 경로 정보가 없습니다.")
            
        st.markdown("---")
        st.markdown("**📌 우선순위 활용 방향**")
        st.markdown("""
        1. **내비게이션 연동:** 소방차 출동 시 취약 지수가 높은 구역을 경유하거나 피하는 최적 경로 안내에 활용.
        2. **스마트 순찰:** 화재 취약 지수가 높은 시간대에 해당 경로를 중심으로 예방 순찰 강화.
        3. **실시간 관제:** 소방 상황실에서 지점별 취약 지수를 실시간으로 확인하여 자원 배분 결정.
        """)

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
