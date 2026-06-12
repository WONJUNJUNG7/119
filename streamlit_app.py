import os
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from PIL import Image
try:
    import folium
    FOLIUM_AVAILABLE = True
except ModuleNotFoundError:
    folium = None
    FOLIUM_AVAILABLE = False
import streamlit.components.v1 as components

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(
    page_title="전국 시·도 소방안전 취약도 및 소화전 인프라 최적화 시스템",
    page_icon="🚒",
    layout="wide"
)

st.markdown(
    """
    <style>
    .streamlit-card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-bottom: 24px;
    }
    .streamlit-card .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f3c78;
        margin-bottom: 14px;
    }
    .streamlit-card ul {
        margin: 0;
        padding-left: 20px;
        color: #22313f;
    }
    .streamlit-card li {
        margin-bottom: 10px;
    }
    .streamlit-card .small-note {
        color: #5c6c7c;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
        
        # 2. 행정구역 데이터 전처리 (시·도 + 시·군·구 계층 구조 처리)
        df_area.columns = [c.replace('\ufeff', '') for c in df_area.columns]
        df_area.rename(columns={'소재지(시군구)별': '지역명_raw', '2025': '면적_B'}, inplace=True)

        sidos = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시", 
                 "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도", "전북특별자치도", 
                 "전라남도", "경상북도", "경상남도", "제주특별자치도"]
        
        # 면적 데이터에서 시도-시군구 관계 생성
        area_list = []
        curr_sido = None
        for _, row in df_area.iterrows():
            name = str(row['지역명_raw']).strip()
            if name in sidos:
                curr_sido = name
                if name == "세종특별자치시": # 세종시는 시도이자 시군구
                    area_list.append({'시도명': curr_sido, '시군구명': curr_sido, '면적_B': row['면적_B']})
            elif curr_sido:
                area_list.append({'시도명': curr_sido, '시군구명': name, '면적_B': row['면적_B']})
        df_area_processed = pd.DataFrame(area_list)

        # 화재 데이터 전처리
        df_fire.rename(columns={'행정구역별(1)': '시도명', '행정구역별(2)': '시군구명', '2025': '화재발생건수_C'}, inplace=True)
        df_fire = df_fire[df_fire['시군구명'] != '소계'].copy()
        df_fire['시군구명'] = df_fire['시군구명'].replace('세종시', '세종특별자치시')
        
        # 3. 소화전 데이터 집계 (주소 분석을 통한 시군구 단위 카운트)
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
                    if not isinstance(val, str) or not val.strip(): return "기타", "기타"
                    val = val.strip()
                    
                    # 1. 시도명 찾기 (긴 명칭 우선 매칭으로 정확도 향상)
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
                    
                    # 2. 시군구명 추출 (시도명/접두사 제거 후 첫 단어)
                    remaining = val[len(matched_prefix):].strip() if matched_prefix else val
                    parts = remaining.split()
                    sigungu = parts[0] if parts else sido
                    return sido, sigungu

                # 성능 최적화: pd.Series 대신 리스트 컴프리헨션 사용 (속도 향상)
                parsed_results = df_hydrants_raw[target_col].apply(parse_addr)
                df_hydrants_raw['시도명_std'] = [res[0] for res in parsed_results]
                df_hydrants_raw['시군구명_std'] = [res[1] for res in parsed_results]
                df_h_count = df_hydrants_raw.groupby(['시도명_std', '시군구명_std']).size().reset_index(name='소화전개소_A')
                
                # 위경도 데이터가 있다면 시군구별 평균 좌표 계산
                hydrants_mapped = normalize_coord_columns(df_hydrants_raw)
                if 'latitude' in hydrants_mapped.columns and 'longitude' in hydrants_mapped.columns:
                    coords_agg = hydrants_mapped.groupby(['시도명_std', '시군구명_std'])[['latitude', 'longitude']].mean().reset_index()
                    df_h_count = pd.merge(df_h_count, coords_agg, on=['시도명_std', '시군구명_std'], how='left')
                
                df_h_count.rename(columns={'시도명_std': '시도명', '시군구명_std': '시군구명'}, inplace=True)
            else:
                df_h_count = pd.DataFrame(columns=['시도명', '소화전개소_A'])
        else:
            df_h_count = pd.DataFrame(columns=['시도명', '시군구명', '소화전개소_A'])

        # 4. 데이터 병합 (시도명 + 시군구명 기준)
        merged = pd.merge(df_area_processed, df_fire, on=['시도명', '시군구명'], how='inner')
        merged = pd.merge(merged, df_h_count, on=['시도명', '시군구명'], how='left')
        
        # 데이터 매칭 확인을 위한 알림
        zero_count = merged['소화전개소_A'].isna().sum()
        if zero_count > 0:
            st.sidebar.info(f"ℹ️ {zero_count}개 지역은 소화전 데이터가 없거나 주소 형식이 다릅니다.")

        merged['소화전개소_A'] = merged['소화전개소_A'].fillna(0)
        
        # 표시용 지역명 생성
        merged['시도명_full'] = merged['시도명']
        merged['시도명'] = merged['시도명'] + " " + merged['시군구명']
        
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

    # 개요
    st.subheader("📌 개요")
    st.markdown("전국 광역시도별 화재·소화전 현황을 비교하여 초기 문제를 정의하고 우선 대응지역을 식별합니다.")

    # 목적
    st.subheader("🎯 목적")
    st.markdown(
        "전국 광역시도별 화재·소화전 현황을 비교합니다.\n초기 문제를 정의하고 우선 대응지역을 식별합니다."
    )

    st.markdown(
        """
        <div class="streamlit-card">
            <div class="card-title">핵심 공식</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.image("공식.png", width=860)
    st.markdown(
        """
        <div style='background:#eff6ff;border:1px solid #c7dbff;border-radius:16px;padding:16px;margin-top:12px;'>
            <div style='font-size:1.2rem;font-weight:700;color:#10316b;'>취약지수 = 0.6 × E + 0.4 × (1 / D)</div>
            <div style='margin-top:10px;color:#32455f;'>E: 화재 발생 밀도 (건/km²) · D: 소화전 설치 밀도 (개/km²)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("법적 근거: 소방시설 설치 기준(예시) — 소화전 권장 배치거리 140m 이내")

    st.divider()

    # 문제정의
    st.subheader("🚩 문제정의")
    st.markdown(
        """
        <div class="streamlit-card">
            <div class="card-title">문제 정의</div>
            <ul>
                <li>화재 다발 지역이나 인구밀집 지역에서 소화전 부족</li>
                <li>화재는 적지만 소화전이 부족한 지역의 예방 취약성</li>
                <li>소화전은 많지만 화재가 발생하는 지역의 운영·접근성 문제</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # 데이터 분석 결과
    st.subheader("📈 전국 상위 취약 지역 차트")

    valid_scores = df_sido["취약_지수"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_scores.empty:
        st.error("유효한 취약 지수 데이터가 없어 분석을 진행할 수 없습니다.")
        st.stop()

    score_cutoff = st.slider(
        "필터링 기준 취약 지수 (높을수록 인프라 결핍 심각)",
        min_value=float(valid_scores.min()),
        max_value=float(valid_scores.max()),
        value=float(valid_scores.min()),
        step=0.5,
    )

    filtered_df = df_sido[df_sido["취약_지수"] >= score_cutoff]

    col_map_view, col_map_table = st.columns([3, 2])

    with col_map_view:
        st.subheader("📍 취약 지역 공간 분포")
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
        st.subheader("📑 행정구역별 인프라 데이터")
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
    st.header("📊 Phase 2: 분석 및 결과")

    # 분석
    st.subheader("📊 분석")
    st.markdown("Phase 1 지표를 활용한 지역별 비교·시각화(바 차트, 4분면 산점도 등)")

    # (차트) 상위 취약 지역
    top_phase1 = (
        df_sido.groupby("시도명_full")[
            ["화재_발생_밀도_E", "소화전_밀도_D", "취약_지수"]
        ]
        .mean()
        .reset_index()
        .sort_values("취약_지수", ascending=False)
        .head(10)
    )
    fig_phase1 = px.bar(
        top_phase1,
        x="시도명_full",
        y=["화재_발생_밀도_E", "소화전_밀도_D"],
        barmode="group",
        title="상위 취약 지역: 화재 발생 밀도 vs 소화전 밀도",
        labels={
            "화재_발생_밀도_E": "화재 발생 밀도 (건/km²)",
            "소화전_밀도_D": "소화전 설치 밀도 (개/km²)",
            "시도명_full": "시도명",
        },
        height=520,
    )
    st.plotly_chart(fig_phase1, use_container_width=True)

    st.write("**분석결과:** 화재 발생↑ & 소화전↓ 지역 = 우선 개선 대상")

    # 4분면 분석
    df_sido["화재대비_소화전비율"] = (
        df_sido["화재발생건수_C"] / df_sido["소화전개소_A"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).round(4)

    median_fire = df_sido["화재발생건수_C"].median()
    median_density = df_sido["소화전_밀도_D"].median()

    def get_zone(f, d):
        if f >= median_fire and d < median_density:
            return "🚨 최우선 관리"
        elif f < median_fire and d >= median_density:
            return "✅ 안전 지대"
        elif f >= median_fire and d >= median_density:
            return "✨ 적정 방어"
        else:
            return "⚠️ 잠재 위험"

    df_sido["분석_구역"] = df_sido.apply(
        lambda x: get_zone(x["화재발생건수_C"], x["소화전_밀도_D"]), axis=1
    )

    fig_quad = px.scatter(
        df_sido,
        x="화재발생건수_C",
        y="소화전_밀도_D",
        color="분석_구역",
        size="취약_지수",
        hover_name="시도명",
        title="4분면 분석: 화재 건수 vs 소화전 밀도",
        color_discrete_map={
            "🚨 최우선 관리": "#EF553B",
            "⚠️ 잠재 위험": "#FECB52",
            "✨ 적정 방어": "#636EFA",
            "✅ 안전 지대": "#00CC96",
        },
        labels={
            "화재발생건수_C": "연간 화재 건수",
            "소화전_밀도_D": "소화전 밀도 (개/km²)",
            "취약_지수": "취약 지수",
            "분석_구역": "분류",
        },
        height=620,
    )
    fig_quad.add_shape(
        type="line",
        x0=median_fire,
        x1=median_fire,
        y0=0,
        y1=df_sido["소화전_밀도_D"].max() * 1.05,
        line=dict(color="gray", dash="dash"),
    )
    fig_quad.add_shape(
        type="line",
        x0=0,
        x1=df_sido["화재발생건수_C"].max() * 1.05,
        y0=median_density,
        y1=median_density,
        line=dict(color="gray", dash="dash"),
    )
    fig_quad.update_layout(
        legend_title_text="분류",
        xaxis_title="연간 화재 건수",
        yaxis_title="소화전 밀도 (개/km²)",
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # 도출된 결과
    st.write("**도출된 결과:**")
    st.markdown("- 우선 개선 대상(🚨): 화재다발·소화전 부족 지역\n- 유지관리 대상(✨/✅): 소화전 수준 양호, 운영·접근성 점검 필요")

    # 지역별 상세 현황
    grouped = (
        df_sido.groupby("시도명_full")[
            ["면적_B", "소화전개소_A", "소화전_밀도_D", "화재발생건수_C", "취약_지수"]
        ]
        .agg({
            "면적_B": "sum",
            "소화전개소_A": "sum",
            "소화전_밀도_D": "mean",
            "화재발생건수_C": "sum",
            "취약_지수": "mean",
        })
        .reset_index()
        .sort_values(by="취약_지수", ascending=False)
    )
    st.dataframe(grouped.rename(columns={"시도명_full": "지역"}), use_container_width=True)

    # 예방 관점 분석
    st.subheader("🔎 예방 관점 분석")
    low_fire_low_hydrant = df_sido[(df_sido["화재발생건수_C"] < median_fire) & (df_sido["소화전_밀도_D"] < median_density)]
    st.markdown("**예방 필요 지역:** 화재는 적지만 소화전 밀도 낮은 지역 (우선 점검·교육 대상)")
    st.dataframe(low_fire_low_hydrant[["시도명", "화재발생건수_C", "소화전_밀도_D", "취약_지수"]].head(10), use_container_width=True)

# ------------------------------------------
# Tab 3 — 우선순위 정책 제언
# ------------------------------------------
elif menu == "3️⃣ Phase 3 | 해결 전략 & 우선순위 제언":
    st.header("💡 Phase 3: 우선순위 지역 식별 및 정책 제언")

    # 개요
    st.subheader("📌 개요")
    st.markdown("Phase 2 결과를 바탕으로 우선 등급을 부여하고 정책적 우선순위를 제시합니다.")

    # 목적
    st.subheader("🎯 목적")
    st.markdown("""
    - 취약지수 기반 우선등급(A/B) 설정
    - 취약지역(상위) 간단 설명 및 정책 권고
    """)

    # 데이터 분석 방법
    st.subheader("📊 분석 방법")
    st.markdown("""
    **우선 등급 기준:**
    - A 등급: 취약지수 상위 30% (우선 투자 및 보강)
    - B 등급: 그 외(정기 점검·유지)
    **분류 기준:** 취약지수 및 화재 발생 빈도 기반
    """)

    st.divider()

    # 도출된 분석 결과
    st.subheader("📈 도출된 분석 결과")

    # 우선 등급 분류
    percentile_70 = df_sido["취약_지수"].quantile(0.7)
    df_sido["출동_우선등급"] = df_sido["취약_지수"].apply(
        lambda x: "A (핵심)" if x >= percentile_70 else "B (일반)"
    )

    st.write("**우선순위 지역 분류:**")
    st.dataframe(
        df_sido[["시도명", "출동_우선등급", "취약_지수", "화재_발생_밀도_E", "소화전_밀도_D"]]
        .sort_values(["출동_우선등급", "취약_지수"], ascending=[True, False])
        .reset_index(drop=True),
        width="stretch",
    )

    st.divider()

    # 상위 3개 지역 강조 + 간단 설명
    st.write("**🏆 취약도 상위 3개 지역:**")
    top3 = df_sido.sort_values(by="취약_지수", ascending=False).head(3)

    col_t1, col_t2, col_t3 = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with [col_t1, col_t2, col_t3][idx]:
            st.error(f"{idx+1}위: {row['시도명']}")
            st.metric("취약지수", f"{row['취약_지수']:.2f}")
            st.write(
                f"- 화재: {row['화재발생건수_C']}건 ({row['화재_발생_밀도_E']:.2f}건/km²)\n"
                f"- 소화전: {row['소화전개소_A']}개 ({row['소화전_밀도_D']:.2f}개/km²)"
            )
            st.markdown("*간단 설명:* 취약지수 상위 지역은 화재 빈도와 소화전 밀도 불균형으로 단기 보강 우선 필요")

    st.divider()

    # 취약지수 기반 지도
    st.write("**취약지수 기반 지도**")
    mean_vuln = df_sido["취약_지수"].mean()
    df_above = df_sido[df_sido["취약_지수"] > mean_vuln]
    st.info(f"선정 기준: 취약지수 > {mean_vuln:.2f} (전국 평균) | 대상: {len(df_above)}개 지역)")
    if not df_above.empty:
        render_vulnerability_map(df_above)
    else:
        st.info("평균보다 높은 취약 지역이 없습니다.")

    st.divider()

    # 개선 방향 및 대안 (A/B 등급 중심)
    st.subheader("🛡️ 개선 방향 및 대안")
    col_strat_a, col_strat_b = st.columns(2)
    with col_strat_a:
        st.markdown("""
        **A 등급 (핵심 지역)**
        - 기준: 취약지수 상위 30%
        - 조치: 소화전 신규 설치·우선 투자, 현장 접근성 개선, 집중 예방교육
        """)
    with col_strat_b:
        st.markdown("""
        **B 등급 (일반 지역)**
        - 기준: 취약지수 하위 70%
        - 조치: 정기 점검·유지관리, 노후 소화전 교체, 지역별 예방 캠페인
        """)

    st.divider()

    # 데이터 활용 관점 (명확화)
    st.subheader("📌 데이터 활용 관점")
    st.markdown("""
    - **운영 최적화:** 실시간 경보·우선 출동 목록 생성, 정기 순찰 스케줄링
    - **투자 계획:** 취약지수 기반 인프라 투자 우선순위 산정(예산 근거)
    - **예방 전략:** 예방교육·캠페인 대상 선별, 취약세대 집중점검
    - **정책 개선:** 소화전 배치거리·기준 재검토, 법적 기준 보완 제안
    - **기술 보완:** IoT 센서·원격 감시로 초기경보 체계 연계
    """)
