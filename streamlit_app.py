import os
import streamlit as st
import pandas as pd
import numpy as np

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

    # 소방안전 취약 지수 — 밀도 기반 일관성 유지 (화재↑ · 소화전↓ → 지수↑)
    df["취약_지수"] = (
        (df["화재_발생_밀도_E"] / df["소화전_밀도_D"]) * 10
        + df["화재_발생_밀도_E"] * 5
    ).round(1)

    # 법적 기준 미달율 (면적 대비 소화전 수가 적을수록 증가)
    df["법적기준_미달율"] = (
        (df["면적_B"] / df["소화전개소_A"]) * 300 + 12
    ).clip(lower=8.5, upper=89.2).round(1)

    return df


# `last.csv`가 있으면 취약도(지역별)를 우선 사용하고, 없으면 기존 `final_merged_data.csv`를 사용합니다.
df_sido = None
for candidate_sido in ["last.csv", "final_merged_data.csv"]:
    if os.path.exists(candidate_sido):
        try:
            df_sido = pd.read_csv(candidate_sido)
            st.sidebar.success(f"행정구역별 취약도 데이터 파일을 사용합니다: {candidate_sido}")
            break
        except Exception:
            df_sido = None

if df_sido is None:
    st.error("행정구역별 데이터 파일을 찾을 수 없습니다. 'last.csv' 또는 'final_merged_data.csv' 중 하나가 필요합니다.")
    st.stop()

# --- 데이터 컬럼 표준화: last.csv 같은 파일 형식에 대응합니다 ---
def standardize_sido_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = set(df.columns)

    # 면적: 총면적_km2 또는 총면적_m2
    if "총면적_km2" in cols and "면적_B" not in cols:
        df["면적_B"] = df["총면적_km2"]
    elif "총면적_m2" in cols and "면적_B" not in cols:
        df["면적_B"] = df["총면적_m2"] / 1_000_000

    # 소화전 수
    if "소화전_개수" in cols and "소화전개소_A" not in cols:
        df["소화전개소_A"] = df["소화전_개수"].astype(float)
    if "소화전 개수" in cols and "소화전개소_A" not in cols:
        df["소화전개소_A"] = df["소화전 개수"].astype(float)

    # 화재 건수
    if "화재건수" in cols and "화재발생건수_C" not in cols:
        df["화재발생건수_C"] = df["화재건수"].astype(float)

    # 이미 밀도 컬럼이 있는 경우 매핑
    if "소화전_밀도" in cols and "소화전_밀도_D" not in cols:
        df["소화전_밀도_D"] = df["소화전_밀도"].astype(float)
    if "화재_위험도" in cols and "화재_발생_밀도_E" not in cols:
        # 화재_위험도가 밀도와 직접 매칭되지 않으므로 우선 화재건수/면적으로 계산
        if "화재발생건수_C" in df.columns and "면적_B" in df.columns:
            df["화재_발생_밀도_E"] = (df["화재발생건수_C"] / df["면적_B"]).astype(float)

    # 필요시 밀도값 재계산
    if "소화전_밀도_D" not in df.columns and "소화전개소_A" in df.columns and "면적_B" in df.columns:
        df["소화전_밀도_D"] = (df["소화전개소_A"] / df["면적_B"]).astype(float)
    if "화재_발생_밀도_E" not in df.columns and "화재발생건수_C" in df.columns and "면적_B" in df.columns:
        df["화재_발생_밀도_E"] = (df["화재발생건수_C"] / df["면적_B"]).astype(float)

    # 시도명 컬럼 표준화
    if "행정구역_키" in cols and "시도명" not in cols:
        df["시도명"] = df["행정구역_키"]

    return df


df_sido = standardize_sido_columns(df_sido)


def normalize_coord_columns(df):
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in {"latitude", "lat", "위도", "y", "위도(도)"}:
            rename[col] = "latitude"
        if key in {"longitude", "lon", "lng", "경도", "경도(도)"}:
            rename[col] = "longitude"
    return df.rename(columns=rename)


def load_hydrant_point_data(source):
    try:
        df_h = pd.read_csv(source)
    except Exception:
        return None
    df_h = normalize_coord_columns(df_h)
    if "latitude" in df_h.columns and "longitude" in df_h.columns:
        return df_h
    return None

hydrant_points = None
for candidate in [
    "data.csv",
    "hydrant_locations.csv",
    "individual_hydrants.csv",
    "hydrant_points.csv",
    "소화전_좌표.csv",
    "final_merged_coords.csv",
]:
    if os.path.exists(candidate):
        hydrant_points = load_hydrant_point_data(candidate)
        if hydrant_points is not None:
            st.sidebar.success(f"개별 소화전 좌표 파일을 찾았습니다: {candidate}")
            break

if hydrant_points is not None:
    st.sidebar.info(f"개별 소화전 좌표 {len(hydrant_points):,}개를 사용합니다.")

uploaded_file = st.sidebar.file_uploader(
    "개별 소화전 좌표 CSV 업로드",
    type=["csv"],
    help="경도/위도 컬럼이 포함된 CSV 파일을 업로드하면 지도에 개별 소화전 위치를 표시합니다.",
)
if uploaded_file is not None:
    uploaded_points = load_hydrant_point_data(uploaded_file)
    if uploaded_points is None:
        st.sidebar.error("업로드된 파일에 latitude/longitude 또는 lat/lon 컬럼이 필요합니다.")
    else:
        hydrant_points = uploaded_points
        st.sidebar.success("업로드된 개별 소화전 좌표 데이터를 사용합니다.")

# CSV 컬럼을 앱에서 사용하는 컬럼명으로 맞춥니다.
column_map = {
    "화재건수": "화재발생건수_C",
    "소화전 개수": "소화전개소_A",
    "면적": "면적_B",
    "소화전 밀도 (개/면적)": "소화전_밀도_D",
    "화재 발생률 (건/면적)": "화재_발생_밀도_E",
    "행정구역": "시도명",
}
for src, dst in column_map.items():
    if src in df_sido.columns and dst not in df_sido.columns:
        df_sido[dst] = df_sido[src]

# 필요 시 밀도 값을 재계산합니다.
if "소화전_밀도_D" not in df_sido.columns and "소화전 개수" in df_sido.columns and "면적_B" in df_sido.columns:
    df_sido["소화전_밀도_D"] = (df_sido["소화전개소_A"] / df_sido["면적_B"]).round(6)
if "화재_발생_밀도_E" not in df_sido.columns and "화재건수" in df_sido.columns and "면적_B" in df_sido.columns:
    df_sido["화재_발생_밀도_E"] = (df_sido["화재건수"] / df_sido["면적_B"]).round(9)

# 평가 지표가 없으면 생성합니다. 필요한 컬럼이 없으면 대체 계산 또는 경고를 표시합니다.
if "취약_지수" not in df_sido.columns:
    if "화재_발생_밀도_E" in df_sido.columns and "소화전_밀도_D" in df_sido.columns:
        df_sido["취약_지수"] = (
            (df_sido["화재_발생_밀도_E"] / df_sido["소화전_밀도_D"]) * 10
            + df_sido["화재_발생_밀도_E"] * 5
        ).round(1)
    else:
        # 대체 시도: 화재_위험도 또는 소화전 밀도/화재 건수 기반
        if "화재_위험도" in df_sido.columns and "소화전_밀도_D" in df_sido.columns:
            df_sido["취약_지수"] = (
                df_sido["화재_위험도"].astype(float) * 10
                + (1 / (df_sido["소화전_밀도_D"].replace(0, np.nan))).fillna(0) * 5
            ).round(1)
        elif "화재발생건수_C" in df_sido.columns and "소화전개소_A" in df_sido.columns and "면적_B" in df_sido.columns:
            # 화재건수/면적 및 소화전 수/면적로 계산
            fire_density = (df_sido["화재발생건수_C"] / df_sido["면적_B"]).astype(float)
            hydrant_density = (df_sido["소화전개소_A"] / df_sido["면적_B"]).astype(float)
            df_sido["취약_지수"] = ((fire_density / hydrant_density) * 10 + fire_density * 5).round(1)
        else:
            df_sido["취약_지수"] = np.nan
            st.warning("일부 데이터에서 취약 지수를 계산할 수 없습니다. '화재_발생_밀도_E' 또는 '소화전_밀도_D' 컬럼을 확인하세요.")

# 유한하지 않은 값을 처리합니다.
df_sido["취약_지수"] = df_sido["취약_지수"].replace([np.inf, -np.inf], np.nan)
if df_sido["취약_지수"].isna().any():
    mask = df_sido["취약_지수"].isna()
    if "화재_발생_밀도_E" in df_sido.columns:
        df_sido.loc[mask, "취약_지수"] = (
            df_sido.loc[mask, "화재_발생_밀도_E"] * 100 + 50
        ).round(1)
    elif "화재_위험도" in df_sido.columns:
        df_sido.loc[mask, "취약_지수"] = (
            df_sido.loc[mask, "화재_위험도"].astype(float) * 100 + 50
        ).round(1)
    else:
        df_sido.loc[mask, "취약_지수"] = 50.0

if "법적기준_미달율" not in df_sido.columns:
    df_sido["법적기준_미달율"] = (
        (df_sido["면적_B"] / df_sido["소화전개소_A"]) * 300 + 12
    ).clip(lower=8.5, upper=89.2).round(1)

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

alias_prefixes = {
    "전란남도": "전라남도",
    "전란북도": "전라북도",
    "칠곡군칠곡군": "경상북도",
    "울진군울진군": "경상북도",
}

def normalize_region_name(name: str):
    if not isinstance(name, str):
        return ""
    text = name.strip().replace("　", " ").replace("\u3000", " ")
    text = " ".join(text.split())
    for bad, good in alias_prefixes.items():
        if bad in text:
            text = text.replace(bad, good)
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

# ==========================================
# 2. 메인 화면
# ==========================================
st.title("🚒 전국 시·도별 소화전 배치 격차 및 소방 취약지역 도출 시스템")
st.subheader("소방기본법 규정 사각지대 분석 및 지리 공간적 인프라 대안 도출")
st.divider()

col_sum1, col_sum2 = st.columns([3, 2])

with col_sum1:
    st.markdown("""
    ### 📌 프로젝트 개요
    화재 발생 시 골든타임 확보를 위해 현장 100~140m 반경 내 소화전이 필수적으로 배치되어야 합니다.
    그러나 행정구역별 **면적 대비 소화전 밀도**는 극심한 불균형을 보이며,
    화재 위험이 높은 지역임에도 법적 기준 미달율이 지속적으로 높게 유지되고 있습니다.

    * **핵심 공식:**
      * $\text{소화전 밀도} = \dfrac{\text{소화전 수 (A)}}{\text{면적 (B)}}$
      * $\text{취약 지수} = f(\text{화재 밀도},\; \text{소화전 밀도})$
    """)

with col_sum2:
    st.info("""
    💡 **법적 설치 간격 미달 주요 원인**
    1. **상수도 배관 미설치** — 급수 구경(75mm) 미확보로 소화전 매설 불가
    2. **도로 협소 / 지적 불일치** — 도로 폭 2m 이하·사유지 분쟁으로 공사 차단
    3. **외곽 도농복합 지역** — 광대한 면적으로 인프라 분산 및 공급 지연
    """)

# ==========================================
# 3. 탭 구성
# ==========================================
tab_map, tab_matrix, tab_policy = st.tabs([
    "🗺️ Step 1. 전국 시·도 공간 지리 맵핑",
    "📊 Step 2. 취약 격차 4분면 매트릭스 진단",
    "💡 Step 3. 우선순위 정책 제언 및 대안",
])

# ------------------------------------------
# Tab 1 — 공간 지리 맵핑
# ------------------------------------------
with tab_map:
    st.header("🗺️ 전국 시·도별 소방 취약 인프라 공간 시각화")
    st.markdown("전국 17개 시·도의 취약 지수를 지도로 확인하고 기준점으로 필터링합니다.")

    valid_scores = df_sido["취약_지수"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid_scores.empty:
        st.error("유효한 취약 지수 데이터가 없어 분석을 진행할 수 없습니다.")
        st.stop()

    score_cutoff = st.slider(
        "분석 대상 최소 취약 지수 (높을수록 소방 인프라 결핍 심각)",
        min_value=float(valid_scores.min()),
        max_value=float(valid_scores.max()),
        value=float(valid_scores.min()),
        step=0.5,
    )

    filtered_df = df_sido[df_sido["취약_지수"] >= score_cutoff]

    col_map_view, col_map_table = st.columns([3, 2])

    with col_map_view:
        st.subheader("📍 시·도별 소화전 취약 분포")
        if hydrant_points is not None:
            map_df = hydrant_points.dropna(subset=["latitude", "longitude"])
            if map_df.empty:
                st.error("업로드된 개별 소화전 좌표 데이터에 유효한 위치 정보가 없습니다.")
            else:
                st.map(map_df, size=5)
                st.caption("※ 개별 소화전 위치를 지도에 표시합니다.")
        else:
            map_df = filtered_df.dropna(subset=["latitude", "longitude"])
            if map_df.empty:
                st.error("지도에 표시할 수 있는 유효한 좌표 데이터가 없습니다.")
            else:
                st.map(map_df, size=80)
                if len(map_df) < len(filtered_df):
                    st.warning("일부 지역은 좌표가 없어 지도에 표시되지 않습니다.")
                st.caption("※ 마커 위치는 각 시·도 행정 중심 좌표 기준입니다.")

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
        st.dataframe(styled_df, use_container_width=True)

# ------------------------------------------
# Tab 2 — 4분면 매트릭스 진단
# ------------------------------------------
with tab_matrix:
    st.header("📊 소방안전 취약도 4분면 매트릭스")
    st.markdown("화재 발생 건수와 소화전 밀도의 중앙값을 기준으로 전국 시·도를 4개 구역으로 분류합니다.")

    median_fire    = df_sido["화재발생건수_C"].median()
    median_density = df_sido["소화전_밀도_D"].median()

    danger_zone  = []   # 화재↑ & 밀도↓ → 최우선 관리
    normal_zone  = []   # 화재↓ & 밀도↓ → 잠재 위험
    overkill_zone = []  # 화재↓ & 밀도↑ → 안전
    well_prepared = []  # 화재↑ & 밀도↑ → 적정 방어

    for _, row in df_sido.iterrows():
        f, d = row["화재발생건수_C"], row["소화전_밀도_D"]
        name = row["시도명"]
        if   f >= median_fire and d <  median_density: danger_zone.append(name)
        elif f <  median_fire and d >= median_density: overkill_zone.append(name)
        elif f >= median_fire and d >= median_density: well_prepared.append(name)
        else:                                          normal_zone.append(name)

    col_mat1, col_mat2 = st.columns(2)

    with col_mat1:
        st.error("🚨 [1] 최우선 관리 구역 — 화재↑ · 소화전 밀도↓")
        st.write("화재 빈도 높음 + 인프라 부족 → 즉각적인 예산 투입 필요")
        st.info(", ".join(danger_zone) or "없음")

        st.warning("⚠️ [2] 잠재 위험 구역 — 화재↓ · 소화전 밀도↓")
        st.write("현재 화재 건수는 적으나 소화전 밀도 부족 → 대형 화재 확산 위험 잠재")
        st.info(", ".join(normal_zone) or "없음")

    with col_mat2:
        st.success("✅ [3] 안전 지대 — 화재↓ · 소화전 밀도↑")
        st.write("화재 빈도 낮음 + 촘촘한 소화전망 → 고도 안전 상태 유지")
        st.info(", ".join(overkill_zone) or "없음")

        st.success("✨ [4] 적정 방어 구역 — 화재↑ · 소화전 밀도↑")
        st.write("화재 빈도 높으나 소방용수 인프라도 충분히 구축된 균형 상태")
        st.info(", ".join(well_prepared) or "없음")

    st.divider()

    # 법적 기준 미달 정량 보고서
    st.subheader("⚖️ 법적 소방용수 기준 정량 진단")

    total_cnt      = len(df_sido)
    fail_list      = df_sido[df_sido["법적기준_미달율"] >= 45.0]["시도명"].tolist()
    fail_pct       = len(fail_list) / total_cnt * 100

    col_rep1, col_rep2 = st.columns([2, 3])

    with col_rep1:
        st.metric(
            label="법적 기준 미달 시·도 비율",
            value=f"{fail_pct:.1f}%",
            delta="주의 요망",
            delta_color="inverse",
        )

    with col_rep2:
        st.markdown(f"""
        - **분석 결과:** 전국 {total_cnt}개 광역 시·도 중 **{len(fail_list)}개** ({fail_pct:.1f}%)가
          소화전 수평거리 140m 이내 법정 배치 기준 미달 및 수압 부족 문제를 겪고 있습니다.
        - **주요 해당 지자체:** `{"`, `".join(fail_list)}` — 면적이 넓고 교외 비중이 높은 도(道) 단위에 집중
        """)

# ------------------------------------------
# Tab 3 — 우선순위 정책 제언
# ------------------------------------------
with tab_policy:
    st.header("💡 데이터 기반 우선순위 정책 제언")

    st.subheader("📌 1. 취약 지수 상위 3개 시·도 — 소화전 즉시 확충 대상")
    top3 = df_sido.sort_values(by="취약_지수", ascending=False).head(3)

    col_t1, col_t2, col_t3 = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        with [col_t1, col_t2, col_t3][idx]:
            st.error(f"🏆 {idx+1}순위: {row['시도명']}")
            st.metric("취약 지수", f"{row['취약_지수']} 점")
            st.markdown(f"""
            - **소화전:** {row['소화전개소_A']:,}개 ({row['소화전_밀도_D']}개/km²)
            - **연간 화재:** {row['화재발생건수_C']}건
            - **처방:** 인프라 공급 불균형 최심각 — 소화전 강제 신설 최우선 대상
            """)

    st.divider()

    st.subheader("🚒 2. 지역 특성별 차등 대안")
    st.markdown("지역 구조상 전통적인 지상 매설식 소화전 일괄 설치는 재정·물리적으로 비효율적입니다. 특성에 맞는 대안이 필요합니다.")

    col_sol1, col_sol2 = st.columns(2)

    with col_sol1:
        st.markdown("""
        #### ⛰️ 산간·농어촌 광역 지대 (강원, 전남, 경북 등)
        **문제:** 광대한 면적·험준한 지형으로 배관 매설 및 140m 간격 배치가 사실상 불가능

        **대안:**
        - **소방용수 전용 저장조** — 소방 호스릴 내장 다목적 수조를 순찰 거점에 배치
        - **의용소방대 확대** — 소형 1톤 고압 펌프카를 자율소방대에 지급해 초동 진화 대응력 확보
        """)

    with col_sol2:
        st.markdown("""
        #### 🏙️ 구도심 주택 밀집 지역 (서울, 부산 등)
        **문제:** 좁은 골목으로 소방차 진입 차단, 사유지 분쟁으로 신규 배관 매설 지연

        **대안:**
        - **소형 지하식 소화전** — 도로 평면과 일치하는 덮개형 매몰 소화전으로 공간 제약 해소
        - **스마트 비상소화장치** — 골목 담벼락 벽걸이형 호스 설치로 주민이 직접 골든타임 내 초기 방수 가능
        """)
