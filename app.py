
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title('행정구역별 화재 안전 분석')
st.write('화재 발생률과 소화전 밀도 간의 상관관계를 분석합니다.')

@st.cache_data
def load_data():
    # Ensure the path is correct if running in a different environment
    try:
        df = pd.read_csv('final_merged_data.csv')
        return df
    except Exception:
        return None

df = load_data()

if df is not None:
    st.subheader('데이터 미리보기')
    st.dataframe(df.head())

    st.subheader('화재 소화전 밀도 vs. 화재 발생률 상관관계')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x='소화전 밀도 (개/면적)',
        y='화재 발생률 (건/면적)',
        data=df,
        ax=ax
    )
    
    ax.set_title('행정구역별 화재 소화전 밀도와 화재 발생률 상관관계')
    ax.set_xlabel('소화전 밀도 (개/면적)')
    ax.set_ylabel('화재 발생률 (건/면적)')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Further analysis or charts can be added here
    st.subheader('추가 분석')
    st.write('여기에 추가적인 분석 내용이나 시각화를 구현할 수 있습니다.')
else:
    st.error("데이터를 로드하는 데 실패했습니다.")
