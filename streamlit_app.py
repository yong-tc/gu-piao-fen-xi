import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="A股选股结果", layout="wide")
st.title("📈 A股智能选股系统 - 每日精选")

# 唯一的数据来源：读取 CSV 文件
@st.cache_data(ttl=3600)
def load_results():
    try:
        df = pd.read_csv('top_stocks.csv', encoding='utf-8')
        return df
    except FileNotFoundError:
        return pd.DataFrame()

df = load_results()

if df.empty:
    st.info("暂无数据，请等待每日扫描完成（工作日15:30后更新）")
else:
    # 直接展示已经计算好的结果
    st.dataframe(df, use_container_width=True)
    
    # 简单的图表（基于已评分的数据）
    fig = go.Figure(data=[go.Bar(x=df['代码'], y=df['评分'])])
    st.plotly_chart(fig)
