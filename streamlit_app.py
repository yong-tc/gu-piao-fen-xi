import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import glob

st.set_page_config(page_title="A股选股结果展示", layout="wide")
st.title("📈 A股量价战法选股系统 - 历史结果")

# 获取所有结果文件
result_files = sorted(glob.glob('results/*.csv'), reverse=True)

if not result_files:
    st.info("暂无扫描结果，请等待首次扫描完成")
else:
    # 选择日期
    file_dates = [f.replace('results/', '').replace('.csv', '') for f in result_files]
    selected_date = st.selectbox("选择日期", file_dates)
    
    # 读取结果
    df = pd.read_csv(f"results/{selected_date}.csv")
    
    st.subheader(f"🏆 {selected_date} 选股结果（共 {len(df)} 只）")
    st.dataframe(df, use_container_width=True)
    
    # 统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("入选股票数", len(df))
    with col2:
        st.metric("最高评分", df['评分'].max())
    with col3:
        st.metric("平均评分", round(df['评分'].mean(), 1))
    with col4:
        buy_count = len(df[df['操作建议'] == '买入'])
        st.metric("买入信号", buy_count)
    
    # 下载按钮
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button("📥 下载结果", csv, f"{selected_date}.csv", "text/csv")
    
    # 历史趋势图
    st.subheader("📊 历史选股数量趋势")
    history_data = []
    for f in result_files[:30]:
        date = f.replace('results/', '').replace('.csv', '')
        temp_df = pd.read_csv(f)
        history_data.append({'日期': date, '数量': len(temp_df), '平均分': temp_df['评分'].mean()})
    
    if history_data:
        history_df = pd.DataFrame(history_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_df['日期'], y=history_df['数量'], mode='lines+markers', name='入选数量'))
        fig.add_trace(go.Scatter(x=history_df['日期'], y=history_df['平均分'], mode='lines+markers', name='平均评分', yaxis='y2'))
        fig.update_layout(
            height=400,
            title="历史选股统计",
            yaxis=dict(title="入选数量"),
            yaxis2=dict(title="平均评分", overlaying='y', side='right')
        )
        st.plotly_chart(fig, use_container_width=True)

with st.expander("📖 说明"):
    st.markdown("""
    - 每日自动扫描时间：16:30（收盘后）
    - 扫描范围：沪深A股+创业板（约5000只）
    - 选股逻辑：风险排除 + 量价战法
    - 结果自动保存，可随时查看历史
    """)
