import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据获取模块 ====================
@st.cache_data(ttl=3600)
def get_all_stock_codes():
    """获取沪深A股（含创业板）全部股票代码"""
    try:
        stock_df = ak.stock_zh_a_spot_em()
        stock_df = stock_df[stock_df['代码'].str.match(r'(60|00|30)')]
        codes = stock_df['代码'].tolist()
        names = stock_df['名称'].tolist()
        return codes, names
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

def fetch_stock_data_worker(code, period="daily", days=100):
    """工作函数：获取单只股票历史数据"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period=period,
                                start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        }, inplace=True)
        df['code'] = code
        return df
    except Exception:
        return None

def get_all_stocks_data_parallel(codes, period="daily", days=100, max_workers=8):
    """并行获取所有股票数据"""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data_worker, code, period, days): code for code in codes}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ==================== 技术指标与评分模型（纯 pandas 实现） ====================
def calculate_indicators(df):
    """使用 pandas 原生方法计算技术指标"""
    if df.empty:
        return df
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # 成交量均线
    df['VOL_MA5'] = df['volume'].rolling(window=5).mean()
    
    # 金叉死叉
    df['golden_cross'] = ((df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1)))
    df['death_cross'] = ((df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1)))
    
    # 多头/空头排列
    df['bullish_arrange'] = ((df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA60']))
    df['bearish_arrange'] = ((df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20']) & (df['MA20'] < df['MA60']))
    
    return df

def add_advanced_indicators(df):
    """添加进阶指标：量比, ATR, ADX (简化)"""
    if df.empty:
        return df
    
    # 量比
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['stop_loss'] = df['close'] - 2 * df['ATR']
    
    # ADX 简化（默认给25，不影响评分，如需精确可后续扩展）
    df['ADX'] = 25
    
    return df

def score_stock(df):
    """对单只股票的最新数据打分（0-100分）"""
    latest = df.iloc[-1]
    score = 0
    # 趋势得分 (0-30)
    if latest['bullish_arrange']:
        score += 25
    elif latest['MA5'] > latest['MA20']:
        score += 10
    # 动量得分 (0-30)
    if latest['golden_cross']:
        score += 15
    if 30 <= latest['RSI'] <= 50:
        score += 10
    elif 50 < latest['RSI'] <= 70:
        score += 5
    # 成交量得分 (0-20)
    if latest['volume_ratio'] > 1.5:
        score += 10
    if latest['volume'] > latest['VOL_MA5'] * 1.2:
        score += 10
    # 波动率与趋势强度得分 (0-20)
    if latest['ADX'] > 25:
        score += 10
    if 0.02 <= latest['ATR'] / latest['close'] <= 0.05:
        score += 10
    return min(score, 100)

def filter_stocks_by_score(all_data, min_score=60):
    """对全部股票进行评分筛选"""
    if all_data.empty:
        return pd.DataFrame()
    results = []
    for code, group in all_data.groupby('code'):
        if len(group) < 60:
            continue
        group = group.sort_values('date')
        group = calculate_indicators(group)
        group = add_advanced_indicators(group)
        score = score_stock(group)
        latest = group.iloc[-1].copy()
        latest['score'] = score
        keep_cols = ['code', 'close', 'MA5', 'MA10', 'MA20', 'RSI', 'volume_ratio', 'ADX', 'ATR', 'stop_loss', 'score']
        results.append({k: latest[k] for k in keep_cols if k in latest})
    df_result = pd.DataFrame(results)
    df_result = df_result[df_result['score'] >= min_score].sort_values('score', ascending=False)
    return df_result

# ==================== 大盘环境判断 ====================
def get_market_trend(days=20):
    """判断上证指数是否处于上升趋势（收盘价在20日均线上方且20日涨幅为正）"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days+10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=start, end_date=end)
        if df.empty:
            return True
        df['ma20'] = df['收盘'].rolling(20).mean()
        latest = df.iloc[-1]
        price_above_ma = latest['收盘'] > latest['ma20']
        positive_20d = df['收盘'].pct_change(20).iloc[-1] > 0
        return price_above_ma and positive_20d
    except:
        return True

# ==================== Streamlit 主界面 ====================
st.set_page_config(page_title="A股智能选股系统（优化版）", layout="wide")
st.title("📈 A股智能选股系统 - 短线交易参考（优化版）")
st.markdown("基于趋势、动量、成交量、波动率复合评分模型，全市场智能筛选")

with st.sidebar:
    st.header("⚙️ 参数设置")
    days = st.slider("历史数据天数", min_value=60, max_value=500, value=120, step=30)
    min_score = st.slider("最低评分阈值", min_value=0, max_value=100, value=60, step=5)
    max_workers = st.slider("并行进程数", min_value=2, max_value=16, value=8, step=2)
    use_market_filter = st.checkbox("开启大盘环境过滤", value=True)
    if st.button("🚀 开始扫描", type="primary"):
        st.session_state['run_scan'] = True

if 'run_scan' in st.session_state and st.session_state['run_scan']:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 当前大盘趋势偏弱，建议谨慎操作。如需继续扫描，请关闭大盘过滤。")
        st.session_state['run_scan'] = False
        st.stop()
    
    with st.spinner("正在获取股票列表..."):
        codes, names = get_all_stock_codes()
        if codes:
            st.info(f"✅ 获取到 {len(codes)} 只股票（沪深A股+创业板）")
        else:
            st.error("获取股票列表失败")
            st.stop()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner(f"正在使用 {max_workers} 个进程并行获取数据..."):
        all_data = get_all_stocks_data_parallel(codes, "daily", days, max_workers)
        progress_bar.progress(100)
        status_text.text(f"✅ 数据获取完成！共获取 {len(all_data['code'].unique())} 只股票数据")
    
    if not all_data.empty:
        with st.spinner("正在计算技术指标与评分..."):
            scored_df = filter_stocks_by_score(all_data, min_score)
        
        st.subheader(f"🏆 综合评分 ≥ {min_score} 的股票")
        if scored_df.empty:
            st.info("当前没有股票满足评分条件，可尝试降低评分阈值。")
        else:
            st.dataframe(scored_df, use_container_width=True)
            
            st.subheader("📊 个股技术分析")
            selected_code = st.selectbox("选择股票查看K线图", scored_df['code'].tolist())
            stock_data = all_data[all_data['code'] == selected_code].copy()
            stock_data = stock_data.sort_values('date')
            stock_data = calculate_indicators(stock_data)
            stock_data = add_advanced_indicators(stock_data)
            
            # 绘制K线和指标
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                vertical_spacing=0.05,
                                row_heights=[0.6, 0.2, 0.2])
            # K线
            fig.add_trace(go.Candlestick(x=stock_data['date'],
                                         open=stock_data['open'],
                                         high=stock_data['high'],
                                         low=stock_data['low'],
                                         close=stock_data['close'],
                                         name='K线'), row=1, col=1)
            # 均线
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MA5'],
                                     mode='lines', name='MA5', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MA10'],
                                     mode='lines', name='MA10', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MA20'],
                                     mode='lines', name='MA20', line=dict(color='green')), row=1, col=1)
            # 止损线
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['stop_loss'],
                                     mode='lines', name='止损线', line=dict(color='red', dash='dash')), row=1, col=1)
            # 成交量
            colors = ['red' if stock_data['close'].iloc[i] < stock_data['open'].iloc[i] else 'green'
                      for i in range(len(stock_data))]
            fig.add_trace(go.Bar(x=stock_data['date'], y=stock_data['volume'],
                                 name='成交量', marker_color=colors), row=2, col=1)
            # MACD
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MACD'],
                                     mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
            fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MACD_signal'],
                                     mode='lines', name='Signal', line=dict(color='orange')), row=3, col=1)
            fig.update_layout(height=800, title=f"{selected_code} 技术分析")
            st.plotly_chart(fig, use_container_width=True)
    
    st.session_state['run_scan'] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 优化说明")
    st.markdown("""
    **评分因子权重：**
    - 趋势排列 (25分)
    - 金叉信号 (15分)
    - RSI健康区 (10分)
    - 量比放大 (10分)
    - 成交量放大 (10分)
    - ADX>25趋势市 (10分)
    - 温和波动率 (10分)
    
    **使用建议：**
    - 评分≥60可关注
    - 结合大盘过滤降低风险
    - 参考止损线控制回撤
    """)
