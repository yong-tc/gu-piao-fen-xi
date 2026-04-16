import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== 重试装饰器 ====================
def is_retryable_exception(exception):
    return isinstance(exception, (requests.exceptions.RequestException, ConnectionError, TimeoutError))

@retry(stop=stop_after_attempt(3), 
       wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(is_retryable_exception))
def safe_fetch_hist_data(code, period, start_date, end_date):
    return ak.stock_zh_a_hist(symbol=code, period=period,
                              start_date=start_date, end_date=end_date, adjust="qfq")

# ==================== 数据获取模块 ====================
@st.cache_data(ttl=86400)
def get_all_stock_codes():
    """优先从本地CSV文件读取股票列表，若失败则联网获取"""
    try:
        df = pd.read_csv('stock_list.csv', encoding='utf-8')
        if '代码' in df.columns and '名称' in df.columns:
            return df['代码'].tolist(), df['名称'].tolist()
        else:
            st.warning("stock_list.csv 列名不正确，使用网络获取")
    except FileNotFoundError:
        st.info("未找到本地股票列表，正在从网络获取...")
    except Exception as e:
        st.warning(f"读取本地股票列表失败（{e}），使用网络获取")
    
    try:
        stock_df = ak.stock_zh_a_spot_em()
        stock_df = stock_df[stock_df['代码'].str.match(r'(60|00|30)')]
        return stock_df['代码'].tolist(), stock_df['名称'].tolist()
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data_worker(code, period="daily", days=100):
    """获取单只股票历史数据（带缓存）"""
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = safe_fetch_hist_data(code, period, start_date, end_date)
        if df.empty:
            return None
        df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        }, inplace=True)
        df['code'] = code
        # 只保留必要列，减少内存
        return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
    except Exception:
        return None

def get_all_stocks_data_parallel(codes, period="daily", days=100, max_workers=8):
    """线程池并行获取数据（限制并发）"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data_worker, code, period, days): code for code in codes}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ==================== 技术指标（向量化，按需计算） ====================
def calculate_indicators(df):
    """一次性计算所有股票的指标，仅保留评分所需的列"""
    if df.empty:
        return df
    # 移动平均线
    df['MA5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['MA10'] = df.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['MA20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['MA60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
    
    # MACD
    def macd_group(group):
        exp1 = group['close'].ewm(span=12, adjust=False).mean()
        exp2 = group['close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp1 - exp2
        group['MACD_signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
        return group
    df = df.groupby('code', group_keys=False).apply(macd_group)
    
    # RSI
    def rsi_group(group):
        delta = group['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))
        return group
    df = df.groupby('code', group_keys=False).apply(rsi_group)
    
    # 成交量均线
    df['VOL_MA5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # 金叉死叉
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df.groupby('code')['MA5'].shift(1) <= df.groupby('code')['MA10'].shift(1))
    df['death_cross'] = (df['MA5'] < df['MA10']) & (df.groupby('code')['MA5'].shift(1) >= df.groupby('code')['MA10'].shift(1))
    
    # 多头/空头排列
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA60'])
    df['bearish_arrange'] = (df['MA5'] < df['MA10']) & (df['MA10'] < df['MA20']) & (df['MA20'] < df['MA60'])
    
    return df

def add_advanced_indicators(df):
    """添加量比、ATR等"""
    if df.empty:
        return df
    # 量比
    df['volume_ratio'] = df.groupby('code')['volume'].transform(lambda x: x / x.rolling(5, min_periods=1).mean())
    
    # ATR
    def atr_group(group):
        high_low = group['high'] - group['low']
        high_close = (group['high'] - group['close'].shift()).abs()
        low_close = (group['low'] - group['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group['ATR'] = tr.rolling(14, min_periods=1).mean()
        group['stop_loss'] = group['close'] - 2 * group['ATR']
        return group
    df = df.groupby('code', group_keys=False).apply(atr_group)
    df['ADX'] = 25  # 简化
    return df

def score_stock(group):
    """对单个股票的整个历史数据评分（取最新一天）"""
    latest = group.iloc[-1]
    score = 0
    if latest['bullish_arrange']:
        score += 25
    elif latest['MA5'] > latest['MA20']:
        score += 10
    if latest['golden_cross']:
        score += 15
    if 30 <= latest['RSI'] <= 50:
        score += 10
    elif 50 < latest['RSI'] <= 70:
        score += 5
    if latest['volume_ratio'] > 1.5:
        score += 10
    if latest['volume'] > latest['VOL_MA5'] * 1.2:
        score += 10
    if latest['ADX'] > 25:
        score += 10
    if 0.02 <= latest['ATR'] / latest['close'] <= 0.05:
        score += 10
    return min(score, 100)

def filter_stocks_by_score(all_data, min_score=60):
    """对全部股票进行评分筛选（内存优化版）"""
    if all_data.empty:
        return pd.DataFrame()
    # 先计算所有指标
    all_data = calculate_indicators(all_data)
    all_data = add_advanced_indicators(all_data)
    
    results = []
    for code, group in all_data.groupby('code'):
        if len(group) < 60:
            continue
        group = group.sort_values('date')
        score = score_stock(group)
        latest = group.iloc[-1].copy()
        latest['score'] = score
        # 只保留评分结果需要的列
        results.append({
            'code': latest['code'],
            'close': latest['close'],
            'MA5': latest['MA5'],
            'MA10': latest['MA10'],
            'MA20': latest['MA20'],
            'RSI': latest['RSI'],
            'volume_ratio': latest['volume_ratio'],
            'ADX': latest['ADX'],
            'ATR': latest['ATR'],
            'stop_loss': latest['stop_loss'],
            'score': score
        })
    # 释放原始数据内存
    del all_data
    gc.collect()
    
    df_result = pd.DataFrame(results)
    df_result = df_result[df_result['score'] >= min_score].sort_values('score', ascending=False)
    return df_result

def get_market_trend(days=20):
    """大盘趋势判断（上证指数）"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days+10)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date=start, end_date=end)
        if df.empty:
            return True
        df['ma20'] = df['收盘'].rolling(20).mean()
        latest = df.iloc[-1]
        return (latest['收盘'] > latest['ma20']) and (df['收盘'].pct_change(20).iloc[-1] > 0)
    except:
        return True

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="A股智能选股系统（内存优化版）", layout="wide")
st.title("📈 A股智能选股系统 - 短线交易参考")
st.markdown("内存优化版：限制扫描数量、仅保留必要列、自动垃圾回收")

with st.sidebar:
    st.header("⚙️ 参数设置（建议值 ≤ 300 只股票）")
    days = st.slider("历史数据天数（越少越快）", 20, 120, 30, 10)
    min_score = st.slider("最低评分阈值", 0, 100, 60, 5)
    max_workers = st.slider("并行线程数", 2, 8, 4, 1)   # 降低最大并发
    max_stocks = st.slider("最大扫描股票数量（建议 ≤ 300）", 50, 1000, 200, 50)
    use_market_filter = st.checkbox("开启大盘过滤", True)
    if st.button("🚀 开始扫描", type="primary"):
        st.session_state['run_scan'] = True

if 'run_scan' in st.session_state and st.session_state['run_scan']:
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state['run_scan'] = False
        st.stop()
    
    with st.spinner("获取股票列表..."):
        codes, names = get_all_stock_codes()
        if not codes:
            st.error("获取股票列表失败")
            st.stop()
        original_count = len(codes)
        if max_stocks > 0 and len(codes) > max_stocks:
            codes = codes[:max_stocks]
            names = names[:max_stocks]
        st.info(f"✅ 共 {original_count} 只股票，本次扫描 {len(codes)} 只（控制内存）")
    
    with st.spinner(f"使用 {max_workers} 个线程并行获取数据（约需1-2分钟）..."):
        all_data = get_all_stocks_data_parallel(codes, "daily", days, max_workers)
        if all_data.empty:
            st.error("数据获取失败，请检查网络或稍后重试")
            st.stop()
        st.success(f"✅ 获取到 {len(all_data['code'].unique())} 只股票数据")
        # 强制垃圾回收
        gc.collect()
    
    with st.spinner("计算指标与评分（向量化+内存回收）..."):
        scored_df = filter_stocks_by_score(all_data, min_score)
        # 释放原始数据，只保留评分结果
        del all_data
        gc.collect()
    
    st.subheader(f"🏆 评分 ≥ {min_score} 的股票（共 {len(scored_df)} 只）")
    if scored_df.empty:
        st.info("无符合条件的股票，可降低评分阈值或增加扫描数量")
    else:
        st.dataframe(scored_df, use_container_width=True)
        
        if len(scored_df) > 0:
            st.subheader("📊 个股详情")
            selected = st.selectbox("选择股票", scored_df['code'].tolist())
            # 注意：为了画图，需要重新获取该股票的详细数据（但仅单只，内存占用极小）
            with st.spinner(f"加载 {selected} 详细数据..."):
                single_df = fetch_stock_data_worker(selected, "daily", days)
                if single_df is not None:
                    stock_data = single_df.sort_values('date')
                    stock_data = calculate_indicators(stock_data)
                    stock_data = add_advanced_indicators(stock_data)
                    
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.05, row_heights=[0.6,0.2,0.2])
                    fig.add_trace(go.Candlestick(x=stock_data['date'],
                                                 open=stock_data['open'], high=stock_data['high'],
                                                 low=stock_data['low'], close=stock_data['close'],
                                                 name='K线'), row=1, col=1)
                    for ma, color in [('MA5','blue'),('MA10','orange'),('MA20','green')]:
                        fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data[ma],
                                                 mode='lines', name=ma, line=dict(color=color)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['stop_loss'],
                                             mode='lines', name='止损线', line=dict(color='red', dash='dash')), row=1, col=1)
                    colors = ['red' if c<o else 'green' for c,o in zip(stock_data['close'], stock_data['open'])]
                    fig.add_trace(go.Bar(x=stock_data['date'], y=stock_data['volume'],
                                         name='成交量', marker_color=colors), row=2, col=1)
                    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MACD'],
                                             mode='lines', name='MACD', line=dict(color='blue')), row=3, col=1)
                    fig.add_trace(go.Scatter(x=stock_data['date'], y=stock_data['MACD_signal'],
                                             mode='lines', name='Signal', line=dict(color='orange')), row=3, col=1)
                    fig.update_layout(height=800, title=f"{selected} 技术分析")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("获取个股数据失败")
    
    st.session_state['run_scan'] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("**内存优化措施**：限制扫描数量 | 只保留必要列 | 主动垃圾回收 | 降低线程数")
    st.markdown("**评分权重**：趋势25 | 金叉15 | RSI 10 | 量比10 | 放量10 | ADX10 | 波动率10")
