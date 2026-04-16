import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

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
    
    # 回退：从东方财富网获取
    try:
        stock_df = ak.stock_zh_a_spot_em()
        stock_df = stock_df[stock_df['代码'].str.match(r'(60|00|30)')]
        return stock_df['代码'].tolist(), stock_df['名称'].tolist()
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

def fetch_stock_data_worker(code, period="daily", days=100):
    """获取单只股票历史数据（线程安全）"""
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
    """线程池并行获取数据"""
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

# ==================== 技术指标（纯 pandas） ====================
def calculate_indicators(df):
    if df.empty:
        return df
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['MA60'] = df['close'].rolling(60).mean()
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BB_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    df['VOL_MA5'] = df['volume'].rolling(5).mean()
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1))
    df['death_cross'] = (df['MA5'] < df['MA10']) & (df['MA5'].shift(1) >= df['MA10'].shift(1))
    # 短期多头（仅用于短线参考）
    df['short_bullish'] = (df['MA5'] > df['MA10']) & (df['close'] > df['MA5'])
    return df

def add_advanced_indicators(df):
    if df.empty:
        return df
    # 量比（相对于5日均量）
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    # ATR
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    df['stop_loss'] = df['close'] - 2 * df['ATR']
    return df

def score_stock(df):
    """短线交易评分（侧重量价异动、短期趋势）"""
    latest = df.iloc[-1]
    # 计算额外短线指标
    # 5日最高价
    high_5d = df['high'].rolling(5).max().iloc[-1]
    # 5日涨幅
    price_5d_ago = df['close'].iloc[-6] if len(df) >= 6 else df['close'].iloc[0]
    gain_5d = (latest['close'] - price_5d_ago) / price_5d_ago * 100
    
    score = 0
    
    # 1. 短期趋势 (0-20)
    if latest['MA5'] > latest['MA10']:
        score += 10
    if latest['close'] > latest['MA5']:
        score += 10
    
    # 2. 动量与突破 (0-30)
    # 今日涨幅
    daily_return = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100 if len(df) >= 2 else 0
    if daily_return > 3:
        score += 10
    if latest['close'] > high_5d * 0.98:   # 接近5日高点
        score += 10
    if latest['golden_cross']:
        score += 10
    
    # 3. 成交量异动 (0-25)
    if latest['volume_ratio'] > 1.5:
        score += 10
    if len(df) >= 2 and latest['volume'] > df['volume'].iloc[-2] * 2:
        score += 10
    # 连续放量（最近3日成交量均 > 20日均量*1.2）
    vol_ma20 = df['volume'].rolling(20).mean().iloc[-1]
    recent_vol_condition = (df['volume'].iloc[-3:] > vol_ma20 * 1.2).all() if len(df) >= 3 else False
    if recent_vol_condition:
        score += 5
    
    # 4. 位置与风险控制 (0-25)
    if 50 <= latest['RSI'] <= 70:
        score += 10
    elif latest['RSI'] > 80:
        score -= 10
    
    if 5 <= gain_5d <= 15:
        score += 10
    elif gain_5d > 20:
        score -= 10
    
    volatility = latest['ATR'] / latest['close'] * 100
    if 2 <= volatility <= 6:
        score += 5
    elif volatility > 10:
        score -= 5
    
    return max(0, min(100, score))

def filter_stocks_by_score(all_data, min_score=60):
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
        keep_cols = ['code', 'close', 'MA5', 'MA10', 'RSI', 'volume_ratio', 'ATR', 'stop_loss', 'score']
        results.append({k: latest[k] for k in keep_cols if k in latest})
    
    if not results:
        return pd.DataFrame()
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
st.set_page_config(page_title="A股短线选股系统", layout="wide")
st.title("⚡ A股短线交易参考系统")
st.markdown("基于量价异动、短期趋势、突破强度复合评分，聚焦1-5日交易机会")

with st.sidebar:
    st.header("⚙️ 参数设置")
    days = st.slider("历史数据天数", 30, 200, 60, 10)
    min_score = st.slider("最低评分阈值", 0, 100, 65, 5)
    max_workers = st.slider("并行线程数", 2, 16, 6, 2)
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
        st.info(f"✅ 共 {len(codes)} 只股票（沪深+创业板）")
    
    with st.spinner(f"使用 {max_workers} 个线程并行获取数据（约需几分钟）..."):
        all_data = get_all_stocks_data_parallel(codes, "daily", days, max_workers)
        if all_data.empty:
            st.error("数据获取失败")
            st.stop()
        st.success(f"✅ 获取到 {len(all_data['code'].unique())} 只股票数据")
    
    with st.spinner("计算短线指标与评分..."):
        scored_df = filter_stocks_by_score(all_data, min_score)
    
    st.subheader(f"🏆 短线评分 ≥ {min_score} 的股票")
    if scored_df.empty:
        st.info("当前无符合条件的短线标的，可降低评分阈值")
    else:
        st.dataframe(scored_df, use_container_width=True)
        
        st.subheader("📊 个股技术分析")
        selected = st.selectbox("选择股票", scored_df['code'].tolist())
        stock_data = all_data[all_data['code'] == selected].sort_values('date')
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
        fig.update_layout(height=800, title=f"{selected} 短线技术分析")
        st.plotly_chart(fig, use_container_width=True)
    
    st.session_state['run_scan'] = False

with st.sidebar:
    st.markdown("---")
    st.markdown("**短线评分因子**：")
    st.markdown("""
    - 短期趋势 (MA5>MA10, 价在MA5上)  
    - 突破强度 (涨幅>3%, 接近5日高点, 金叉)  
    - 量能异动 (量比>1.5, 倍量, 连续放量)  
    - 风险控制 (RSI 50-70, 5日涨幅5-15%, 波动率2-6%)
    """)
