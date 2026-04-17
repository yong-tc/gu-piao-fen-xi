import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DEFAULT_MAX_STOCKS = 200      # 默认200只，内存安全
DEFAULT_DAYS = 30
DEFAULT_WORKERS = 4

# ==================== 股票列表获取 ====================
@st.cache_data(ttl=3600)
def get_stock_list_by_metric(sort_by="成交额", max_stocks=200):
    """获取股票列表，支持按成交额或涨跌幅排序"""
    try:
        df = ak.stock_zh_a_spot_em()
        # 筛选沪深A股 + 创业板（60/00/30开头）
        df = df[df['代码'].str.match(r'(60|00|30)')]
        
        if sort_by == "成交额":
            df = df.sort_values('成交额', ascending=False)
            st.info(f"📊 按成交额排序，选取前 {max_stocks} 只活跃股票")
        else:
            df = df.sort_values('涨跌幅', ascending=False)
            st.info(f"📈 按涨跌幅排序，选取前 {max_stocks} 只强势股票")
        
        codes = df['代码'].tolist()[:max_stocks]
        names = df['名称'].tolist()[:max_stocks]
        
        return codes, names
        
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

# ==================== 数据获取（确保列名正确） ====================
def fetch_stock_data(code, days=30):
    """获取单只股票历史数据，返回标准化的DataFrame"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return None
        
        # 统一列名（英文，小写）
        df.rename(columns={
            '日期': 'date', 
            '开盘': 'open', 
            '收盘': 'close',
            '最高': 'high', 
            '最低': 'low', 
            '成交量': 'volume'
        }, inplace=True)
        
        df['code'] = code
        df['date'] = pd.to_datetime(df['date'])
        
        # 只保留必要列，减少内存
        return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        return None

def get_data_batch(codes, days=30, max_workers=4):
    """并行获取数据"""
    results = []
    progress_bar = st.progress(0)
    total = len(codes)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data, code, days): idx for idx, code in enumerate(codes)}
        
        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
            progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ==================== 技术指标计算（修复版） ====================
def calculate_indicators(df):
    """计算技术指标，增加列存在性检查"""
    if df.empty:
        return df
    
    # 检查必要列是否存在
    required_cols = ['code', 'close', 'volume', 'high', 'low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"缺少必要列: {missing_cols}")
        return df
    
    # 移动平均线
    df['MA5'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['MA10'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    df['MA20'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    
    # 成交量均线
    df['VOL_MA5'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    # RSI
    def calc_rsi(group):
        if len(group) < 14:
            group['RSI'] = 50.0
            return group
        delta = group['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))
        return group
    
    df = df.groupby('code', group_keys=False).apply(calc_rsi)
    
    # 金叉判断
    df['MA5_shift'] = df.groupby('code')['MA5'].shift(1)
    df['MA10_shift'] = df.groupby('code')['MA10'].shift(1)
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5_shift'] <= df['MA10_shift'])
    
    # 多头排列
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    
    # 量比
    df['volume_ratio'] = df.groupby('code')['volume'].transform(
        lambda x: x / x.rolling(5, min_periods=1).mean()
    )
    
    # ATR
    def calc_atr(group):
        if len(group) < 14:
            group['ATR'] = group['close'] * 0.02
            group['stop_loss'] = group['close'] * 0.95
            return group
        high_low = group['high'] - group['low']
        high_close = (group['high'] - group['close'].shift()).abs()
        low_close = (group['low'] - group['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        group['ATR'] = tr.rolling(14, min_periods=1).mean()
        group['stop_loss'] = group['close'] - 2 * group['ATR']
        return group
    
    df = df.groupby('code', group_keys=False).apply(calc_atr)
    
    # 删除临时列
    df = df.drop(['MA5_shift', 'MA10_shift'], axis=1, errors='ignore')
    
    return df

def score_stock(group):
    """对单只股票评分（取最新数据）"""
    if group.empty:
        return 0
    
    latest = group.iloc[-1]
    score = 0
    
    # 多头排列
    if latest.get('bullish_arrange', False):
        score += 25
    
    # 金叉
    if latest.get('golden_cross', False):
        score += 15
    
    # RSI
    rsi = latest.get('RSI', 50)
    if pd.notna(rsi):
        if 30 <= rsi <= 50:
            score += 10
        elif 50 < rsi <= 70:
            score += 5
    
    # 量比
    volume_ratio = latest.get('volume_ratio', 1)
    if pd.notna(volume_ratio) and volume_ratio > 1.5:
        score += 10
    
    # 成交量放大
    volume = latest.get('volume', 0)
    vol_ma5 = latest.get('VOL_MA5', 1)
    if pd.notna(volume) and pd.notna(vol_ma5) and volume > vol_ma5 * 1.2:
        score += 10
    
    # ATR波动率
    atr = latest.get('ATR', 0)
    close = latest.get('close', 1)
    if pd.notna(atr) and pd.notna(close) and close > 0:
        if 0.02 <= atr / close <= 0.05:
            score += 10
    
    return min(score, 80)

def filter_stocks(all_data, min_score=50):
    """筛选评分股票"""
    if all_data.empty:
        return pd.DataFrame()
    
    # 计算指标
    all_data = calculate_indicators(all_data)
    
    results = []
    for code, group in all_data.groupby('code'):
        if len(group) < 20:
            continue
        
        group = group.sort_values('date')
        score = score_stock(group)
        
        if score >= min_score:
            latest = group.iloc[-1]
            results.append({
                '代码': code,
                '收盘': round(latest['close'], 2) if pd.notna(latest['close']) else 0,
                'MA5': round(latest['MA5'], 2) if pd.notna(latest['MA5']) else 0,
                'MA10': round(latest['MA10'], 2) if pd.notna(latest['MA10']) else 0,
                'MA20': round(latest['MA20'], 2) if pd.notna(latest['MA20']) else 0,
                'RSI': round(latest['RSI'], 1) if pd.notna(latest['RSI']) else 50,
                '量比': round(latest['volume_ratio'], 2) if pd.notna(latest['volume_ratio']) else 1,
                '止损': round(latest['stop_loss'], 2) if pd.notna(latest['stop_loss']) else 0,
                '评分': score
            })
    
    del all_data
    gc.collect()
    
    if results:
        return pd.DataFrame(results).sort_values('评分', ascending=False)
    return pd.DataFrame()

# ==================== 大盘趋势 ====================
def get_market_trend():
    """判断大盘趋势"""
    try:
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                                start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                                end_date=datetime.now().strftime("%Y%m%d"))
        if df.empty:
            return True
        df['ma20'] = df['收盘'].rolling(20).mean()
        latest_close = df['收盘'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]
        return latest_close > ma20
    except:
        return True

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="A股实时选股", layout="wide")
st.title("📈 A股实时选股系统 - 短线交易参考")
st.markdown("基于趋势、动量、成交量复合评分模型 | 支持成交额/涨跌幅排序")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    sort_method = st.radio(
        "股票排序方式",
        ["成交额排序（活跃股票）", "涨跌幅排序（强势股票）"],
        help="成交额排序：选择市场最活跃的股票；涨跌幅排序：选择当日涨幅最大的股票"
    )
    
    max_stocks = st.slider("扫描股票数量", 50, 500, 200, 50,
                           help="200只约占用300MB内存，500只约650MB")
    days = st.slider("历史数据天数", 20, 60, 30, 5)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    workers = st.slider("并行线程数", 2, 6, 4, 1)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    if st.button("🚀 开始扫描", type="primary"):
        st.session_state['run'] = True

if 'run' in st.session_state and st.session_state['run']:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state['run'] = False
        st.stop()
    
    # 获取股票列表
    with st.spinner("获取股票列表..."):
        sort_key = "成交额" if "成交额" in sort_method else "涨跌幅"
        codes, names = get_stock_list_by_metric(sort_key, max_stocks)
        
        if not codes:
            st.error("获取股票列表失败")
            st.stop()
        
        st.success(f"✅ 已按{sort_key}排序，选取前 {len(codes)} 只股票")
    
    # 获取数据
    with st.spinner(f"正在获取 {len(codes)} 只股票数据（{days}天历史）..."):
        all_data = get_data_batch(codes, days, workers)
        if all_data.empty:
            st.error("数据获取失败，请稍后重试")
            st.stop()
        st.success(f"✅ 成功获取 {len(all_data['code'].unique())} 只股票数据")
        gc.collect()
    
    # 计算评分
    with st.spinner("正在计算技术指标和评分..."):
        result_df = filter_stocks(all_data, min_score)
        del all_data
        gc.collect()
    
    # 显示结果
    if result_df.empty:
        st.info(f"📭 没有股票达到评分阈值 {min_score}，可降低阈值或调整参数")
    else:
        st.subheader(f"🏆 评分 ≥ {min_score} 的股票（共 {len(result_df)} 只）")
        st.dataframe(result_df, use_container_width=True)
        
        # 评分分布图
        if len(result_df) > 0:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)
    
    st.session_state['run'] = False

with st.expander("📖 使用说明"):
    st.markdown("""
    ### 股票排序方式
    - **成交额排序**：选择市场最活跃的股票，流动性好，适合短线
    - **涨跌幅排序**：选择当日涨幅最大的股票，捕捉强势股
    
    ### 内存安全建议
    - 200只股票约占用300MB内存 ✅ 安全
    - 300只股票约占用450MB内存 ✅ 安全
    - 500只股票约占用700MB内存 ⚠️ 接近上限
    
    ### 评分模型
    - 多头排列：25分
    - 金叉信号：15分
    - RSI健康区：10分
    - 量比放大：10分
    - 成交量放大：10分
    - 温和波动率：10分
    """)
