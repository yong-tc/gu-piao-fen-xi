import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 内存安全限制（Streamlit Cloud 免费版约 1GB）
DEFAULT_MAX_STOCKS = 300      # 默认扫描300只（约占用300-400MB）
DEFAULT_DAYS = 30             # 历史数据天数（30天足够短线）
DEFAULT_WORKERS = 4           # 线程数（不宜过高）

# ==================== 数据获取 ====================
@st.cache_data(ttl=86400)
def get_all_stock_codes():
    """获取股票列表（优先本地缓存）"""
    try:
        df = pd.read_csv('stock_list.csv', encoding='utf-8')
        if '代码' in df.columns:
            return df['代码'].tolist(), df['名称'].tolist()
    except:
        pass
    
    try:
        stock_df = ak.stock_zh_a_spot_em()
        stock_df = stock_df[stock_df['代码'].str.match(r'(60|00|30)')]
        return stock_df['代码'].tolist(), stock_df['名称'].tolist()
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

def fetch_stock_data(code, days=30):
    """获取单只股票数据（精简版，只保留必要字段）"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={
            '日期': 'date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'volume'
        }, inplace=True)
        df['code'] = code
        # 只保留必要列，减少内存
        return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
    except:
        return None

def get_data_batch(codes, days=30, max_workers=4):
    """分批获取数据（避免一次性加载过多）"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data, code, days): code for code in codes}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

# ==================== 指标计算（纯pandas，向量化） ====================
def calculate_indicators(df):
    """计算技术指标（优化版，减少中间变量）"""
    if df.empty:
        return df
    
    # 移动平均线
    df['MA5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['MA10'] = df.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['MA20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    
    # RSI
    def calc_rsi(group):
        delta = group['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))
        return group
    df = df.groupby('code', group_keys=False).apply(calc_rsi)
    
    # 成交量均线
    df['VOL_MA5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    # 金叉和多头排列
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df.groupby('code')['MA5'].shift(1) <= df.groupby('code')['MA10'].shift(1))
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    
    # 量比
    df['volume_ratio'] = df.groupby('code')['volume'].transform(lambda x: x / x.rolling(5, min_periods=1).mean())
    
    # ATR（简化）
    def calc_atr(group):
        tr = pd.concat([
            group['high'] - group['low'],
            (group['high'] - group['close'].shift()).abs(),
            (group['low'] - group['close'].shift()).abs()
        ], axis=1).max(axis=1)
        group['ATR'] = tr.rolling(14, min_periods=1).mean()
        group['stop_loss'] = group['close'] - 2 * group['ATR']
        return group
    df = df.groupby('code', group_keys=False).apply(calc_atr)
    
    return df

def score_stock(group):
    """对单只股票评分"""
    latest = group.iloc[-1]
    score = 0
    if latest['bullish_arrange']:
        score += 25
    if latest['golden_cross']:
        score += 15
    if 30 <= latest['RSI'] <= 50:
        score += 10
    if latest['volume_ratio'] > 1.5:
        score += 10
    if latest['volume'] > latest['VOL_MA5'] * 1.2:
        score += 10
    if 0.02 <= latest['ATR'] / latest['close'] <= 0.05:
        score += 10
    return min(score, 80)  # 最高80分（简化版）

def filter_stocks(all_data, min_score=50):
    """筛选评分股票（内存优化版）"""
    if all_data.empty:
        return pd.DataFrame()
    
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
                '收盘': round(latest['close'], 2),
                'MA5': round(latest['MA5'], 2),
                'MA10': round(latest['MA10'], 2),
                'RSI': round(latest['RSI'], 1),
                '量比': round(latest['volume_ratio'], 2),
                '止损': round(latest['stop_loss'], 2),
                '评分': score
            })
    
    del all_data
    gc.collect()
    
    return pd.DataFrame(results).sort_values('评分', ascending=False)

# ==================== 大盘趋势 ====================
def get_market_trend():
    """简单判断大盘趋势"""
    try:
        df = ak.stock_zh_a_hist(symbol="000001", period="daily", 
                                start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                                end_date=datetime.now().strftime("%Y%m%d"))
        if df.empty:
            return True
        latest = df.iloc[-1]
        ma20 = df['收盘'].rolling(20).mean().iloc[-1]
        return latest['收盘'] > ma20
    except:
        return True

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="A股实时选股", layout="wide")
st.title("📈 A股实时选股系统 - 短线交易参考")

# 内存使用提示
st.info("💡 内存安全模式：默认扫描300只股票，约占用300-400MB内存（Streamlit限制1GB）")

with st.sidebar:
    st.header("⚙️ 内存安全参数")
    
    max_stocks = st.slider("扫描股票数量（内存安全建议 ≤ 500）", 50, 1000, DEFAULT_MAX_STOCKS, 50,
                           help="股票数量越多，内存占用越大。300只约400MB，500只约600MB")
    days = st.slider("历史数据天数（越少越快）", 20, 60, DEFAULT_DAYS, 5,
                     help="30天足够短线参考，减少天数可降低内存")
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    workers = st.slider("并行线程数", 2, 8, DEFAULT_WORKERS, 1)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    if st.button("🚀 开始扫描", type="primary"):
        st.session_state['run'] = True
    
    st.markdown("---")
    st.markdown("**内存优化措施**")
    st.markdown("- 限制扫描数量（默认300）")
    st.markdown("- 减少历史天数（默认30）")
    st.markdown("- 只保留必要列")
    st.markdown("- 主动垃圾回收")

if 'run' in st.session_state and st.session_state['run']:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎")
        st.session_state['run'] = False
        st.stop()
    
    # 获取股票列表
    with st.spinner("获取股票列表..."):
        codes, names = get_all_stock_codes()
        if not codes:
            st.error("获取失败")
            st.stop()
        
        # 限制数量
        if len(codes) > max_stocks:
            codes = codes[:max_stocks]
        st.info(f"✅ 扫描 {len(codes)} 只股票（内存安全）")
    
    # 获取数据
    with st.spinner(f"获取数据（{days}天，{workers}线程）..."):
        all_data = get_data_batch(codes, days, workers)
        if all_data.empty:
            st.error("数据获取失败")
            st.stop()
        st.success(f"✅ 获取 {len(all_data['code'].unique())} 只")
        gc.collect()
    
    # 计算评分
    with st.spinner("计算评分..."):
        result_df = filter_stocks(all_data, min_score)
        del all_data
        gc.collect()
    
    # 显示结果
    if result_df.empty:
        st.info(f"无股票达到评分阈值 {min_score}")
    else:
        st.subheader(f"🏆 评分 ≥ {min_score} 的股票")
        st.dataframe(result_df, use_container_width=True)
        
        # 评分分布
        fig = go.Figure(data=[go.Bar(x=result_df['代码'], y=result_df['评分'])])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.session_state['run'] = False

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 内存安全模式说明
    
    Streamlit Cloud 免费版限制 **1GB 内存**，本程序已优化：
    
    | 扫描数量 | 预计内存 | 是否安全 |
    |---------|---------|---------|
    | 200只 | ~250MB | ✅ 安全 |
    | 300只 | ~400MB | ✅ 安全（默认） |
    | 500只 | ~650MB | ⚠️ 接近上限 |
    | 800只 | ~900MB | ❌ 可能超限 |
    | 1000只 | ~1.1GB | ❌ 会超限 |
    
    **建议**：
    - 首次使用：扫描 **200只** 测试
    - 稳定后：可尝试 **300-400只**
    - 如需扫描更多，请使用拆分架构方案
    """)
