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
DEFAULT_MAX_STOCKS = 300      # 默认扫描300只
DEFAULT_DAYS = 30             # 历史数据天数
DEFAULT_WORKERS = 4           # 线程数

# ==================== 股票列表获取（支持多种排序） ====================
@st.cache_data(ttl=3600)
def get_stock_list_by_metric(sort_by="成交额", max_stocks=300):
    """
    获取股票列表，支持按成交额或涨跌幅排序
    sort_by: "成交额" 或 "涨跌幅"
    """
    try:
        # 获取A股实时行情
        df = ak.stock_zh_a_spot_em()
        # 筛选沪深A股 + 创业板（60/00/30开头）
        df = df[df['代码'].str.match(r'(60|00|30)')]
        
        if sort_by == "成交额":
            # 按成交额降序排序（成交额最大的在前）
            df = df.sort_values('成交额', ascending=False)
            st.info(f"📊 按成交额排序，选取前 {max_stocks} 只活跃股票")
        else:  # 涨跌幅
            # 按涨跌幅降序排序（涨幅最大的在前）
            df = df.sort_values('涨跌幅', ascending=False)
            st.info(f"📈 按涨跌幅排序，选取前 {max_stocks} 只强势股票")
        
        # 取前N只
        codes = df['代码'].tolist()[:max_stocks]
        names = df['名称'].tolist()[:max_stocks]
        
        # 同时保存一些信息供展示
        extra_info = {
            '成交额': df['成交额'].tolist()[:max_stocks] if '成交额' in df.columns else [],
            '涨跌幅': df['涨跌幅'].tolist()[:max_stocks] if '涨跌幅' in df.columns else []
        }
        
        return codes, names, extra_info
        
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], [], {}

# ==================== 数据获取 ====================
def fetch_stock_data(code, days=30):
    """获取单只股票历史数据"""
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
        return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
    except:
        return None

def get_data_batch(codes, days=30, max_workers=4):
    """并行获取数据"""
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

# ==================== 技术指标计算 ====================
def calculate_indicators(df):
    """计算技术指标"""
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
    
    # ATR
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
    return min(score, 80)

def filter_stocks(all_data, min_score=50):
    """筛选评分股票"""
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
                'MA20': round(latest['MA20'], 2),
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
    """判断大盘趋势"""
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
st.markdown("基于趋势、动量、成交量复合评分模型 | 支持成交额/涨跌幅排序")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 股票排序方式选择
    sort_method = st.radio(
        "股票排序方式",
        ["成交额排序（活跃股票）", "涨跌幅排序（强势股票）"],
        help="成交额排序：选择市场最活跃的股票；涨跌幅排序：选择当日涨幅最大的股票"
    )
    
    max_stocks = st.slider("扫描股票数量", 50, 800, 300, 50,
                           help="300只约占用400MB内存，500只约650MB")
    days = st.slider("历史数据天数", 20, 60, 30, 5)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    workers = st.slider("并行线程数", 2, 8, 4, 1)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    if st.button("🚀 开始扫描", type="primary"):
        st.session_state['run'] = True
    
    st.markdown("---")
    st.markdown("**排序说明**")
    if sort_method == "成交额排序（活跃股票）":
        st.markdown("✅ 选择成交额最大的股票，流动性好，适合短线")
    else:
        st.markdown("✅ 选择涨跌幅最大的股票，捕捉强势股，波动较大")

if 'run' in st.session_state and st.session_state['run']:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state['run'] = False
        st.stop()
    
    # 获取股票列表（根据排序方式）
    with st.spinner("获取股票列表..."):
        sort_key = "成交额" if "成交额" in sort_method else "涨跌幅"
        codes, names, extra = get_stock_list_by_metric(sort_key, max_stocks)
        
        if not codes:
            st.error("获取股票列表失败")
            st.stop()
        
        st.success(f"✅ 已按{sort_key}排序，选取前 {len(codes)} 只股票")
        
        # 显示前10只股票信息
        with st.expander(f"📋 前10只{sort_key}最大股票"):
            preview_df = pd.DataFrame({
                '代码': codes[:10],
                '名称': names[:10],
                sort_key: extra.get(sort_key, ['-']*10)[:10]
            })
            st.dataframe(preview_df, use_container_width=True)
    
    # 获取数据
    with st.spinner(f"正在获取 {len(codes)} 只股票数据（{days}天历史）..."):
        all_data = get_data_batch(codes, days, workers)
        if all_data.empty:
            st.error("数据获取失败")
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
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 评分区间统计
            bins = [0, 40, 50, 60, 70, 100]
            labels = ['<40', '40-50', '50-60', '60-70', '≥70']
            result_df['评分区间'] = pd.cut(result_df['评分'], bins=bins, labels=labels)
            score_dist = result_df['评分区间'].value_counts().sort_index()
            fig2 = go.Figure(data=[go.Bar(x=score_dist.index, y=score_dist.values)])
            fig2.update_layout(height=400, title="评分分布")
            st.plotly_chart(fig2, use_container_width=True)
        
        # 个股详情（可选）
        st.subheader("📊 个股详情")
        selected = st.selectbox("选择股票查看详情", result_df['代码'].tolist())
        stock_row = result_df[result_df['代码'] == selected].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("收盘价", f"{stock_row['收盘']:.2f}")
            st.metric("RSI", f"{stock_row['RSI']:.1f}")
        with col2:
            st.metric("MA5", f"{stock_row['MA5']:.2f}")
            st.metric("MA10", f"{stock_row['MA10']:.2f}")
        with col3:
            st.metric("MA20", f"{stock_row['MA20']:.2f}")
            st.metric("量比", f"{stock_row['量比']:.2f}")
    
    st.session_state['run'] = False

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### 股票排序方式说明
    
    | 排序方式 | 说明 | 适用场景 |
    |---------|------|----------|
    | **成交额排序** | 选择市场成交额最大的股票，流动性好 | 短线交易、稳健策略 |
    | **涨跌幅排序** | 选择当日涨幅最大的股票，捕捉强势股 | 追涨策略、激进交易 |
    
    ### 内存安全建议
    
    | 扫描数量 | 预计内存 | 是否安全 |
    |---------|---------|---------|
    | 200只 | ~250MB | ✅ 安全 |
    | 300只 | ~400MB | ✅ 安全（默认） |
    | 500只 | ~650MB | ⚠️ 接近上限 |
    | 800只 | ~900MB | ❌ 可能超限 |
    
    ### 评分模型
    - 多头排列：25分
    - 金叉信号：15分
    - RSI健康区：10分
    - 量比放大：10分
    - 成交量放大：10分
    - 温和波动率：10分
    """)
