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
DEFAULT_MAX_STOCKS = 300
DEFAULT_DAYS = 60
DEFAULT_WORKERS = 4

# ==================== 股票列表获取 ====================
@st.cache_data(ttl=3600)
def get_stock_list_by_metric(sort_by="成交额", max_stocks=300):
    """获取股票列表，支持按成交额或涨跌幅排序"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[df['代码'].str.match(r'(60|00|30)')]
        
        if sort_by == "成交额":
            df = df.sort_values('成交额', ascending=False)
        else:
            df = df.sort_values('涨跌幅', ascending=False)
        
        codes = df['代码'].tolist()[:max_stocks]
        names = df['名称'].tolist()[:max_stocks]
        
        return codes, names
        
    except Exception as e:
        st.error(f"获取股票列表失败: {e}")
        return [], []

# ==================== 数据获取 ====================
def fetch_stock_data(code, days=60):
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
        df['date'] = pd.to_datetime(df['date'])
        
        return df[['code', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
    except Exception:
        return None

def get_data_batch(codes, days=60, max_workers=4):
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

# ==================== 技术指标计算 ====================
def calculate_indicators(df):
    """计算技术指标（MACD、KDJ、RSI、均线、ATR等）"""
    if df.empty:
        return df
    
    # 均线
    df['MA5'] = df.groupby('code')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['MA10'] = df.groupby('code')['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    df['MA20'] = df.groupby('code')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['MA60'] = df.groupby('code')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
    
    # 成交量均线
    df['VOL_MA5'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['VOL_MA10'] = df.groupby('code')['volume'].transform(lambda x: x.rolling(10, min_periods=1).mean())
    
    # 量比
    df['volume_ratio'] = df['volume'] / df['VOL_MA5']
    
    # ========== MACD指标 ==========
    def calc_macd(group):
        if len(group) < 26:
            group['MACD'] = 0
            group['MACD_signal'] = 0
            group['MACD_hist'] = 0
            return group
        exp1 = group['close'].ewm(span=12, adjust=False).mean()
        exp2 = group['close'].ewm(span=26, adjust=False).mean()
        group['MACD'] = exp1 - exp2
        group['MACD_signal'] = group['MACD'].ewm(span=9, adjust=False).mean()
        group['MACD_hist'] = group['MACD'] - group['MACD_signal']
        return group
    df = df.groupby('code', group_keys=False).apply(calc_macd)
    
    # ========== KDJ指标 ==========
    def calc_kdj(group):
        if len(group) < 9:
            group['K'] = 50
            group['D'] = 50
            group['J'] = 50
            return group
        
        low_list = group['low'].rolling(9, min_periods=1).min()
        high_list = group['high'].rolling(9, min_periods=1).max()
        rsv = (group['close'] - low_list) / (high_list - low_list) * 100
        
        group['K'] = rsv.ewm(com=2, adjust=False).mean()
        group['D'] = group['K'].ewm(com=2, adjust=False).mean()
        group['J'] = 3 * group['K'] - 2 * group['D']
        return group
    df = df.groupby('code', group_keys=False).apply(calc_kdj)
    
    # ========== RSI ==========
    def calc_rsi(group):
        if len(group) < 14:
            group['RSI'] = 50
            return group
        delta = group['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        group['RSI'] = 100 - (100 / (1 + rs))
        return group
    df = df.groupby('code', group_keys=False).apply(calc_rsi)
    
    # ========== ATR（波动率） ==========
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
    
    # 金叉判断
    df['MA5_shift'] = df.groupby('code')['MA5'].shift(1)
    df['MA10_shift'] = df.groupby('code')['MA10'].shift(1)
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5_shift'] <= df['MA10_shift'])
    
    # 多头排列
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    
    # 放量检测
    df['volume_surge_3d'] = df.groupby('code')['volume'].transform(
        lambda x: x.rolling(3, min_periods=1).mean() / x.rolling(10, min_periods=1).mean()
    )
    
    # 连续放量天数
    def calc_consecutive_surge(group):
        consecutive = 0
        surge_days = []
        for vol, vol_ma5 in zip(group['volume'], group['VOL_MA5']):
            if vol > vol_ma5 * 1.2:
                consecutive += 1
            else:
                consecutive = 0
            surge_days.append(consecutive)
        group['consecutive_surge'] = surge_days
        return group
    df = df.groupby('code', group_keys=False).apply(calc_consecutive_surge)
    
    # 删除临时列
    df = df.drop(['MA5_shift', 'MA10_shift'], axis=1, errors='ignore')
    
    return df

# ==================== 风险排除模块 ====================

def exclude_high_position(df, lookback=20, high_threshold=0.75):
    """1. 排除K线高位的股票"""
    if len(df) < lookback:
        return True, "数据不足"
    
    latest = df.iloc[-1]
    recent_high = df['high'].tail(lookback).max()
    recent_low = df['low'].tail(lookback).min()
    
    if recent_high == recent_low:
        return False, ""
    
    position_pct = (latest['close'] - recent_low) / (recent_high - recent_low)
    
    if position_pct > high_threshold:
        return True, f"K线高位({position_pct:.0%})"
    return False, ""


def exclude_continuous_surge(df, surge_days=3, surge_threshold=0.04):
    """2. 排除连续拉升中的股票"""
    if len(df) < surge_days + 1:
        return False, ""
    
    recent_returns = df['close'].pct_change().tail(surge_days)
    is_continuous_surge = (recent_returns > surge_threshold).all()
    
    if is_continuous_surge:
        return True, f"连续拉升{surge_days}天"
    return False, ""


def exclude_limit_up(df):
    """3. 排除打板中的股票"""
    if len(df) < 1:
        return False, ""
    
    latest = df.iloc[-1]
    pct_chg = (latest['close'] - latest['open']) / latest['open'] if latest['open'] > 0 else 0
    
    if pct_chg > 0.095:
        return True, "打板中"
    return False, ""


def exclude_bearish_doji(df):
    """4. 排除空头绿十字K线"""
    if len(df) < 2:
        return False, ""
    
    latest = df.iloc[-1]
    
    is_green = latest['close'] < latest['open']
    body = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    
    if total_range == 0:
        return False, ""
    
    is_doji = body < total_range * 0.3
    
    if is_green and is_doji:
        return True, "空头绿十字"
    return False, ""


def exclude_macd_green_peak(df):
    """5. 排除MACD全阶段绿峰"""
    if 'MACD' not in df.columns or len(df) < 5:
        return False, ""
    
    latest_macd = df['MACD'].iloc[-1]
    latest_hist = df['MACD_hist'].iloc[-1]
    
    if latest_macd < 0 and latest_hist < 0:
        return True, "MACD绿峰"
    return False, ""


def is_macd_red_early_stage(df, early_days=3):
    """6. 判断是否红峰初期（加分项）"""
    if 'MACD' not in df.columns or len(df) < early_days + 5:
        return False
    
    recent_macd = df['MACD'].tail(early_days + 3)
    was_negative = (recent_macd.iloc[:-early_days] < 0).any()
    is_now_positive = recent_macd.iloc[-1] > 0
    
    return was_negative and is_now_positive


def exclude_kdj_peak(df, lookback=20):
    """7. 排除KDJ峰顶过高的股票"""
    if 'K' not in df.columns or len(df) < lookback:
        return False, ""
    
    latest_k = df['K'].iloc[-1]
    latest_j = df['J'].iloc[-1]
    
    k_mean = df['K'].tail(lookback).mean()
    j_mean = df['J'].tail(lookback).mean()
    
    # KDJ超买区
    if latest_k > 80 and latest_j > 100:
        return True, "KDJ超买区"
    
    # 峰顶已过（从高位回落）
    if len(df) >= 3:
        if df['K'].iloc[-2] > 80 and df['K'].iloc[-1] < df['K'].iloc[-2]:
            return True, "KDJ峰顶已过"
    
    return False, ""


def exclude_low_volatility(df, min_atr_pct=0.01):
    """8. 排除波动过小的股票"""
    if 'ATR' not in df.columns or len(df) < 20:
        return False, ""
    
    atr = df['ATR'].iloc[-1]
    close = df['close'].iloc[-1]
    
    if close <= 0:
        return False, ""
    
    volatility = atr / close
    
    if volatility < min_atr_pct:
        return True, f"波动过小(ATR/价格={volatility:.2%})"
    return False, ""


def is_kdj_bottom_stage(df, lookback=20):
    """9. 判断是否KDJ底部区域（加分项）"""
    if 'K' not in df.columns or len(df) < lookback:
        return False
    
    latest_k = df['K'].iloc[-1]
    k_min = df['K'].tail(lookback).min()
    
    # KDJ底部区域（K值<30且接近近期低点）
    if latest_k < 30 and latest_k < k_min + 10:
        return True
    return False


def apply_risk_exclusion(df):
    """
    应用所有风险排除规则
    返回: (是否排除, 排除原因列表, 加分项列表)
    """
    if df.empty or len(df) < 30:
        return True, ["数据不足"], []
    
    exclude_reasons = []
    bonus_reasons = []
    
    # 排除规则
    is_excluded, reason = exclude_high_position(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_continuous_surge(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_limit_up(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_bearish_doji(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_macd_green_peak(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_kdj_peak(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    is_excluded, reason = exclude_low_volatility(df)
    if is_excluded:
        exclude_reasons.append(reason)
    
    # 加分项（不排除，只是加分）
    if is_macd_red_early_stage(df):
        bonus_reasons.append("MACD红峰初期")
    
    if is_kdj_bottom_stage(df):
        bonus_reasons.append("KDJ底部区域")
    
    # 量比放大
    if 'volume_ratio' in df.columns and df['volume_ratio'].iloc[-1] > 1.3:
        bonus_reasons.append(f"量比{df['volume_ratio'].iloc[-1]:.1f}")
    
    # 连续放量
    if 'consecutive_surge' in df.columns and df['consecutive_surge'].iloc[-1] >= 2:
        bonus_reasons.append(f"连续放量{df['consecutive_surge'].iloc[-1]}天")
    
    is_excluded_final = len(exclude_reasons) > 0
    return is_excluded_final, exclude_reasons, bonus_reasons


def calculate_final_score(df):
    """计算最终得分（排除法通过后的评分）"""
    if df.empty:
        return 0
    
    latest = df.iloc[-1]
    score = 50  # 基础分
    
    # 加分项
    if 'bullish_arrange' in df.columns and latest['bullish_arrange']:
        score += 15
    
    if 'golden_cross' in df.columns and latest['golden_cross']:
        score += 15
    
    if 'RSI' in df.columns:
        rsi = latest['RSI']
        if 30 <= rsi <= 50:
            score += 10
        elif 50 < rsi <= 70:
            score += 5
    
    if 'volume_ratio' in df.columns:
        vol_ratio = latest['volume_ratio']
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio > 1.2:
            score += 5
    
    if 'consecutive_surge' in df.columns:
        surge_days = latest['consecutive_surge']
        if surge_days >= 3:
            score += 10
        elif surge_days >= 2:
            score += 5
    
    # 红峰初期加分
    if is_macd_red_early_stage(df):
        score += 15
    
    # KDJ底部加分
    if is_kdj_bottom_stage(df):
        score += 10
    
    # ATR波动率加分
    if 'ATR' in df.columns and latest['close'] > 0:
        atr_pct = latest['ATR'] / latest['close']
        if 0.02 <= atr_pct <= 0.05:
            score += 5
    
    return min(score, 100)


def filter_stocks_with_exclusion(all_data, min_score=50):
    """风险排除版选股"""
    if all_data.empty:
        return pd.DataFrame()
    
    # 计算指标
    all_data = calculate_indicators(all_data)
    
    results = []
    for code, group in all_data.groupby('code'):
        if len(group) < 30:
            continue
        
        group = group.sort_values('date')
        
        # 应用风险排除
        is_excluded, exclude_reasons, bonus_reasons = apply_risk_exclusion(group)
        
        if is_excluded:
            continue  # 排除该股票
        
        # 计算得分
        score = calculate_final_score(group)
        
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
                '连续放量': int(latest['consecutive_surge']) if latest['consecutive_surge'] else 0,
                'MACD状态': '红峰初期' if is_macd_red_early_stage(group) else ('绿峰' if exclude_macd_green_peak(group)[0] else '其他'),
                'KDJ状态': '底部' if is_kdj_bottom_stage(group) else ('高位' if exclude_kdj_peak(group)[0] else '正常'),
                '止损参考': round(latest['stop_loss'], 2),
                '评分': score,
                '加分项': ','.join(bonus_reasons) if bonus_reasons else ''
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
st.set_page_config(page_title="A股风险排除选股系统", layout="wide")
st.title("🛡️ A股风险排除选股系统")
st.markdown("基于**风险排除法**：K线高位排除 | 连续拉升排除 | MACD绿峰排除 | KDJ峰值排除 | 波动过小排除")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    sort_method = st.radio(
        "股票排序方式",
        ["成交额排序（活跃股票）", "涨跌幅排序（强势股票）"]
    )
    
    max_stocks = st.slider("扫描股票数量", 50, 500, 200, 50)
    days = st.slider("历史数据天数", 30, 90, 60, 10)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    workers = st.slider("并行线程数", 2, 6, 4, 1)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    st.markdown("---")
    st.markdown("### 🚫 风险排除规则")
    st.markdown("""
    - ❌ K线高位（>75%位置）
    - ❌ 连续拉升3天以上
    - ❌ 打板中的股票
    - ❌ 空头绿十字K线
    - ❌ MACD绿峰
    - ❌ KDJ超买区/峰顶已过
    - ❌ 波动过小(ATR/价格<1%)
    """)
    
    st.markdown("### ✅ 加分项")
    st.markdown("""
    - ⭐ MACD红峰初期
    - ⭐ KDJ底部区域
    - ⭐ 量比放大
    - ⭐ 连续放量
    - ⭐ 多头排列
    - ⭐ 金叉信号
    """)
    
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
            st.error("数据获取失败")
            st.stop()
        st.success(f"✅ 成功获取 {len(all_data['code'].unique())} 只股票数据")
        gc.collect()
    
    # 计算评分并排除
    with st.spinner("正在应用风险排除规则并计算评分..."):
        result_df = filter_stocks_with_exclusion(all_data, min_score)
        del all_data
        gc.collect()
    
    # 显示结果
    if result_df.empty:
        st.info(f"📭 通过风险排除后，没有股票达到评分阈值 {min_score}")
    else:
        st.subheader(f"🏆 通过风险排除的股票（共 {len(result_df)} 只）")
        
        # 显示表格
        st.dataframe(result_df, use_container_width=True)
        
        # 统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入选股票数", len(result_df))
        with col2:
            st.metric("最高评分", result_df['评分'].max())
        with col3:
            st.metric("平均评分", round(result_df['评分'].mean(), 1))
        with col4:
            st.metric("红峰初期股票", len(result_df[result_df['MACD状态'] == '红峰初期']))
        
        # 评分分布图
        if len(result_df) > 0:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)

st.session_state['run'] = False

with st.expander("📖 使用说明"):
    st.markdown("""
    ### 风险排除法核心逻辑
    
    本程序基于你的**风险排除法**实现：
    
    1. **K线高位排除**：股价处于近期高位（>75%分位）的排除
    2. **连续拉升排除**：连续3天涨幅>4%的排除
    3. **打板排除**：当日涨幅>9.5%的排除
    4. **空头绿十字排除**：阴线十字星排除
    5. **MACD绿峰排除**：MACD为负且柱状线为负的排除
    6. **KDJ峰值排除**：K>80且J>100，或从高位回落的排除
    7. **波动过小排除**：ATR/价格<1%的排除
    
    **加分项**：通过排除后，根据红峰初期、KDJ底部、放量等条件加分
    """)
