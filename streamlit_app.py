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
BATCH_SIZE = 200  # 每批处理200只股票
DEFAULT_DAYS = 60
DEFAULT_WORKERS = 4

# ==================== 获取全市场股票列表 ====================
@st.cache_data(ttl=3600)
def get_all_stock_codes():
    """获取沪深A股+创业板全部股票代码"""
    try:
        df = ak.stock_zh_a_spot_em()
        # 筛选沪深A股 + 创业板（60/00/30开头）
        df = df[df['代码'].str.match(r'(60|00|30)')]
        codes = df['代码'].tolist()
        names = df['名称'].tolist()
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

# ==================== 单只股票指标计算 ====================
def calculate_single_stock_indicators(df):
    """计算单只股票的所有技术指标"""
    if df.empty or len(df) < 30:
        return df
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # 确保数值类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 均线
    df['MA5'] = df['close'].rolling(5, min_periods=1).mean()
    df['MA10'] = df['close'].rolling(10, min_periods=1).mean()
    df['MA20'] = df['close'].rolling(20, min_periods=1).mean()
    df['MA60'] = df['close'].rolling(60, min_periods=1).mean()
    
    # 成交量均线
    df['VOL_MA5'] = df['volume'].rolling(5, min_periods=1).mean()
    df['VOL_MA10'] = df['volume'].rolling(10, min_periods=1).mean()
    
    # 量比
    df['volume_ratio'] = df['volume'] / df['VOL_MA5']
    df['volume_ratio'] = df['volume_ratio'].fillna(1)
    
    # MACD
    if len(df) >= 26:
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    else:
        df['MACD'] = 0
        df['MACD_signal'] = 0
        df['MACD_hist'] = 0
    
    # KDJ
    if len(df) >= 9:
        low_list = df['low'].rolling(9, min_periods=1).min()
        high_list = df['high'].rolling(9, min_periods=1).max()
        rsv = (df['close'] - low_list) / (high_list - low_list) * 100
        rsv = rsv.fillna(50)
        
        k_values = [50]
        d_values = [50]
        
        for i in range(1, len(rsv)):
            k = (2/3) * k_values[-1] + (1/3) * rsv.iloc[i]
            d = (2/3) * d_values[-1] + (1/3) * k
            k_values.append(k)
            d_values.append(d)
        
        df['K'] = k_values
        df['D'] = d_values
        df['J'] = 3 * df['K'] - 2 * df['D']
    else:
        df['K'] = 50
        df['D'] = 50
        df['J'] = 50
    
    # RSI
    if len(df) >= 14:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    else:
        df['RSI'] = 50
    
    # ATR
    if len(df) >= 14:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14, min_periods=1).mean()
        df['stop_loss'] = df['close'] - 2 * df['ATR']
    else:
        df['ATR'] = df['close'] * 0.02
        df['stop_loss'] = df['close'] * 0.95
    
    # 金叉
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1))
    df['golden_cross'] = df['golden_cross'].fillna(False)
    
    # 多头排列
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    df['bullish_arrange'] = df['bullish_arrange'].fillna(False)
    
    # 放量检测
    df['volume_surge_3d'] = df['volume'].rolling(3, min_periods=1).mean() / df['volume'].rolling(10, min_periods=1).mean()
    df['volume_surge_3d'] = df['volume_surge_3d'].fillna(1)
    
    # 连续放量天数
    consecutive = 0
    surge_days = []
    for vol, vol_ma5 in zip(df['volume'], df['VOL_MA5']):
        if vol > vol_ma5 * 1.2:
            consecutive += 1
        else:
            consecutive = 0
        surge_days.append(consecutive)
    df['consecutive_surge'] = surge_days
    
    # 填充NaN
    for col in ['MA5', 'MA10', 'MA20', 'MA60', 'VOL_MA5', 'VOL_MA10', 
                'RSI', 'K', 'D', 'J', 'ATR', 'stop_loss']:
        if col in df.columns:
            df[col] = df[col].fillna(df['close'] if col != 'stop_loss' else df['close'] * 0.95)
    
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
    if recent_returns.isna().any():
        return False, ""
    
    is_continuous_surge = (recent_returns > surge_threshold).all()
    
    if is_continuous_surge:
        return True, f"连续拉升{surge_days}天"
    return False, ""


def exclude_limit_up(df):
    """3. 排除打板中的股票"""
    if len(df) < 1:
        return False, ""
    
    latest = df.iloc[-1]
    if latest['open'] <= 0:
        return False, ""
    
    pct_chg = (latest['close'] - latest['open']) / latest['open']
    
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
    
    if latest_k > 80 and latest_j > 100:
        return True, "KDJ超买区"
    
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
        return True, "波动过小"
    return False, ""


def is_kdj_bottom_stage(df, lookback=20):
    """9. 判断是否KDJ底部区域（加分项）"""
    if 'K' not in df.columns or len(df) < lookback:
        return False
    
    latest_k = df['K'].iloc[-1]
    k_min = df['K'].tail(lookback).min()
    
    if latest_k < 30 and latest_k < k_min + 10:
        return True
    return False


def apply_risk_exclusion(df):
    """应用所有风险排除规则"""
    if df.empty or len(df) < 30:
        return True, ["数据不足"], []
    
    exclude_reasons = []
    bonus_reasons = []
    
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
    
    if is_macd_red_early_stage(df):
        bonus_reasons.append("MACD红峰初期")
    
    if is_kdj_bottom_stage(df):
        bonus_reasons.append("KDJ底部区域")
    
    if 'volume_ratio' in df.columns and df['volume_ratio'].iloc[-1] > 1.3:
        bonus_reasons.append(f"量比{df['volume_ratio'].iloc[-1]:.1f}")
    
    if 'consecutive_surge' in df.columns and df['consecutive_surge'].iloc[-1] >= 2:
        bonus_reasons.append(f"连续放量{df['consecutive_surge'].iloc[-1]}天")
    
    return len(exclude_reasons) > 0, exclude_reasons, bonus_reasons


def calculate_final_score(df):
    """计算最终得分"""
    if df.empty:
        return 0
    
    latest = df.iloc[-1]
    score = 50
    
    if latest.get('bullish_arrange', False):
        score += 15
    
    if latest.get('golden_cross', False):
        score += 15
    
    rsi = latest.get('RSI', 50)
    if 30 <= rsi <= 50:
        score += 10
    elif 50 < rsi <= 70:
        score += 5
    
    vol_ratio = latest.get('volume_ratio', 1)
    if vol_ratio > 1.5:
        score += 10
    elif vol_ratio > 1.2:
        score += 5
    
    surge_days = latest.get('consecutive_surge', 0)
    if surge_days >= 3:
        score += 10
    elif surge_days >= 2:
        score += 5
    
    if is_macd_red_early_stage(df):
        score += 15
    
    if is_kdj_bottom_stage(df):
        score += 10
    
    if latest.get('close', 0) > 0:
        atr_pct = latest.get('ATR', 0) / latest['close']
        if 0.02 <= atr_pct <= 0.05:
            score += 5
    
    return min(score, 100)


def process_batch(codes, batch_id, total_batches, days=60, min_score=50):
    """处理一批股票"""
    batch_results = []
    
    for code in codes:
        df = fetch_stock_data(code, days)
        if df is None or df.empty or len(df) < 30:
            continue
        
        df = calculate_single_stock_indicators(df)
        is_excluded, exclude_reasons, bonus_reasons = apply_risk_exclusion(df)
        
        if is_excluded:
            continue
        
        score = calculate_final_score(df)
        
        if score >= min_score:
            latest = df.iloc[-1]
            batch_results.append({
                '代码': code,
                '收盘': round(latest['close'], 2),
                'MA5': round(latest['MA5'], 2),
                'MA10': round(latest['MA10'], 2),
                'MA20': round(latest['MA20'], 2),
                'RSI': round(latest['RSI'], 1),
                '量比': round(latest['volume_ratio'], 2),
                '连续放量': int(latest['consecutive_surge']),
                'MACD状态': '红峰初期' if is_macd_red_early_stage(df) else ('绿峰' if exclude_macd_green_peak(df)[0] else '其他'),
                'KDJ状态': '底部' if is_kdj_bottom_stage(df) else ('高位' if exclude_kdj_peak(df)[0] else '正常'),
                '止损参考': round(latest['stop_loss'], 2),
                '评分': score,
                '加分项': ','.join(bonus_reasons) if bonus_reasons else ''
            })
        
        # 释放内存
        del df
        gc.collect()
    
    return batch_results


def scan_all_stocks(codes, days=60, min_score=50, batch_size=BATCH_SIZE):
    """分批扫描全市场股票"""
    total = len(codes)
    num_batches = (total + batch_size - 1) // batch_size
    
    all_results = []
    
    # 创建进度条
    batch_progress = st.progress(0)
    batch_status = st.empty()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_codes = codes[start_idx:end_idx]
        
        batch_status.text(f"正在处理第 {batch_idx + 1}/{num_batches} 批 (股票 {start_idx + 1}-{end_idx}/{total})")
        
        batch_results = process_batch(batch_codes, batch_idx, num_batches, days, min_score)
        all_results.extend(batch_results)
        
        # 更新进度
        batch_progress.progress((batch_idx + 1) / num_batches)
        
        # 每批结束后强制释放内存
        gc.collect()
    
    batch_status.empty()
    batch_progress.empty()
    
    if all_results:
        return pd.DataFrame(all_results).sort_values('评分', ascending=False)
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
st.info("📌 分批处理模式：每批200只股票，覆盖沪深A股+创业板全市场")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    days = st.slider("历史数据天数", 30, 90, 60, 10)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    st.markdown("---")
    st.markdown("### 📊 扫描模式")
    st.markdown(f"""
    - **分批数量**: 每批 {BATCH_SIZE} 只
    - **总股票数**: 约5000只
    - **总批次数**: 约{(5000 + BATCH_SIZE - 1) // BATCH_SIZE}批
    - **预计耗时**: 15-30分钟
    """)
    
    st.markdown("### 🚫 风险排除规则")
    st.markdown("""
    - ❌ K线高位（>75%位置）
    - ❌ 连续拉升3天以上
    - ❌ 打板中的股票
    - ❌ 空头绿十字K线
    - ❌ MACD绿峰
    - ❌ KDJ超买区/峰顶已过
    - ❌ 波动过小
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
    
    if st.button("🚀 开始全市场扫描", type="primary"):
        st.session_state['run'] = True

if 'run' in st.session_state and st.session_state['run']:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state['run'] = False
        st.stop()
    
    # 获取全市场股票列表
    with st.spinner("获取全市场股票列表（沪深A股+创业板）..."):
        all_codes, all_names = get_all_stock_codes()
        
        if not all_codes:
            st.error("获取股票列表失败")
            st.stop()
        
        st.success(f"✅ 获取到 {len(all_codes)} 只股票（沪深A股+创业板）")
        st.info(f"📊 将分 {(len(all_codes) + BATCH_SIZE - 1) // BATCH_SIZE} 批处理，每批 {BATCH_SIZE} 只，预计耗时 15-30 分钟")
    
    # 分批扫描
    result_df = scan_all_stocks(all_codes, days, min_score, BATCH_SIZE)
    
    # 显示结果
    if result_df.empty:
        st.info(f"📭 通过风险排除后，没有股票达到评分阈值 {min_score}")
    else:
        st.subheader(f"🏆 通过风险排除的股票（共 {len(result_df)} 只）")
        st.dataframe(result_df, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入选股票数", len(result_df))
        with col2:
            st.metric("最高评分", result_df['评分'].max())
        with col3:
            st.metric("平均评分", round(result_df['评分'].mean(), 1))
        with col4:
            red_early_count = len(result_df[result_df['MACD状态'] == '红峰初期'])
            st.metric("红峰初期股票", red_early_count)
        
        if len(result_df) > 0:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)
    
    st.session_state['run'] = False

with st.expander("📖 使用说明"):
    st.markdown("""
    ### 分批处理模式说明
    
    本程序采用**分批处理**策略覆盖全市场：
    
    - **每批处理**: 200只股票
    - **内存控制**: 每批处理完立即释放内存
    - **全市场覆盖**: 沪深A股+创业板（约5000只）
    - **预计耗时**: 15-30分钟（取决于网络和服务器）
    
    ### 风险排除法核心逻辑
    
    1. **K线高位排除**：股价处于近期高位（>75%分位）的排除
    2. **连续拉升排除**：连续3天涨幅>4%的排除
    3. **打板排除**：当日涨幅>9.5%的排除
    4. **空头绿十字排除**：阴线十字星排除
    5. **MACD绿峰排除**：MACD为负且柱状线为负的排除
    6. **KDJ峰值排除**：K>80且J>100，或从高位回落的排除
    7. **波动过小排除**：ATR/价格<1%的排除
    
    **加分项**：通过排除后，根据红峰初期、KDJ底部、放量等条件加分
    """)
