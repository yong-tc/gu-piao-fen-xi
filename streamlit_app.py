import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BATCH_SIZE = 200
DEFAULT_DAYS = 60
DEFAULT_WORKERS = 4
MAX_RETRIES = 3
RETRY_DELAY = 2

# ==================== 初始化 session_state ====================
if 'scan_started' not in st.session_state:
    st.session_state.scan_started = False
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False

# ==================== 带重试的数据获取函数 ====================
def fetch_with_retry(func, *args, **kwargs):
    """带重试机制的函数调用"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                raise e
    return None

# ==================== 获取全市场股票列表 ====================
@st.cache_data(ttl=3600)
def get_all_stock_codes():
    """获取沪深A股+创业板全部股票代码"""
    try:
        df = fetch_with_retry(ak.stock_zh_a_spot_em)
        df = df[df['代码'].str.match(r'(60|00|30)')]
        codes = df['代码'].tolist()
        names = df['名称'].tolist()
        return codes, names
    except Exception as e:
        st.warning(f"获取股票列表失败: {e}")
        return [], []

# ==================== 数据获取 ====================
def fetch_stock_data(code, days=60):
    """获取单只股票历史数据"""
    try:
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = fetch_with_retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                              start_date=start, end_date=end, adjust="qfq")
        if df is None or df.empty:
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
    
    # ========== 量价战法指标 ==========
    
    # 1. 高量识别：当天成交量大于前三天最大值
    df['前三最高量'] = df['volume'].shift(1).rolling(3).max()
    df['is_高量'] = df['volume'] > df['前三最高量']
    
    # 2. 高量实体低点（支撑位）
    df['实体低点'] = df.apply(lambda x: min(x['open'], x['close']), axis=1)
    df['支撑位'] = df.apply(lambda x: x['实体低点'] if x['is_高量'] else None, axis=1)
    df['支撑位'] = df['支撑位'].ffill()
    
    # 3. 高量实体高点（压力位）
    df['实体高点'] = df.apply(lambda x: max(x['open'], x['close']), axis=1)
    df['压力位'] = df.apply(lambda x: x['实体高点'] if x['is_高量'] else None, axis=1)
    df['压力位'] = df['压力位'].ffill()
    
    # 4. 梯量识别
    df['连续放量'] = (df['volume'] > df['volume'].shift(1)) & (df['volume'].shift(1) > df['volume'].shift(2))
    df['上涨梯量'] = df['连续放量'] & (df['close'] > df['close'].shift(1))
    df['下跌梯量'] = df['连续放量'] & (df['close'] < df['close'].shift(1))
    
    # 5. 下跌趋势中的高量次数
    df['is_下跌趋势'] = df['close'] < df['MA20']
    df['下跌中高量'] = df['is_下跌趋势'] & df['is_高量']
    df['下跌高量累计'] = df['下跌中高量'].rolling(20).sum()
    
    # 6. 缩量大长腿
    df['下影线'] = df['low'] - df['close']
    df['下影比例'] = -df['下影线'] / (df['high'] - df['low'] + 0.001)
    df['is_大长腿'] = df['下影比例'] > 0.5
    df['is_缩量'] = df['volume'] < df['VOL_MA5'] * 0.8
    df['缩量大长腿'] = df['is_大长腿'] & df['is_缩量']
    
    # 7. 上影线碰压力位
    df['上影线'] = df['high'] - df['close']
    df['上影比例'] = df['上影线'] / (df['high'] - df['low'] + 0.001)
    df['is_上影线'] = df['上影比例'] > 0.5
    df['碰压力位'] = df['is_上影线'] & (df['high'] > df['压力位'].shift(1))
    
    # 8. 缩量平量反弹
    df['is_平量'] = (df['volume_ratio'] > 0.9) & (df['volume_ratio'] < 1.1)
    df['is_反弹'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    df['缩量平量反弹'] = df['is_缩量'] & df['is_平量'] & df['is_反弹']
    
    # 9. 底分型
    def check_di_fenxing(df, idx):
        if idx < 2 or idx >= len(df) - 1:
            return False
        mid_low = df.iloc[idx]['low']
        left_low = df.iloc[idx-1]['low']
        right_low = df.iloc[idx+1]['low']
        return (mid_low < left_low) and (mid_low < right_low)
    df['底分型'] = [check_di_fenxing(df, i) for i in range(len(df))]
    
    # 10. 红三兵
    df['is_阳线'] = df['close'] > df['open']
    df['红三兵'] = df['is_阳线'] & df['is_阳线'].shift(1) & df['is_阳线'].shift(2)
    
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
                'RSI', 'K', 'D', 'J', 'ATR', 'stop_loss', '支撑位', '压力位']:
        if col in df.columns:
            df[col] = df[col].fillna(df['close'] if col != 'stop_loss' else df['close'] * 0.95)
    
    return df

# ==================== 风险排除模块 ====================

def exclude_high_position(df, lookback=20, high_threshold=0.75):
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
    if len(df) < surge_days + 1:
        return False, ""
    recent_returns = df['close'].pct_change().tail(surge_days)
    if recent_returns.isna().any():
        return False, ""
    if (recent_returns > surge_threshold).all():
        return True, f"连续拉升{surge_days}天"
    return False, ""

def exclude_limit_up(df):
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
    if 'MACD' not in df.columns or len(df) < 5:
        return False, ""
    if df['MACD'].iloc[-1] < 0 and df['MACD_hist'].iloc[-1] < 0:
        return True, "MACD绿峰"
    return False, ""

def exclude_kdj_peak(df, lookback=20):
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
    if 'ATR' not in df.columns or len(df) < 20:
        return False, ""
    volatility = df['ATR'].iloc[-1] / df['close'].iloc[-1]
    if volatility < min_atr_pct:
        return True, "波动过小"
    return False, ""

# ==================== 量价战法信号 ====================

def get_volume_signal(df):
    """根据量价战法生成交易信号"""
    if df.empty or len(df) < 10:
        return "观望", [], []
    
    latest = df.iloc[-1]
    risk_signals = []
    opp_signals = []
    
    # 风险信号
    if latest.get('上涨梯量', False):
        risk_signals.append("上涨梯量=风险")
    
    if latest.get('碰压力位', False):
        risk_signals.append("上影线碰压力位=减仓")
    
    if latest.get('缩量平量反弹', False):
        risk_signals.append("缩量平量反弹=观望")
    
    if latest.get('缩量大长腿', False):
        risk_signals.append("高量后缩量大长腿=观望")
    
    # 机会信号
    if latest.get('下跌高量累计', 0) >= 3:
        opp_signals.append("下跌趋势三次高量=机会")
    
    if latest.get('底分型', False) and latest['close'] > latest.get('支撑位', 0):
        opp_signals.append("支撑线上方底分型=买入")
    
    if latest.get('红三兵', False) and latest['close'] > latest.get('支撑位', 0):
        opp_signals.append("支撑线上方红三兵=买入")
    
    if latest.get('下跌梯量', False):
        opp_signals.append("下跌梯量=机会")
    
    # 判断最终操作
    if len(risk_signals) > 0:
        return "观望/减仓", risk_signals, opp_signals
    elif len(opp_signals) > 0:
        return "关注/买入", risk_signals, opp_signals
    else:
        return "观望", risk_signals, opp_signals

# ==================== 综合评分 ====================

def apply_risk_exclusion(df):
    """应用所有风险排除规则"""
    if df.empty or len(df) < 30:
        return True, ["数据不足"], []
    
    exclude_reasons = []
    
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
    
    return len(exclude_reasons) > 0, exclude_reasons


def calculate_final_score(df):
    """计算最终得分（风险排除后 + 量价信号加分）"""
    if df.empty:
        return 0
    
    latest = df.iloc[-1]
    score = 50
    
    # 技术指标加分
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
    
    if latest.get('close', 0) > 0:
        atr_pct = latest.get('ATR', 0) / latest['close']
        if 0.02 <= atr_pct <= 0.05:
            score += 5
    
    # 量价战法加分
    action, risk_signals, opp_signals = get_volume_signal(df)
    
    if "买入" in str(opp_signals):
        score += 20
    elif "机会" in str(opp_signals):
        score += 10
    
    if len(risk_signals) > 0:
        score -= 15
    
    return min(max(score, 0), 100), action, risk_signals, opp_signals

# ==================== 分批处理 ====================

def process_batch(codes, batch_id, total_batches, days=60, min_score=50):
    """处理一批股票"""
    batch_results = []
    
    for code in codes:
        df = fetch_stock_data(code, days)
        if df is None or df.empty or len(df) < 30:
            continue
        
        df = calculate_single_stock_indicators(df)
        is_excluded, exclude_reasons = apply_risk_exclusion(df)
        
        if is_excluded:
            continue
        
        score, action, risk_signals, opp_signals = calculate_final_score(df)
        
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
                '支撑位': round(latest['支撑位'], 2),
                '压力位': round(latest['压力位'], 2),
                '量价信号': ', '.join(opp_signals) if opp_signals else ('风险: ' + ', '.join(risk_signals) if risk_signals else '无'),
                '操作建议': action,
                '止损参考': round(latest['stop_loss'], 2),
                '评分': score
            })
        
        del df
        gc.collect()
    
    return batch_results


def scan_all_stocks(codes, days=60, min_score=50, batch_size=BATCH_SIZE):
    """分批扫描全市场股票"""
    total = len(codes)
    num_batches = (total + batch_size - 1) // batch_size
    
    all_results = []
    batch_progress = st.progress(0)
    batch_status = st.empty()
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_codes = codes[start_idx:end_idx]
        
        batch_status.text(f"正在处理第 {batch_idx + 1}/{num_batches} 批 (股票 {start_idx + 1}-{end_idx}/{total})")
        
        batch_results = process_batch(batch_codes, batch_idx, num_batches, days, min_score)
        all_results.extend(batch_results)
        
        batch_progress.progress((batch_idx + 1) / num_batches)
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
        df = fetch_with_retry(ak.stock_zh_a_hist, symbol="000001", period="daily",
                              start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
                              end_date=datetime.now().strftime("%Y%m%d"))
        if df is None or df.empty:
            return True
        df['ma20'] = df['收盘'].rolling(20).mean()
        return df['收盘'].iloc[-1] > df['ma20'].iloc[-1]
    except:
        return True

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="A股量价战法选股系统", layout="wide")
st.title("🛡️ A股量价战法选股系统")
st.markdown("**风险排除法 + 量价关系战法**：高量识别 | 梯量分析 | 支撑压力 | 底分型 | 红三兵")

with st.sidebar:
    st.header("⚙️ 参数设置")
    days = st.slider("历史数据天数", 30, 90, 60, 10)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    st.markdown("---")
    st.markdown("### 📊 量价战法规则")
    st.markdown("""
    **风险信号:**
    - ⚠️ 上涨梯量=风险
    - ⚠️ 上影线碰压力位=减仓
    - ⚠️ 缩量平量反弹=观望
    - ⚠️ 高量后缩量大长腿=观望
    
    **机会信号:**
    - ✅ 下跌趋势三次高量=机会
    - ✅ 支撑线上方底分型=买入
    - ✅ 支撑线上方红三兵=买入
    - ✅ 下跌梯量=机会
    """)
    
    if st.button("🚀 开始全市场扫描", type="primary"):
        st.session_state.scan_started = True
        st.session_state.scan_complete = False
        st.session_state.scan_results = None
        st.rerun()

# 执行扫描
if st.session_state.scan_started and not st.session_state.scan_complete:
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state.scan_started = False
        st.stop()
    
    with st.spinner("获取全市场股票列表..."):
        all_codes, all_names = get_all_stock_codes()
        if not all_codes:
            st.error("获取股票列表失败")
            st.session_state.scan_started = False
            st.stop()
        st.success(f"✅ 获取到 {len(all_codes)} 只股票")
    
    result_df = scan_all_stocks(all_codes, days, min_score, BATCH_SIZE)
    st.session_state.scan_results = result_df
    st.session_state.scan_complete = True
    st.session_state.scan_started = False
    st.rerun()

# 显示结果
if st.session_state.scan_complete and st.session_state.scan_results is not None:
    result_df = st.session_state.scan_results
    
    if result_df.empty:
        st.info(f"📭 通过风险排除后，没有股票达到评分阈值 {min_score}")
    else:
        st.subheader(f"🏆 选股结果（共 {len(result_df)} 只）")
        st.dataframe(result_df, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("入选股票数", len(result_df))
        with col2:
            st.metric("最高评分", result_df['评分'].max())
        with col3:
            st.metric("平均评分", round(result_df['评分'].mean(), 1))
        with col4:
            buy_signals = len(result_df[result_df['操作建议'].str.contains('买入', na=False)])
            st.metric("买入信号", buy_signals)
        
        if len(result_df) > 0:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.button("重新扫描"):
        st.session_state.scan_complete = False
        st.session_state.scan_results = None
        st.rerun()

with st.expander("📖 使用说明"):
    st.markdown("""
    ### 量价战法核心逻辑
    
    本程序结合了**风险排除法**和**量价关系战法**：
    
    **风险排除（先过滤）:**
    - K线高位、连续拉升、打板、空头绿十字、MACD绿峰、KDJ超买、波动过小
    
    **量价战法（再评分）:**
    - 高量识别、梯量分析、支撑压力、底分型、红三兵
    
    **操作建议:**
    - 关注/买入：出现机会信号，可加入观察
    - 观望/减仓：出现风险信号，谨慎操作
    """)
