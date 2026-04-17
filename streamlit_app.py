import streamlit as st
import pandas as pd
import akshare as ak
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import io
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
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'total_batches' not in st.session_state:
    st.session_state.total_batches = 0
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

# ==================== 带重试的数据获取函数 ====================
def fetch_with_retry(func, *args, **kwargs):
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
    try:
        df = fetch_with_retry(ak.stock_zh_a_spot_em)
        df = df[df['代码'].str.match(r'(60|00|30)')]
        return df['代码'].tolist(), df['名称'].tolist()
    except Exception as e:
        st.warning(f"获取股票列表失败: {e}")
        return [], []

# ==================== 数据获取 ====================
def fetch_stock_data(code, days=60):
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
    df['VOL_MA5'] = df['volume'].rolling(5, min_periods=1).mean()
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
    
    df['golden_cross'] = (df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1))
    df['golden_cross'] = df['golden_cross'].fillna(False)
    df['bullish_arrange'] = (df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20'])
    df['bullish_arrange'] = df['bullish_arrange'].fillna(False)
    
    consecutive = 0
    surge_days = []
    for vol, vol_ma5 in zip(df['volume'], df['VOL_MA5']):
        if vol > vol_ma5 * 1.2:
            consecutive += 1
        else:
            consecutive = 0
        surge_days.append(consecutive)
    df['consecutive_surge'] = surge_days
    
    # 量价战法指标
    df['前三最高量'] = df['volume'].shift(1).rolling(3).max()
    df['is_高量'] = df['volume'] > df['前三最高量']
    df['实体低点'] = df.apply(lambda x: min(x['open'], x['close']), axis=1)
    df['支撑位'] = df.apply(lambda x: x['实体低点'] if x['is_高量'] else None, axis=1)
    df['支撑位'] = df['支撑位'].ffill()
    df['实体高点'] = df.apply(lambda x: max(x['open'], x['close']), axis=1)
    df['压力位'] = df.apply(lambda x: x['实体高点'] if x['is_高量'] else None, axis=1)
    df['压力位'] = df['压力位'].ffill()
    df['连续放量'] = (df['volume'] > df['volume'].shift(1)) & (df['volume'].shift(1) > df['volume'].shift(2))
    df['上涨梯量'] = df['连续放量'] & (df['close'] > df['close'].shift(1))
    df['下跌梯量'] = df['连续放量'] & (df['close'] < df['close'].shift(1))
    df['is_下跌趋势'] = df['close'] < df['MA20']
    df['下跌中高量'] = df['is_下跌趋势'] & df['is_高量']
    df['下跌高量累计'] = df['下跌中高量'].rolling(20).sum()
    df['下影线'] = df['low'] - df['close']
    df['下影比例'] = -df['下影线'] / (df['high'] - df['low'] + 0.001)
    df['is_大长腿'] = df['下影比例'] > 0.5
    df['is_缩量'] = df['volume'] < df['VOL_MA5'] * 0.8
    df['缩量大长腿'] = df['is_大长腿'] & df['is_缩量']
    df['上影线'] = df['high'] - df['close']
    df['上影比例'] = df['上影线'] / (df['high'] - df['low'] + 0.001)
    df['is_上影线'] = df['上影比例'] > 0.5
    df['碰压力位'] = df['is_上影线'] & (df['high'] > df['压力位'].shift(1))
    df['is_平量'] = (df['volume_ratio'] > 0.9) & (df['volume_ratio'] < 1.1)
    df['is_反弹'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    df['缩量平量反弹'] = df['is_缩量'] & df['is_平量'] & df['is_反弹']
    
    def check_di_fenxing(df, idx):
        if idx < 2 or idx >= len(df) - 1:
            return False
        return (df.iloc[idx]['low'] < df.iloc[idx-1]['low']) and (df.iloc[idx]['low'] < df.iloc[idx+1]['low'])
    df['底分型'] = [check_di_fenxing(df, i) for i in range(len(df))]
    df['is_阳线'] = df['close'] > df['open']
    df['红三兵'] = df['is_阳线'] & df['is_阳线'].shift(1) & df['is_阳线'].shift(2)
    
    return df

# ==================== 风险排除 ====================
def apply_risk_exclusion(df):
    if df.empty or len(df) < 30:
        return True, []
    
    exclude_reasons = []
    latest = df.iloc[-1]
    
    # K线高位
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    if recent_high != recent_low:
        position_pct = (latest['close'] - recent_low) / (recent_high - recent_low)
        if position_pct > 0.75:
            exclude_reasons.append(f"K线高位({position_pct:.0%})")
    
    # 连续拉升
    recent_returns = df['close'].pct_change().tail(3)
    if not recent_returns.isna().any():
        if (recent_returns > 0.04).all():
            exclude_reasons.append("连续拉升3天")
    
    # 打板
    if latest['open'] > 0:
        pct_chg = (latest['close'] - latest['open']) / latest['open']
        if pct_chg > 0.095:
            exclude_reasons.append("打板中")
    
    # MACD绿峰
    if df['MACD'].iloc[-1] < 0 and df['MACD_hist'].iloc[-1] < 0:
        exclude_reasons.append("MACD绿峰")
    
    # KDJ超买
    if latest['K'] > 80 and latest['J'] > 100:
        exclude_reasons.append("KDJ超买")
    
    # 波动过小
    if latest['ATR'] / latest['close'] < 0.01:
        exclude_reasons.append("波动过小")
    
    return len(exclude_reasons) > 0, exclude_reasons

# ==================== 量价信号 ====================
def get_volume_signals(df):
    if df.empty or len(df) < 10:
        return [], []
    
    latest = df.iloc[-1]
    risk_signals = []
    opp_signals = []
    
    if latest.get('上涨梯量', False):
        risk_signals.append("上涨梯量")
    if latest.get('碰压力位', False):
        risk_signals.append("碰压力位")
    if latest.get('缩量平量反弹', False):
        risk_signals.append("缩量平量反弹")
    
    if latest.get('下跌高量累计', 0) >= 3:
        opp_signals.append("三次高量见底")
    if latest.get('底分型', False) and latest['close'] > latest.get('支撑位', 0):
        opp_signals.append("底分型")
    if latest.get('红三兵', False) and latest['close'] > latest.get('支撑位', 0):
        opp_signals.append("红三兵")
    if latest.get('下跌梯量', False):
        opp_signals.append("下跌梯量")
    if latest.get('golden_cross', False):
        opp_signals.append("金叉")
    if latest.get('bullish_arrange', False):
        opp_signals.append("多头排列")
    
    return risk_signals, opp_signals

# ==================== 评分 ====================
def calculate_score(df):
    if df.empty:
        return 0
    
    latest = df.iloc[-1]
    risk_signals, opp_signals = get_volume_signals(df)
    score = 50
    
    if latest.get('bullish_arrange', False):
        score += 15
    if latest.get('golden_cross', False):
        score += 15
    if 30 <= latest.get('RSI', 50) <= 50:
        score += 10
    if latest.get('volume_ratio', 1) > 1.5:
        score += 10
    if latest.get('consecutive_surge', 0) >= 3:
        score += 10
    if latest.get('下跌高量累计', 0) >= 3:
        score += 15
    if latest.get('底分型', False):
        score += 15
    if latest.get('红三兵', False):
        score += 15
    if latest.get('下跌梯量', False):
        score += 10
    
    if latest.get('上涨梯量', False):
        score -= 15
    if latest.get('碰压力位', False):
        score -= 10
    
    return min(max(score, 0), 100)

# ==================== 处理单只股票 ====================
def process_single_stock(code, days=60):
    df = fetch_stock_data(code, days)
    if df is None or df.empty or len(df) < 30:
        return None
    
    df = calculate_single_stock_indicators(df)
    is_excluded, exclude_reasons = apply_risk_exclusion(df)
    if is_excluded:
        return None
    
    score = calculate_score(df)
    if score < 50:
        return None
    
    risk_signals, opp_signals = get_volume_signals(df)
    latest = df.iloc[-1]
    
    if len(risk_signals) > 0:
        action = "观望"
    elif len(opp_signals) >= 2:
        action = "买入"
    elif len(opp_signals) == 1:
        action = "关注"
    else:
        action = "持有"
    
    return {
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
        '机会信号': ';'.join(opp_signals) if opp_signals else '无',
        '风险信号': ';'.join(risk_signals) if risk_signals else '无',
        '操作建议': action,
        '止损参考': round(latest['stop_loss'], 2),
        '评分': score
    }

# ==================== 分批处理 ====================
def process_batch(codes, batch_id, total_batches, days=60, progress_callback=None):
    batch_results = []
    for idx, code in enumerate(codes):
        if progress_callback:
            progress_callback(batch_id, idx, len(codes))
        result = process_single_stock(code, days)
        if result:
            batch_results.append(result)
        if idx % 10 == 0:
            gc.collect()
    return batch_results

def scan_all_stocks(codes, days=60, batch_size=BATCH_SIZE, progress_callback=None):
    total = len(codes)
    num_batches = (total + batch_size - 1) // batch_size
    all_results = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total)
        batch_codes = codes[start_idx:end_idx]
        
        if progress_callback:
            progress_callback(batch_idx, num_batches, start_idx, end_idx, total)
        
        batch_results = process_batch(batch_codes, batch_idx, num_batches, days)
        all_results.extend(batch_results)
        gc.collect()
    
    return all_results

# ==================== 大盘趋势 ====================
def get_market_trend():
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
st.markdown("**风险排除法 + 量价关系分析** | 分批处理 | 全市场扫描")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 参数设置")
    days = st.slider("历史数据天数", 30, 90, 60, 10)
    min_score = st.slider("最低评分阈值", 30, 80, 50, 5)
    use_market_filter = st.checkbox("大盘过滤", True)
    
    st.markdown("---")
    st.markdown("### 📊 量价战法规则")
    st.markdown("""
    **风险信号:**
    - ⚠️ 上涨梯量
    - ⚠️ 上影线碰压力位
    - ⚠️ 缩量平量反弹
    
    **机会信号:**
    - ✅ 三次高量见底
    - ✅ 底分型
    - ✅ 红三兵
    - ✅ 下跌梯量
    - ✅ 金叉/多头排列
    """)
    
    st.markdown("---")
    st.markdown("### 🚫 风险排除")
    st.markdown("""
    - K线高位 >75%
    - 连续拉升3天
    - 打板中
    - MACD绿峰
    - KDJ超买区
    - 波动过小
    """)
    
    # 开始扫描按钮
    if st.button("🚀 开始扫描", type="primary", use_container_width=True):
        if st.session_state.scanning:
            st.warning("扫描正在进行中，请稍候...")
        else:
            st.session_state.scan_started = True
            st.session_state.scanning = True
            st.session_state.all_results = []
            st.session_state.scan_complete = False
            st.rerun()

# 主区域
if st.session_state.scan_started and not st.session_state.scan_complete:
    # 大盘过滤
    if use_market_filter and not get_market_trend():
        st.warning("⚠️ 大盘趋势偏弱，建议谨慎操作")
        st.session_state.scan_started = False
        st.session_state.scanning = False
        st.stop()
    
    # 获取股票列表
    with st.spinner("获取全市场股票列表..."):
        all_codes, all_names = get_all_stock_codes()
        if not all_codes:
            st.error("获取股票列表失败")
            st.session_state.scan_started = False
            st.session_state.scanning = False
            st.stop()
        
        total_stocks = len(all_codes)
        total_batches = (total_stocks + BATCH_SIZE - 1) // BATCH_SIZE
        st.success(f"✅ 获取到 {total_stocks} 只股票，分 {total_batches} 批处理")
    
    # 进度显示
    progress_bar = st.progress(0)
    batch_status = st.empty()
    stock_status = st.empty()
    results_area = st.empty()
    
    all_results = []
    
    def update_progress(batch_idx, total_batches, start_idx, end_idx, total):
        progress_bar.progress((batch_idx + 1) / total_batches)
        batch_status.text(f"📊 批次 {batch_idx + 1}/{total_batches} | 股票 {start_idx + 1}-{end_idx}/{total}")
    
    def update_stock_progress(batch_id, stock_idx, total_stocks_in_batch):
        stock_status.text(f"⏳ 处理中: {stock_idx + 1}/{total_stocks_in_batch}")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_stocks)
        batch_codes = all_codes[start_idx:end_idx]
        
        update_progress(batch_idx, total_batches, start_idx, end_idx, total_stocks)
        
        batch_results = []
        for stock_idx, code in enumerate(batch_codes):
            update_stock_progress(batch_idx, stock_idx, len(batch_codes))
            result = process_single_stock(code, days)
            if result:
                batch_results.append(result)
            if stock_idx % 10 == 0:
                gc.collect()
        
        if batch_results:
            all_results.extend(batch_results)
            temp_df = pd.DataFrame(all_results).sort_values('评分', ascending=False)
            results_area.dataframe(temp_df, use_container_width=True)
        
        gc.collect()
    
    progress_bar.empty()
    batch_status.empty()
    stock_status.empty()
    
    st.session_state.all_results = all_results
    st.session_state.scan_complete = True
    st.session_state.scan_started = False
    st.session_state.scanning = False
    st.rerun()

# 显示结果
if st.session_state.scan_complete:
    results = st.session_state.all_results
    
    if not results:
        st.info(f"📭 通过风险排除后，没有股票达到评分阈值 {min_score}")
    else:
        result_df = pd.DataFrame(results).sort_values('评分', ascending=False)
        
        st.subheader(f"🏆 选股结果（共 {len(result_df)} 只）")
        st.dataframe(result_df, use_container_width=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("入选股票数", len(result_df))
        with col2:
            st.metric("最高评分", result_df['评分'].max())
        with col3:
            st.metric("平均评分", round(result_df['评分'].mean(), 1))
        with col4:
            buy_count = len(result_df[result_df['操作建议'] == '买入'])
            st.metric("买入信号", buy_count)
        with col5:
            watch_count = len(result_df[result_df['操作建议'] == '关注'])
            st.metric("关注信号", watch_count)
        
        # 评分分布图
        if len(result_df) > 0:
            fig = go.Figure(data=[go.Bar(x=result_df['代码'][:20], y=result_df['评分'][:20])])
            fig.update_layout(height=400, title="Top 20 股票评分")
            st.plotly_chart(fig, use_container_width=True)
        
        # 下载按钮
        csv_final = result_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载选股结果 (CSV)",
            data=csv_final,
            file_name=f"选股结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    if st.button("🔄 重新扫描"):
        st.session_state.scan_complete = False
        st.session_state.all_results = []
        st.rerun()

# 历史结果展示（如果有）
with st.expander("📂 历史结果查看"):
    st.info("点击「开始扫描」运行新扫描，结果会自动保存")
    st.markdown("""
    **使用说明:**
    1. 点击侧边栏的「开始扫描」按钮
    2. 系统会自动获取全市场股票列表
    3. 分批处理（每批200只），实时显示进度
    4. 完成后自动显示结果，可下载CSV
    """)
    
