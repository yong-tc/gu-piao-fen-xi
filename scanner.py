#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股量价战法选股系统 - 后台扫描脚本
用于 GitHub Actions 定时运行，完成后自动发送邮件
"""
import pandas as pd
import akshare as ak
import numpy as np
from datetime import datetime, timedelta
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置 ====================
BATCH_SIZE = 200
DAYS = 60
MIN_SCORE = 50
MAX_RETRIES = 3
RETRY_DELAY = 2

# 邮件配置（使用 GitHub Secrets 存储）
SMTP_SERVER = "smtp.126.com"
SMTP_PORT = 465
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECEIVER_EMAIL = "zhangyong_zhongyao@126.com"

# ==================== 邮件发送函数 ====================
def send_email(subject, body, attachment_path=None):
    """发送邮件"""
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("邮件配置缺失，跳过发送")
        return False
    
    try:
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        
        # 邮件正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 添加附件
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={os.path.basename(attachment_path)}'
                )
                msg.attach(part)
        
        # 发送邮件
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"邮件发送成功: {subject}")
        return True
        
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False

# ==================== 带重试的数据获取 ====================
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

# ==================== 获取股票列表 ====================
def get_all_stock_codes():
    try:
        df = fetch_with_retry(ak.stock_zh_a_spot_em)
        df = df[df['代码'].str.match(r'(60|00|30)')]
        return df['代码'].tolist()
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        return []

# ==================== 获取单只股票数据 ====================
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

# ==================== 指标计算 ====================
def calculate_indicators(df):
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
        return True
    
    latest = df.iloc[-1]
    
    # K线高位
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    if recent_high != recent_low:
        position_pct = (latest['close'] - recent_low) / (recent_high - recent_low)
        if position_pct > 0.75:
            return True
    
    # 连续拉升
    recent_returns = df['close'].pct_change().tail(3)
    if not recent_returns.isna().any():
        if (recent_returns > 0.04).all():
            return True
    
    # 打板
    if latest['open'] > 0:
        pct_chg = (latest['close'] - latest['open']) / latest['open']
        if pct_chg > 0.095:
            return True
    
    # MACD绿峰
    if df['MACD'].iloc[-1] < 0 and df['MACD_hist'].iloc[-1] < 0:
        return True
    
    # KDJ超买
    if latest['K'] > 80 and latest['J'] > 100:
        return True
    
    # 波动过小
    if latest['ATR'] / latest['close'] < 0.01:
        return True
    
    return False

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
    if latest.get('底分型', False):
        opp_signals.append("底分型")
    if latest.get('红三兵', False):
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
def process_stock(code, days=60):
    df = fetch_stock_data(code, days)
    if df is None or df.empty or len(df) < 30:
        return None
    
    df = calculate_indicators(df)
    
    if apply_risk_exclusion(df):
        return None
    
    score = calculate_score(df)
    if score < MIN_SCORE:
        return None
    
    latest = df.iloc[-1]
    risk_signals, opp_signals = get_volume_signals(df)
    
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

# ==================== 分批扫描 ====================
def scan_all_stocks(codes, days=60):
    all_results = []
    total = len(codes)
    
    for idx, code in enumerate(codes):
        if idx % 100 == 0:
            print(f"进度: {idx}/{total} ({idx/total*100:.1f}%)")
        
        result = process_stock(code, days)
        if result:
            all_results.append(result)
        
        if idx % 50 == 0:
            time.sleep(0.5)
    
    return all_results

# ==================== 生成邮件内容 ====================
def generate_email_body(df_result, total_stocks, scan_time):
    """生成邮件正文"""
    buy_count = len(df_result[df_result['操作建议'] == '买入'])
    watch_count = len(df_result[df_result['操作建议'] == '关注'])
    hold_count = len(df_result[df_result['操作建议'] == '持有'])
    
    # 买入信号股票列表
    buy_stocks = df_result[df_result['操作建议'] == '买入'].head(10)
    watch_stocks = df_result[df_result['操作建议'] == '关注'].head(10)
    
    body = f"""
========================================
A股量价战法选股系统 - 扫描报告
========================================

扫描时间: {scan_time}
扫描股票: {total_stocks} 只
入选股票: {len(df_result)} 只

📊 统计摘要:
- 买入信号: {buy_count} 只
- 关注信号: {watch_count} 只  
- 持有观望: {hold_count} 只
- 最高评分: {df_result['评分'].max()}
- 平均评分: {df_result['评分'].mean():.1f}

{'='*40}
🔴 买入信号股票 Top 10:
{'='*40}
"""
    
    if buy_count > 0:
        for _, row in buy_stocks.iterrows():
            body += f"\n{row['代码']} | 收盘:{row['收盘']} | RSI:{row['RSI']} | 量比:{row['量比']} | 评分:{row['评分']}"
            if row['机会信号'] != '无':
                body += f"\n  机会: {row['机会信号']}"
    else:
        body += "\n暂无买入信号股票"
    
    body += f"""

{'='*40}
🟡 关注信号股票 Top 10:
{'='*40}
"""
    
    if watch_count > 0:
        for _, row in watch_stocks.iterrows():
            body += f"\n{row['代码']} | 收盘:{row['收盘']} | RSI:{row['RSI']} | 量比:{row['量比']} | 评分:{row['评分']}"
            if row['机会信号'] != '无':
                body += f"\n  机会: {row['机会信号']}"
    else:
        body += "\n暂无关注信号股票"
    
    body += f"""

{'='*40}
📈 评分分布:
{'='*40}
- 90分以上: {len(df_result[df_result['评分'] >= 90])} 只
- 80-89分: {len(df_result[(df_result['评分'] >= 80) & (df_result['评分'] < 90)])} 只
- 70-79分: {len(df_result[(df_result['评分'] >= 70) & (df_result['评分'] < 80)])} 只
- 60-69分: {len(df_result[(df_result['评分'] >= 60) & (df_result['评分'] < 70)])} 只
- 50-59分: {len(df_result[(df_result['评分'] >= 50) & (df_result['评分'] < 60)])} 只

========================================
详细数据请查看附件 CSV 文件
系统每天自动运行，如有问题请及时反馈
========================================
"""
    
    return body

# ==================== 主函数 ====================
def main():
    scan_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"开始扫描 - {scan_time}")
    
    # 获取股票列表
    codes = get_all_stock_codes()
    if not codes:
        print("获取股票列表失败")
        send_email(
            subject=f"【错误】A股选股系统 - {datetime.now().strftime('%Y-%m-%d')}",
            body=f"扫描时间: {scan_time}\n\n错误: 获取股票列表失败\n\n请检查网络或数据源"
        )
        return
    
    total_stocks = len(codes)
    print(f"获取到 {total_stocks} 只股票")
    
    # 扫描
    results = scan_all_stocks(codes, DAYS)
    
    if not results:
        print("没有符合条件的股票")
        send_email(
            subject=f"【结果】A股选股系统 - {datetime.now().strftime('%Y-%m-%d')}",
            body=f"扫描时间: {scan_time}\n扫描股票: {total_stocks}\n\n没有符合条件的股票，请调整评分阈值"
        )
        return
    
    # 保存结果
    df_result = pd.DataFrame(results).sort_values('评分', ascending=False)
    
    os.makedirs('results', exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"results/{today}.csv"
    df_result.to_csv(filename, index=False, encoding='utf-8-sig')
    df_result.to_csv('results/latest.csv', index=False, encoding='utf-8-sig')
    
    print(f"扫描完成！共选出 {len(results)} 只股票")
    print(f"结果已保存到 {filename}")
    print(f"最高评分: {df_result['评分'].max()}")
    print(f"平均评分: {df_result['评分'].mean():.1f}")
    print(f"买入信号: {len(df_result[df_result['操作建议'] == '买入'])} 只")
    
    # 发送邮件
    body = generate_email_body(df_result, total_stocks, scan_time)
    send_email(
        subject=f"【选股结果】A股量价战法 - {today} (入选{len(results)}只)",
        body=body,
        attachment_path=filename
    )

if __name__ == "__main__":
    main()
