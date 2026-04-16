#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立扫描脚本，用于GitHub Actions定时运行，结果可发送到企业微信/钉钉
"""
import os
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import requests
import warnings
warnings.filterwarnings('ignore')

# -------------------- 以下函数与 main.py 中保持一致 --------------------
def calculate_indicators(df):
    if df.empty:
        return df
    import pandas_ta as ta
    df['MA5'] = ta.sma(df['close'], length=5)
    df['MA10'] = ta.sma(df['close'], length=10)
    df['MA20'] = ta.sma(df['close'], length=20)
    df['MA60'] = ta.sma(df['close'], length=60)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['RSI'] = ta.rsi(df['close'], length=14)
    df['VOL_MA5'] = ta.sma(df['volume'], length=5)
    df['golden_cross'] = ((df['MA5'] > df['MA10']) & (df['MA5'].shift(1) <= df['MA10'].shift(1)))
    df['bullish_arrange'] = ((df['MA5'] > df['MA10']) & (df['MA10'] > df['MA20']) & (df['MA20'] > df['MA60']))
    return df

def add_advanced_indicators(df):
    if df.empty:
        return df
    import pandas_ta as ta
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['stop_loss'] = df['close'] - 2 * df['ATR']
    return df

def score_stock(df):
    latest = df.iloc[-1]
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

def get_all_stock_codes():
    stock_df = ak.stock_zh_a_spot_em()
    stock_df = stock_df[stock_df['代码'].str.match(r'(60|00|30)')]
    return stock_df['代码'].tolist()

def fetch_stock_data_worker(code, days=120):
    try:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start_date, end_date=end_date, adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={'日期':'date','开盘':'open','收盘':'close',
                           '最高':'high','最低':'low','成交量':'volume'}, inplace=True)
        df['code'] = code
        return df
    except:
        return None

def run_scan(days=120, min_score=60, max_workers=8):
    codes = get_all_stock_codes()
    print(f"获取到 {len(codes)} 只股票，开始并行下载...")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_data_worker, code, days): code for code in codes}
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)
    if not results:
        return pd.DataFrame()
    all_data = pd.concat(results, ignore_index=True)
    print("数据下载完成，开始评分...")
    final = []
    for code, group in all_data.groupby('code'):
        if len(group) < 60:
            continue
        group = group.sort_values('date')
        group = calculate_indicators(group)
        group = add_advanced_indicators(group)
        score = score_stock(group)
        latest = group.iloc[-1]
        final.append({
            'code': code,
            'close': latest['close'],
            'score': score,
            'RSI': latest['RSI'],
            'volume_ratio': latest['volume_ratio'],
            'ADX': latest['ADX'],
            'stop_loss': latest['stop_loss']
        })
    df_result = pd.DataFrame(final)
    df_result = df_result[df_result['score'] >= min_score].sort_values('score', ascending=False)
    return df_result

def send_wechat(content, webhook_url):
    """发送企业微信消息"""
    if not webhook_url:
        return
    data = {"msgtype": "markdown", "markdown": {"content": content}}
    try:
        requests.post(webhook_url, json=data, timeout=10)
    except Exception as e:
        print(f"发送失败: {e}")

def main():
    min_score = int(os.getenv('MIN_SCORE', '60'))
    webhook = os.getenv('WEBHOOK_URL', '')
    days = int(os.getenv('HISTORY_DAYS', '120'))
    
    result_df = run_scan(days=days, min_score=min_score, max_workers=8)
    if result_df.empty:
        msg = f"## {datetime.now().strftime('%Y-%m-%d')} 扫描结果\n暂无股票达到评分阈值({min_score})"
        print(msg)
        send_wechat(msg, webhook)
        return
    
    top_stocks = result_df.head(10)
    msg = f"## {datetime.now().strftime('%Y-%m-%d')} 选股结果\n"
    msg += f"**评分阈值:** {min_score}\n"
    msg += f"**入选数量:** {len(result_df)}\n\n"
    msg += "| 代码 | 收盘价 | 评分 | RSI | 量比 | ADX | 止损参考 |\n"
    msg += "|------|--------|------|-----|------|-----|----------|\n"
    for _, row in top_stocks.iterrows():
        msg += f"| {row['code']} | {row['close']:.2f} | {row['score']} | {row['RSI']:.1f} | {row['volume_ratio']:.2f} | {row['ADX']:.1f} | {row['stop_loss']:.2f} |\n"
    print(msg)
    send_wechat(msg, webhook)

if __name__ == "__main__":
    main()
