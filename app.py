import streamlit as st
import akshare as ak
import pandas as pd
import time
import yfinance as yf  # 引入新援 Yahoo Finance

# ==========================================
# 🔧 数据源适配层 (核心修改)
# ==========================================

def get_history_data(code, market):
    """
    混合数据获取引擎：
    - A股：使用 Akshare
    - 港/美股：使用 yfinance
    返回标准化的 DataFrame: ['日期', '收盘', '最高', '最低']
    """
    df = pd.DataFrame()
    
    try:
        # --- 分支 1: A股 (保持原样) ---
        if market == "A股 (沪深)":
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
            # Akshare 返回列名已经是中文，无需映射
            
        # --- 分支 2: 港股 (使用 yfinance) ---
        elif market == "港股":
            # Akshare返回的代码通常是 5位 (00700)，Yahoo需要 4位+后缀 (0700.HK)
            # 处理逻辑：去掉前导0，补齐为4位，加 .HK
            # 例: 00700 -> 0700.HK, 09988 -> 9988.HK
            yf_code = f"{int(code):04d}.HK"
            
            # 下载数据 (auto_adjust=True 表示自动复权)
            data = yf.download(yf_code, start="2024-01-01", progress=False, auto_adjust=True)
            
            if not data.empty:
                # 格式统一化
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low']]
                df.columns = ['日期', '收盘', '最高', '最低']

        # --- 分支 3: 美股 (使用 yfinance) ---
        elif market == "美股":
            # 美股代码通常通用，无需转换 (如 AAPL, TSLA)
            yf_code = code
            
            data = yf.download(yf_code, start="2024-01-01", progress=False, auto_adjust=True)
            
            if not data.empty:
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low']]
                df.columns = ['日期', '收盘', '最高', '最低']
                
    except Exception as e:
        # 默默失败，不影响主流程，只是这只股票会被跳过
        pass
        
    return df

# ==========================================
# 🧠 量化算法区 (通用)
# ==========================================

def calculate_indicators_and_check(code, market, strategies):
    """
    通用技术分析函数
    """
    # 1. 调用混合引擎获取数据
    df = get_history_data(code, market)
    
    if df.empty or len(df) < 30:
        return False
        
    # 2. 计算指标 (Pandas 实现，通用)
    close = df['收盘']
    high = df['最高']
    low = df['最低']
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # KDJ
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    
    # 布林带
    mid = close.rolling(window=20).mean()
    std = close.rolling(window=20).std()
    upper = mid + 2 * std

    # 3. 信号判断 (取最后两日)
    try:
        # 注意：yfinance返回的数据索引有时是Timestamp，iloc定位最稳妥
        curr_dif, prev_dif = dif.iloc[-1], dif.iloc[-2]
        curr_dea, prev_dea = dea.iloc[-1], dea.iloc[-2]
        curr_k, prev_k = k.iloc[-1], k.iloc[-2]
        curr_d, prev_d = d.iloc[-1], d.iloc[-2]
        curr_rsi = rsi.iloc[-1]
        curr_close = close.iloc[-1]
        curr_upper = upper.iloc[-1]
        
        results = []
        if 'macd_gold' in strategies:
            results.append((curr_dif > curr_dea) and (prev_dif < prev_dea))
        if 'rsi_oversold' in strategies:
            results.append(curr_rsi < 30)
        if 'kdj_gold' in strategies:
            results.append((curr_k > curr_d) and (prev_k < prev_d))
        if 'boll_breakup' in strategies:
            results.append(curr_close > curr_upper)
            
        return all(results)
    except:
        return False

# ==========================================
# 🖥️ 页面 UI
# ==========================================

st.set_page_config(page_title="全球量化选股", page_icon="🌍", layout="wide")
st.markdown("<style>.stProgress > div > div > div > div { background-color: #f63366; }</style>", unsafe_allow_html=True)

st.title("🌍 全球量化选股 (混合引擎版)")
st.caption("A股数据源: Akshare | 港美股数据源: Yahoo Finance")
st.markdown("---")

with st.sidebar:
    st.header("1️⃣ 市场选择")
    market = st.selectbox("目标市场", ("A股 (沪深)", "港股", "美股"))
    
    # 动态滑块
    if market == "A股 (沪深)":
        limit = (-20.0, 20.0)
        default = (3.0, 9.0)
    else:
        limit = (-50.0, 100.0) # 放开限制
        default = (5.0, 20.0)
        
    pct_range = st.slider("涨跌幅 (%)", limit[0], limit[1], default)
    
    st.header("2️⃣ 技术筛选 (全市场支持)")
    use_tech = st.checkbox("启用技术指标筛选", value=False)
    
    strategies = []
    if use_tech:
        c1, c2 = st.columns(2)
        with c1:
            if st.checkbox("MACD 金叉"): strategies.append('macd_gold')
            if st.checkbox("RSI 超卖 (<30)"): strategies.append('rsi_oversold')
        with c2:
            if st.checkbox("KDJ 金叉"): strategies.append('kdj_gold')
            if st.checkbox("突破布林上轨"): strategies.append('boll_breakup')

    st.markdown("---")
    start_btn = st.button("🚀 开始扫描", type="primary", use_container_width=True)

if start_btn:
    with st.spinner(f"正在从 Akshare 拉取 {market} 实时榜单..."):
        df = pd.DataFrame()
        if market == "A股 (沪深)":
            df = ak.stock_zh_a_spot_em()
            df = df[~df['名称'].str.contains('ST|退')]
        elif market == "港股":
            df = ak.stock_hk_spot_em()
        elif market == "美股":
            df = ak.stock_us_spot_em()

    if not df.empty:
        # 数据清洗
        for col in ['最新价', '涨跌幅']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 初筛
        mask = (df['涨跌幅'] >= pct_range[0]) & (df['涨跌幅'] <= pct_range[1])
        filtered_df = df[mask].copy()
        
        final_df = filtered_df
        
        # 技术分析
        if use_tech and strategies:
            st.info(f"正在使用 Yahoo Finance 对 {len(filtered_df)} 只股票进行技术分析，请耐心等待...")
            progress = st.progress(0)
            status = st.empty()
            
            valid_codes = []
            check_list = filtered_df.head(100)['代码'].tolist() # 限制最大数量
            
            for i, code in enumerate(check_list):
                status.text(f"正在分析: {code} ...")
                progress.progress((i + 1) / len(check_list))
                
                # 调用混合引擎
                if calculate_indicators_and_check(code, market, strategies):
                    valid_codes.append(code)
                
                # 稍微休眠，对 Yahoo 友好一点
                time.sleep(0.1)
                
            final_df = filtered_df[filtered_df['代码'].isin(valid_codes)]
            status.text("分析完成")
            progress.empty()

        # 展示
        st.success(f"最终选出 {len(final_df)} 只股票")
        cols = ['代码', '名称', '最新价', '涨跌幅', '成交额']
        show_cols = [c for c in cols if c in final_df.columns]
        
        st.dataframe(final_df[show_cols].sort_values('涨跌幅', ascending=False), use_container_width=True)
        
        st.subheader("📋 结果代码")
        st.code(",".join(final_df['代码'].tolist()))
        
    else:
        st.error("行情数据获取失败")
