import streamlit as st
import akshare as ak
import pandas as pd
import time
import yfinance as yf
import numpy as np
import concurrent.futures
import re
import io

# ==========================================
# 🔧 1. 数据源处理 (新增文件解析)
# ==========================================

def clean_stock_codes(raw_text, market):
    """智能清洗函数"""
    if not raw_text: return []
    text = raw_text.replace("\n", ",").replace("\t", ",").replace(" ", ",").replace("，", ",")
    tokens = [x.strip() for x in text.split(",") if x.strip()]
    valid_codes = []
    return process_raw_tokens(tokens, market)
    
def process_file_upload(uploaded_file, market):
    """文件解析：支持 CSV / Excel"""
    codes = []
    try:
        df = pd.DataFrame()
        # 1. 读取文件
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                # 尝试 GBK (Moomoo 导出的 CSV 常见编码)
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='gbk')
        else:
            df = pd.read_excel(uploaded_file)
        
        # 2. 寻找代码列
        # Moomoo 导出通常叫 "代码", Yahoo 叫 "Symbol", 英文叫 "Code"
        target_col = None
        possible_names = ['代码', 'Code', 'Symbol', '股票代码', 'symbol', 'code']
        
        for col in df.columns:
            if col.strip() in possible_names:
                target_col = col
                break
        
        # 如果没找到同名列，尝试找第一列看起来像代码的
        if target_col is None:
            target_col = df.columns[0] # 盲猜第一列
            
        # 3. 提取并转为字符串
        if target_col:
            raw_list = df[target_col].astype(str).tolist()
            return process_raw_tokens(raw_list, market)
            
    except Exception as e:
        st.error(f"文件解析失败: {e}")
    return codes

def process_raw_tokens(tokens, market):
    """统一的正则提取逻辑"""
    valid_codes = []
    for token in tokens:
        # 去除前缀后缀
        clean_token = token.upper().replace("SH.", "").replace("SZ.", "").replace("HK.", "").replace("US.", "")
        clean_token = clean_token.replace(".SH", "").replace(".SZ", "").replace(".HK", "").replace(".US", "")
        
        if market == "A股 (沪深)":
            match = re.search(r'\d{6}', clean_token)
            if match: valid_codes.append(match.group())
        elif market == "港股":
            match = re.search(r'\d{4,5}', clean_token)
            if match: valid_codes.append(match.group())
        elif market == "美股":
            # 排除纯数字
            if clean_token.isalpha() and len(clean_token) <= 5:
                valid_codes.append(clean_token)
    
    return list(dict.fromkeys(valid_codes))

# === 核心：数据缓存 (TTL设为12小时) ===
# 即使你切出去2小时再回来，只要不重启服务器，之前下载过的数据都会秒读
@st.cache_data(ttl=43200, show_spinner=False)
def get_history_data_cached(code, market):
    """
    带缓存的数据获取函数。
    """
    df = pd.DataFrame()
    try:
        if market == "A股 (沪深)":
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
        elif market == "港股":
            code_str = str(code).zfill(4)
            if not code_str.endswith(".HK"): code_str = f"{code_str}.HK"
            data = yf.download(code_str, start="2024-01-01", progress=False, auto_adjust=True)
            if not data.empty:
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low', 'Volume']]
                df.columns = ['日期', '收盘', '最高', '最低', '成交量']
        elif market == "美股":
            data = yf.download(code, start="2024-01-01", progress=False, auto_adjust=True)
            if not data.empty:
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low', 'Volume']]
                df.columns = ['日期', '收盘', '最高', '最低', '成交量']
    except: pass
    return df

# ==========================================
# 🧠 2. 指标计算 & 策略
# ==========================================

def calculate_indicators(df):
    if df.empty: return df
    df = df.sort_values(by='日期', ascending=True).reset_index(drop=True)
    
    close = df['收盘']
    volume = df['成交量']
    
    # 均线
    df['ma5'] = close.rolling(5).mean()
    df['ma10'] = close.rolling(10).mean()
    df['ma20'] = close.rolling(20).mean()
    df['ma60'] = close.rolling(60).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd_bar'] = (df['dif'] - df['dea']) * 2

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # BOLL
    df['boll_mid'] = close.rolling(20).mean()
    df['boll_std'] = close.rolling(20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
    df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']

    # OBV
    df['obv'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    # 量
    df['vol_ma20'] = volume.rolling(20).mean()
    df['vol_ratio'] = volume / df['vol_ma20']

    return df

# --- 策略函数 ---
def check_macd_bar_divergence(df, window=30):
    if len(df) < window + 5: return False
    recent = df.iloc[-window:]
    if recent['最低'].iloc[-1] > recent['最低'].min() * 1.01: return False 
    if recent['macd_bar'].iloc[-1] > 0: return False
    curr_bar_min = recent['macd_bar'].iloc[-5:].min()
    prev_bars = recent['macd_bar'].iloc[:-10]
    if len(prev_bars[prev_bars < 0]) == 0: return False
    return curr_bar_min > prev_bars.min()

def check_ma_alignment(df):
    c = df.iloc[-1]
    return (c['ma5'] > c['ma10']) and (c['ma10'] > c['ma20']) and (c['ma20'] > c['ma60'])

def check_vcp_pattern(df):
    if len(df) < 60: return False
    w1 = df['boll_width'].iloc[-20:].mean()
    w2 = df['boll_width'].iloc[-40:-20].mean()
    return (w1 < w2 * 0.9) and (df['成交量'].iloc[-1] < df['vol_ma20'].iloc[-1])

def check_boll_squeeze_breakout(df):
    if len(df) < 22: return False
    curr = df.iloc[-1]
    if not (curr['收盘'] > curr['boll_upper']): return False
    past_width = df['boll_width'].iloc[-10:-1].mean()
    return (curr['boll_width'] > past_width * 1.1) and (curr['成交量'] > curr['vol_ma20'] * 1.5)

def check_obv_trend(df):
    if len(df) < 20: return False
    curr = df.iloc[-1]
    return (curr['obv'] > curr['obv_ma20']) and (curr['obv'] > df['obv'].iloc[-5])

# --- 调度器 ---
def check_technical_signals(code, market, strategies, lookback_days):
    # 使用带缓存的函数
    df = get_history_data_cached(code, market)
    
    if df.empty or len(df) < 60: return (False, None)
    
    df = calculate_indicators(df)
    
    for i in range(lookback_days):
        end_idx = -1 - i
        if end_idx == -1: current_slice = df
        else: current_slice = df.iloc[:end_idx+1]
        
        if len(current_slice) < 60: continue
        
        daily_res = []
        try:
            if 'macd_bar_div' in strategies: daily_res.append(check_macd_bar_divergence(current_slice))
            if 'rsi_oversold' in strategies: daily_res.append(current_slice.iloc[-1]['rsi'] < 30)
            if 'ma_alignment' in strategies: daily_res.append(check_ma_alignment(current_slice))
            if 'vcp_squeeze' in strategies: daily_res.append(check_vcp_pattern(current_slice))
            if 'boll_breakout' in strategies: daily_res.append(check_boll_squeeze_breakout(current_slice))
            if 'macd_gold' in strategies:
                c = current_slice.iloc[-1]; p = current_slice.iloc[-2]
                daily_res.append((c['dif'] > c['dea']) and (p['dif'] < p['dea']))
            if 'obv_trend' in strategies: daily_res.append(check_obv_trend(current_slice))

            if all(daily_res): return (True, current_slice.iloc[-1])
        except: continue
        
    return (False, None)

# ==========================================
# 🖥️ 4. UI (文件上传版)
# ==========================================

st.set_page_config(page_title="Stock Analyzer", page_icon="🦅", layout="wide")
st.markdown("<style>.stProgress > div > div > div > div { background-color: #f63366; }</style>", unsafe_allow_html=True)

st.title("🦅 Stock Analyzer")

# 初始化 Session State
if 'scan_results' not in st.session_state: st.session_state['scan_results'] = None
if 'scan_market' not in st.session_state: st.session_state['scan_market'] = ""
if 'scan_time' not in st.session_state: st.session_state['scan_time'] = ""

tab_scan, tab_help = st.tabs(["🚀 策略扫描", "📖 筛选标准与指南"])

# ===================== Tab 1: 扫描 =====================
with tab_scan:
    # 🌟 核心：使用 st.form 锁住所有交互，防止误触刷新
    with st.form("scanner_form"):
        st.caption("⚙️ 支持直接上传 Moomoo 导出的 Excel/CSV 文件。")
        
        col_input, col_settings = st.columns([1, 1])
        
        with col_input:
            st.subheader("1. 股票池导入")
            market = st.selectbox("市场选择", ("A股 (沪深)", "港股", "美股"))
            
            # === 新增：文件上传控件 ===
            uploaded_file = st.file_uploader("📂 上传 Moomoo 导出文件 (Excel/CSV)", type=['xlsx', 'csv'])
            
            raw_codes = st.text_area("📋 或直接粘贴代码", height=100, 
                placeholder="US.NVDA\n00700\n600519",
                help="如果不想上传文件，也可以手动粘贴。")

        with col_settings:
            st.subheader("2. 策略引擎")
            lookback_days = st.slider("信号回溯 (天)", 1, 5, 3)
            
            strategies = []
            with st.expander("🅰️ 左侧抄底 (Reversal)", expanded=True):
                if st.checkbox("MACD 柱状体底背离"): strategies.append('macd_bar_div')
                if st.checkbox("RSI 超卖 (<30)"): strategies.append('rsi_oversold')
                
            with st.expander("🅱️ 右侧追涨 (Trend)", expanded=True):
                if st.checkbox("均线多头 (MA5>10>20>60)"): strategies.append('ma_alignment')
                if st.checkbox("VCP 波动收缩"): strategies.append('vcp_squeeze')
                if st.checkbox("布林收口真突破"): strategies.append('boll_breakout')
                if st.checkbox("MACD 金叉"): strategies.append('macd_gold')
                
            with st.expander("📊 辅助确认", expanded=True):
                if st.checkbox("OBV 能量潮向上 🔥"): strategies.append('obv_trend')

        st.markdown("---")
        # 🌟 唯一的触发按钮
        submit_btn = st.form_submit_button("🚀 开始分析", type="primary", use_container_width=True)

    # 逻辑处理
    if submit_btn:
        code_list = []
        
        # 1. 处理上传文件
        if uploaded_file is not None:
            file_codes = process_file_upload(uploaded_file, market)
            if file_codes:
                code_list.extend(file_codes)
                st.toast(f"从文件中提取到 {len(file_codes)} 个代码")
        
        # 2. 处理粘贴文本
        if raw_codes.strip():
            text_codes = clean_stock_codes(raw_codes, market)
            code_list.extend(text_codes)
        
        # 去重
        code_list = list(dict.fromkeys(code_list))

        if not code_list:
            st.error("未提取到有效代码！请上传文件或粘贴文本。")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            if market != "A股 (沪深)" and len(code_list) > 300:
                st.warning(f"⚠️ 正在扫描 {len(code_list)} 只 {market} 股票，请耐心等待。")

            status_text.info(f"⏳ 正在分析 {len(code_list)} 只标的...")
            
            def process_task(args):
                c, m, s, d = args
                is_hit, snapshot = check_technical_signals(str(c), m, s, d)
                return (c, is_hit, snapshot)
            # -----------------------------------------------------------

            start_time = time.time()
            task_args = [(c, market, strategies, lookback_days) for c in code_list]
            
            valid_data = []
            # 动态调整并发
            max_workers = 10 if market == "A股 (沪深)" else 5 

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_task, arg): arg for arg in task_args}
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    c, is_hit, snapshot = future.result()
                    if is_hit:
                        valid_data.append({
                            "代码": c,
                            "最新价": round(snapshot['收盘'], 2),
                            "RSI值": round(snapshot['rsi'], 1),
                            "量比": round(snapshot['vol_ratio'], 1),
                            "布林带宽": round(snapshot['boll_width'], 3),
                            "OBV趋势": "⬆️" if snapshot['obv'] > snapshot['obv_ma20'] else "⬇️"
                        })
                    progress_bar.progress((i + 1) / len(code_list))
            
            end_time = time.time()
            progress_bar.empty()
            
            st.session_state['scan_results'] = valid_data
            st.session_state['scan_market'] = market
            st.session_state['scan_time'] = f"{end_time - start_time:.2f}s"
            status_text.empty()

    # 结果展示
    if st.session_state['scan_results'] is not None:
        data = st.session_state['scan_results']
        mkt = st.session_state['scan_market']
        
        if data:
            st.success(f"🎯 命中 {len(data)} 只 (耗时 {st.session_state['scan_time']})")
            df_res = pd.DataFrame(data)
            if mkt == "A股 (沪深)":
                df_res['代码'] = df_res['代码'].apply(lambda x: f"{int(x):06d}" if str(x).isdigit() else x)
            
            st.dataframe(df_res, use_container_width=True)
            st.code(",".join(df_res['代码'].astype(str).tolist()))
        else:
            st.warning("🍂 无股票命中。")

# ===================== Tab 2: 指南 =====================
with tab_help:
    st.markdown("""

    ## 📖 SOP 标准作业程序
    
    ### 1. 业务流程 (Workflow)
    * **Step 1 (PC端 Moomoo)**: 使用选股器选股 -> `Ctrl+A` 全选 -> 导出列表。
    * **Step 2 (本工具)**: 上传导出的文件 -> 选择【左侧】或【右侧】策略 -> 运行筛选。
    * **Step 3**: 复制本工具筛选出的精选代码 -> 填入 Daily Stock Analysis -> 运行进一步的分析。
    * **Step 4**: 在飞书/Lark查看 AI 研报。
    
    ### 2. Moomoo 选股器参数 (SOP)
    
    #### 🅰️ 左侧交易 (找超跌)
    * **A股**: 市值>100亿 | 价格<20日线 | RSI<40
    * **美/港**: 市值>50亿/200亿 | 价格<20日线 | RSI<40
    * **本工具策略**: 
        * **稳健**: `MACD底背离` + `RSI超卖`
        * **激进**: `MACD底背离`

    #### 🅱️ 右侧交易 (找主升)
    * **A股**: 市值>50亿 | 价格>60日线 | 换手>3%
    * **美/港**: 市值>20亿/100亿 | 价格>60日线 | 成交额>1千万/3千万
    * **本工具策略**: 
        * **稳健**: `均线多头` + `VCP收缩` + `OBV向上`
        * **激进**: `布林真突破` + `MACD金叉`
    """)
