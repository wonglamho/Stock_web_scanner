import streamlit as st
import akshare as ak
import pandas as pd
import time
import yfinance as yf
import numpy as np
import concurrent.futures  # 引入并发库

# ==========================================
# 🔧 1. 数据源适配层 (保持不变)
# ==========================================

def get_history_data(code, market):
    """
    统一获取 A/港/美 股的历史K线数据
    """
    df = pd.DataFrame()
    try:
        if market == "A股 (沪深)":
            # A股使用 Akshare 接口，adjust="qfq" 代表前复权，消除分红影响
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date="20240101", adjust="qfq")
        elif market == "港股":
            # 港股需要拼接 .HK 后缀
            yf_code = f"{int(code):04d}.HK"
            data = yf.download(yf_code, start="2024-01-01", progress=False, auto_adjust=True)
            if not data.empty:
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low']]
                df.columns = ['日期', '收盘', '最高', '最低']
        elif market == "美股":
            # 美股直接使用代码
            yf_code = code
            data = yf.download(yf_code, start="2024-01-01", progress=False, auto_adjust=True)
            if not data.empty:
                df = data.reset_index()
                df = df[['Date', 'Close', 'High', 'Low']]
                df.columns = ['日期', '收盘', '最高', '最低']
    except Exception:
        pass
    return df

# ==========================================
# 🧠 2. 核心量化算法 (保持不变，仅增加注释)
# ==========================================

def calculate_indicators(df):
    """
    计算技术指标
    MACD (12, 26, 9)
    RSI (14)
    KDJ (9, 3, 3)
    BOLL (20, 2)
    """
    # 必须按日期升序排列，否则指标计算会反向
    df = df.sort_values(by='日期', ascending=True).reset_index(drop=True)
    close = df['收盘']
    low = df['最低']
    high = df['最高']
    
    # --- 1. MACD (异同移动平均线) ---
    # 参数: 快线=12, 慢线=26, 信号线=9
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['dif'] = ema12 - ema26
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['dif'] - df['dea']) * 2

    # --- 2. RSI (相对强弱指标) ---
    # 参数: 周期=14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- 3. KDJ (随机指标) ---
    # 参数: 周期=9, K平滑=3, D平滑=3
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    df['k'] = rsv.ewm(com=2, adjust=False).mean()
    df['d'] = df['k'].ewm(com=2, adjust=False).mean()
    
    # --- 4. BOLL (布林带) ---
    # 参数: 周期=20, 宽度=2倍标准差
    df['boll_mid'] = close.rolling(window=20).mean()
    df['boll_std'] = close.rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    
    return df

def check_technical_signals(code, market, strategies):
    """
    根据选定的策略检查股票
    """
    df = get_history_data(code, market)
    # 如果数据少于60天，无法准确计算 MACD 背离等长周期指标，直接跳过
    if df.empty or len(df) < 60: return False
    
    df = calculate_indicators(df)
    curr = df.iloc[-1] # 当日数据
    prev = df.iloc[-2] # 昨日数据
    
    results = []
    try:
        # MACD 金叉: 当日 DIF > DEA 且 昨日 DIF < DEA
        if 'macd_gold' in strategies:
            results.append((curr['dif'] > curr['dea']) and (prev['dif'] < prev['dea']))
        
        # RSI 超卖: RSI 数值小于 30，通常视为反弹信号
        if 'rsi_oversold' in strategies:
            results.append(curr['rsi'] < 30)
            
        # KDJ 金叉: K线上穿D线
        if 'kdj_gold' in strategies:
            results.append((curr['k'] > curr['d']) and (prev['k'] < prev['d']))
            
        # 布林带突破: 收盘价站上布林上轨，通常为强势单边行情的开始
        if 'boll_breakup' in strategies:
            results.append(curr['收盘'] > curr['boll_upper'])
            
        # MACD 底背离: 股价创近20日新低，但MACD的DIF值未创新低
        if 'macd_div' in strategies:
            window = 20
            is_price_low = curr['收盘'] <= df['收盘'].tail(window).min()
            is_dif_higher = curr['dif'] > df['dif'].tail(window).min()
            is_underwater = curr['dif'] < 0 # 必须在零轴下方
            results.append(is_price_low and is_dif_higher and is_underwater)
            
        return all(results)
    except:
        return False

# ==========================================
# 🖥️ 3. UI 交互层 (包含详细 Help 提示)
# ==========================================

st.set_page_config(page_title="全球量化选股 Turbo", page_icon="⚡", layout="wide")
st.markdown("<style>.stProgress > div > div > div > div { background-color: #f63366; }</style>", unsafe_allow_html=True)

st.title("⚡ 全球量化选股 (Turbo版)")
st.caption("A股/港股/美股 | 多线程并发 | 混合引擎")
st.markdown("---")

with st.sidebar:
    st.header("1️⃣ 市场与基础筛选")
    market = st.selectbox("目标市场", ("A股 (沪深)", "港股", "美股"))
    
    if market == "A股 (沪深)":
        limit = (-20.0, 20.0); default = (3.0, 9.0)
    else:
        limit = (-100.0, 100.0); default = (5.0, 20.0)
    pct_range = st.slider("涨跌幅 (%)", limit[0], limit[1], default)
    
    st.subheader("📊 进阶基本面 (若有数据)")
    
    turnover_min = st.number_input(
        "最小换手率 (%)", 
        value=0.0, 
        step=1.0,
        help="推荐设置：\n- 3%~7%: 交易活跃，人气正常\n- 7%~15%: 强势股特征\n- >15%: 极度活跃或主力出货风险"
    )
    
    amount_min = st.number_input(
        "最小成交额 (万元)", 
        value=0, 
        step=1000,
        help="过滤流动性指标：\n- 输入 10000 (1亿): 过滤掉大部分垃圾股和冷门股\n- 输入 50000 (5亿): 筛选机构和大资金关注的流动性充沛标的\n*注：该数值直接对应筛选单位，美股/港股建议根据实际体量调整"
    )
    
    vr_min = st.number_input(
        "最小量比", 
        value=0.0, 
        step=0.1,
        help="量比衡量相对成交量：\n- > 1.0: 放量，交易比平时活跃\n- > 2.0: 明显放量，可能有主力资金介入\n- 推荐设置 1.5 左右作为门槛"
    )
    
    pe_max = st.number_input(
        "最大市盈率 (PE)", 
        value=0, 
        step=10,
        help="估值指标：\n- < 30: 价值股/低估值区域\n- 30~60: 成长股常见区间\n- 输入 0 表示不限制"
    )

    st.markdown("---")
    
    st.header("2️⃣ 技术信号")
    use_tech = st.checkbox("启用技术指标筛选", value=False)
    
    strategies = []
    if use_tech:
        c1, c2 = st.columns(2)
        with c1:
            if st.checkbox("MACD 金叉", help="DIF 上穿 DEA (12,26,9)"): strategies.append('macd_gold')
            if st.checkbox("MACD 底背离 🔥", help="股价创新低但MACD指标未创新低，强力抄底信号"): strategies.append('macd_div')
            if st.checkbox("RSI 超卖 (<30)", help="RSI(14) 进入超卖区，存在反弹需求"): strategies.append('rsi_oversold')
        with c2:
            if st.checkbox("KDJ 金叉", help="K线 上穿 D线 (9,3,3)"): strategies.append('kdj_gold')
            if st.checkbox("突破布林上轨", help="收盘价站上布林带(20,2)上轨，强势特征"): strategies.append('boll_breakup')

    st.markdown("---")
    start_btn = st.button("🚀 开始极速扫描", type="primary", use_container_width=True)

# 封装多线程任务
def process_stock_task(args):
    code, mkt, strats = args
    if check_technical_signals(str(code), mkt, strats):
        return code
    return None

if start_btn:
    with st.spinner(f"正在拉取 {market} 实时数据..."):
        df = pd.DataFrame()
        if market == "A股 (沪深)":
            df = ak.stock_zh_a_spot_em()
            df = df[~df['名称'].str.contains('ST|退')]
        elif market == "港股":
            df = ak.stock_hk_spot_em()
        elif market == "美股":
            df = ak.stock_us_spot_em()
    
    if not df.empty:
        # === 核心修正 1: 智能列名映射与类型转换 ===
        exclude_cols = ['代码', 'code', 'symbol', '名称', 'name', 'cname']
        
        for col in df.columns:
            if col in exclude_cols: continue 
            try: df[col] = pd.to_numeric(df[col], errors='ignore')
            except: pass
        
        # 映射列名
        pct_col = '涨跌幅' if '涨跌幅' in df.columns else 'diff_rate'
        
        # 强制清洗
        df = df.dropna(subset=[pct_col])
        df[pct_col] = pd.to_numeric(df[pct_col], errors='coerce')
        
        # 基础过滤
        mask = (df[pct_col] >= pct_range[0]) & (df[pct_col] <= pct_range[1])
        
        amt_col = '成交额' if '成交额' in df.columns else 'amount'
        if amt_col in df.columns and amount_min > 0:
            # 这里的单位转换逻辑：A股输入单位是万元，所以需要 *10000 还原为元进行比较
            limit_val = amount_min * 10000 if market == "A股 (沪深)" else amount_min
            mask = mask & (df[amt_col] >= limit_val)
            
        to_col = '换手率' if '换手率' in df.columns else 'turnover'
        if to_col in df.columns and turnover_min > 0:
            mask = mask & (df[to_col] >= turnover_min)
            
        vr_col = '量比'
        if vr_col in df.columns and vr_min > 0:
            mask = mask & (df[vr_col] >= vr_min)
            
        pe_col = '市盈率-动态'
        if pe_col in df.columns and pe_max > 0:
            mask = mask & (df[pe_col] <= pe_max) & (df[pe_col] > 0)

        filtered_df = df[mask].copy()
        
        # 技术筛选
        final_df = filtered_df
        if use_tech and strategies:
            max_check = 200 if market == "A股 (沪深)" else 100
            check_list = filtered_df.head(max_check)
            
            code_col = '代码' if '代码' in df.columns else 'symbol'
            if code_col not in check_list.columns: code_col = 'code'
            
            codes_to_check = check_list[code_col].tolist()
            st.info(f"🚀 正在并发分析 {len(codes_to_check)} 只股票...")
            
            start_time = time.time()
            task_args = [(c, market, strategies) for c in codes_to_check]
            
            # 使用10个线程并发，提高速度
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                results = executor.map(process_stock_task, task_args)
            
            valid_codes = [r for r in results if r is not None]
            end_time = time.time()
            st.caption(f"⚡ 技术分析耗时: {end_time - start_time:.2f} 秒")
                    
            final_df = filtered_df[filtered_df[code_col].isin(valid_codes)]

        # === 核心修正 2: 结果展示前的数据清洗 ===
        st.success(f"筛选完成！命中 {len(final_df)} 只")
        
        # A. 修复代码前导零 (只针对A股)
        code_col = '代码' if '代码' in final_df.columns else 'symbol'
        if market == "A股 (沪深)" and code_col in final_df.columns:
            final_df[code_col] = final_df[code_col].apply(lambda x: f"{int(x):06d}" if str(x).isdigit() else x)

        # B. 优化成交额显示 (转为亿元)
        amt_raw_col = '成交额' if '成交额' in final_df.columns else 'amount'
        display_amt_col = amt_raw_col 
        
        if amt_raw_col in final_df.columns:
            new_col_name = '成交额(亿)'
            # 将原始数值除以1亿，方便阅读。对于美股/港股，这里显示的是 亿美元/亿港币
            final_df[new_col_name] = (final_df[amt_raw_col] / 100000000).round(2)
            display_amt_col = new_col_name 

        # 设置展示列
        display_cols = []
        priority = [code_col, '名称', 'name', '最新价', 'price', '涨跌幅', 'diff_rate', 
                   display_amt_col, '换手率', '量比', '市盈率-动态']
        
        for c in priority:
            if c in final_df.columns:
                display_cols.append(c)
                
        st.dataframe(final_df[display_cols].head(100), use_container_width=True)
        
        if code_col in final_df.columns:
            st.subheader("📋 代码列表")
            st.code(",".join(final_df[code_col].astype(str).tolist()))
        
    else:
        st.error("未获取到行情数据。")
