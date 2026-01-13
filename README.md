# Stock_web_scanner
基于 Streamlit 的 A股/港美股全能选股器，全部代码由Gemini输出
支持 A股、港股、美股的实时涨跌幅筛选及 MACD/KDJ/RSI 技术面分析。

## ✨ 功能特点
- **多市场支持**：混合引擎，A股 (Akshare) + 港美股 (Yfinance)。
- **技术筛选**：支持 MACD 金叉、RSI 超卖、KDJ 金叉、布林带突破。
- **纯 Docker**：一键部署，无需配置环境。

## 🚀 快速部署 (Docker)

确保服务器已安装 Docker Compose，打开到对应文件夹的根目录然后运行：

```bash
# 1. 把你在 GitHub 上保存的代码拉下来
git clone https://github.com/wonglamho/Stock_web_scanner.git
# (系统会自动在当前目录下面创建一个 Stock_web_scanner 文件夹)

# 2. 进入目录
cd Stock_web_scanner

# 3. 启动！
docker compose up -d --build
