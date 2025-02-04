# 🎯 QuickPredict AI Agent
![{BD14F932-9E92-4AF6-8745-A99934105617}](https://github.com/user-attachments/assets/4e601227-f554-40d8-a361-1a3143164ca8)

> 🤖 AI-powered price prediction agent that analyzes cryptocurrency market data and predicts 30-second price movements using technical analysis and machine learning.

## ✨ Current Features

- 📊 Real-time BTC/USDT market data analysis
- 🧠 GPT-3.5 Turbo powered predictions
- 📈 Technical indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Stochastic Oscillator
  - ADX (Average Directional Index)
- ⚡ 30-second price movement forecasts
- 📝 Automatic prediction verification
- 📊 Performance tracking:
  - Accuracy metrics
  - Success streaks
  - Confidence analysis
- 🎨 Beautiful console visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Supabase account and project

### Setup

1. Clone repository
2. Create virtual environment: `python -m venv venv`
3. Activate environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```
6. Run: `python main.py`

## 📊 Monitoring

The agent provides multiple ways to monitor its performance:

### Console Output
- Real-time technical analysis
- Live predictions with confidence levels
- Current performance metrics
- Success/failure verification

### Database Tables
- `technical_indicators`: Stores all calculated indicators
- `predictions`: Contains predictions and their verification
- `performance_metrics`: Tracks overall agent performance

### Log File
Detailed logs are available in `crypto_agent.log`

## ⚠️ Disclaimer

This is an experimental prediction tool for analysis purposes only. It is not intended for trading and should not be used to make investment decisions.

## 🛣️ Roadmap

- [ ] Support for multiple timeframes
- [ ] Additional technical indicators
- [ ] Backtesting functionality
- [ ] Web interface for monitoring
- [ ] Multiple cryptocurrency pairs support
- [ ] Advanced performance analytics
- [ ] Machine learning model integration

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions, issues and feature requests are welcome! Feel free to check [issues page](https://github.com/Huston1992/quickpredict-ai/issues).

## 🌟 Show your support

Give a ⭐️ if this project helped you!

## Structure

```
ai_price/
├── config/           # Configuration files
├── src/             # Source code
│   ├── analysis/    # Technical analysis
│   ├── database/    # Database interface
│   ├── models/      # Data models
│   └── utils/       # Utilities
└── tests/           # Tests
```

## Features

- Real-time price analysis
- AI-powered predictions
- Technical indicators
- Performance tracking
- Database storage
