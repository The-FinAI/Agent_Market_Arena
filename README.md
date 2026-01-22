# Agent Market Arena

This is a testbed environment for testing and evaluating trading agents. The testbed provides a standardized framework for wrapping agents as HTTP endpoints and calling them for trading decisions.

## Overview

The testbed is designed to:
1. **Wrap trading agents as HTTP endpoints** - Each agent needs to be deployed as a REST API service
2. **Call agents for trading decisions** - The testbed orchestrates calls to multiple agents in parallel
3. **Collect and evaluate results** - Trading decisions are saved and can be analyzed for performance

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Testbed                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │ get_daily_    │  │ get_daily_    │  │ get_return.py │        │
│  │ news.py       │──▶│ action.py    │──▶│ (Analysis)    │        │
│  │ (Data Fetch)  │  │ (Orchestrate) │  │               │        │
│  └───────────────┘  └───────────────┘  └───────────────┘        │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Agent Endpoints                          ││
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            ││
│  │  │   Agent A   │ │   Agent B  │ │   Agent C   │  ...      ││
│  │  │ :62233      │ │ :62234      │ │ :62235       │           ││
│  │  └─────────────┘ └─────────────┘ └─────────────┘            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Configure API Keys

Create a `.env.local` file in the testbed directory and add your API keys:

```bash
# .env.local
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GEMINI_API_KEY="your-gemini-key"
export TOGETHER_API_KEY="your-together-key"
export CRYPTONEWS_API_KEY="your-cryptonews-key"
export FINNHUB_API_KEY="your-finnhub-key"
export NEWSDATA_API_KEY="your-newsdata-key"
# ... other keys as needed
```

Then source it before running:
```bash
source .env.local
```

### 2. Start Agent Endpoints

**Each agent must be wrapped as an HTTP endpoint before running the testbed.**

Configure your agents in `configs/agents.json`:

```json
{
  "agents": [
    {
      "name": "AgentA",
      "url": "http://localhost:8001/trading_action/"
    },
    {
      "name": "AgentB", 
      "url": "http://localhost:8002/trading_action/"
    },
    {
      "name": "AgentC",
      "url": "http://localhost:8003/trading_action/"
    }
  ]
}
```

### 3. Configure Models

Edit `configs/models.json` to specify which LLM models to use:

```json
{
  "models": [
    {
      "name": "claude-sonnet-4-20250514",
      "chat_model_inference_engine": "anthropic",
      "chat_endpoint": "https://api.anthropic.com/v1/messages"
    },
    {
      "name": "gpt-4.1",
      "chat_model_inference_engine": "openai",
      "chat_endpoint": "https://api.openai.com/v1/chat/completions"
    }
  ]
}
```

## Usage

### Run Daily Action (Manual)

```bash
# Run for a single asset and date
python get_daily_action.py BTC 2025-01-15 gpt-4o

# Run for multiple assets and dates
./get_action.sh
```

### Run Automated Daily Pipeline

```bash
# Run the full daily pipeline manually
./auto_daily_action.sh run

# Setup automatic cron job (runs at 00:01 UTC daily)
./auto_daily_action.sh setup
```

### Analyze Returns

```bash
python get_return.py
```

## Agent Endpoint API Specification

Each agent endpoint must implement the following REST API:

### Endpoint: `POST /trading_action/`

**Request Body:**
```json
{
  "date": "2025-01-15",
  "price": {"BTC": 45000.00},
  "news": {"BTC": "News summary text..."},
  "symbol": ["BTC"],
  "model": "gpt-4o",
  "10k": {"BTC": ["10K filing 1", "10K filing 2"]},
  "10q": {"BTC": ["10Q filing 1"]},
  "history_price": {"BTC": [{"date": "2025-01-14", "price": 44500}]}
}
```

**Response Body:**
```json
{
  "recommended_action": "BUY",
  "reasoning": "Based on the analysis..."
}
```

**Supported Actions:**
- `BUY` - Open a long position
- `SELL` - Open a short position  
- `HOLD` - Close all positions / stay flat

## Directory Structure

```
testbed/
├── auto_daily_action.sh      # Automated daily pipeline script
├── get_action.sh             # Batch action generation script
├── get_daily_action.py       # Daily action orchestrator
├── get_daily_news.py         # News and price data fetcher
├── get_return.py             # Return analysis tool
├── configs/                  # Configuration files
│   ├── agents.json           # Agent endpoint configuration
│   └── models.json           # LLM model configuration
├── data/                     # Data storage
│   └── paper_trading_*.json  # Historical price and news data
└── action/                   # Trading decisions output
    └── *_trading_decisions.json
```

## Supported Assets

### Cryptocurrencies
BTC, ETH, ADA, SOL, DOT, LINK, UNI, MATIC, AVAX, ATOM

### Stocks
TSLA, AAPL, MSFT, GOOGL, AMZN, NVDA, META, NFLX, AMD, INTC, BMRN, MRNA

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GEMINI_API_KEY` | Google Gemini API key |
| `TOGETHER_API_KEY` | Together AI API key |
| `CRYPTONEWS_API_KEY` | CryptoNews API key |
| `FINNHUB_API_KEY` | Finnhub API key |
| `NEWSDATA_API_KEY` | NewsData API key |
| `AGENTS` | Comma-separated list of agents to use (default: "all") |
| `TRADING_STRATEGY` | Trading strategy to use (default: "aggressive") |
| `ENABLE_HISTORY_PRICE` | Enable historical price in payload (default: "true") |

## License

This testbed is provided as-is for research and evaluation purposes.
