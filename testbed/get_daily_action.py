#!/usr/bin/env python3
"""
Daily Action Script - Run get_daily_news.py to fetch data, then read the last day's data
"""

import os
import sys
import json
import subprocess
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_get_daily_news(symbol, date_str):
    """Run get_daily_news.py to fetch data"""
    print(f"🚀 Running get_daily_news.py {symbol} {date_str}...")
    
    try:
        result = subprocess.run(
            [sys.executable, 'get_daily_news.py', symbol, date_str],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully fetched data for {symbol} on {date_str}")
            return True
        else:
            print(f"❌ Failed to fetch data: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Runtime error: {str(e)}")
        return False

def get_last_day_data(symbol, date_str=None):
    """Read data for specified date, or read the last day's data if no date is specified"""
    json_file = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file does not exist: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print(f"❌ JSON file is empty: {json_file}")
            return None
        
        # If date is specified, use specified date; otherwise use the last day
        if date_str:
            if date_str not in data:
                print(f"❌ Data for date {date_str} does not exist")
                return None
            target_date = date_str
            target_data = data[date_str]
            print(f"\n📊 Data for {symbol} on {date_str}:")
        else:
            dates = sorted(data.keys())
            target_date = dates[-1]
            target_data = data[target_date]
            print(f"\n📊 Last day's data for {symbol} ({target_date}):")
        
        print(f"💰 Price: ${target_data.get('prices', 'N/A')}")
        print(f"📊 Sentiment: {target_data.get('momentum', 'N/A')}")
        print(f"📰 News analysis length: {len(target_data.get('news', '')[0])} characters")
        # print(target_data.get('news', ''))
        # print(f"📈 Future price diff: {target_data.get('future_price_diff', 'N/A')}")
        
        return {
            "date": target_date,
            "data": target_data
        }
        
    except Exception as e:
        print(f"❌ Failed to read JSON file: {str(e)}")
        return None

def save_decision_to_json(symbol, date_str, price, decision, model="gpt-4o", identifier=None, symbol_name=None):
    """Save decision directly to JSON file with quarterly structure"""
    model_str = model.replace("-", "_")  # Replace hyphens with underscores for filename
    if identifier:
        json_file = f"action/{identifier}_{symbol_name}_{model_str}_trading_decisions.json"
    else:
        json_file = f"action/{model_str}_trading_decisions.json"
    
    # Ensure action directory exists
    os.makedirs("action", exist_ok=True)
    
    # Initialize or load existing data
    decisions_data = {
        "status": "success",
        "start_date": date_str,
        "end_date": date_str,
        "model": model,
        "recommendations": []
    }
    
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                decisions_data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read existing decision file: {e}")
    
    # Get existing recommendations if any
    all_recommendations = decisions_data.get("recommendations", [])
    
    # Get news data from paper_trading file
    news_data = None
    news_count = 0
    momentum = "neutral"
    paper_trading_file = f"data/paper_trading_{symbol}.json"
    if os.path.exists(paper_trading_file):
        try:
            with open(paper_trading_file, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
                if date_str in paper_data:
                    news_data = paper_data[date_str].get("news", [""])[0]
                    news_count = 1 if news_data else 0
                    momentum = paper_data[date_str].get("momentum", "neutral")
        except Exception as e:
            print(f"⚠️ Failed to read news data: {e}")
    
    # Create recommendation entry
    recommendation = {
        "date": date_str,
        "price": price,
        "recommended_action": decision.get("recommended_action", "HOLD"),
        "news_count": news_count,
        "sentiment": momentum,  # Use momentum value from paper_trading file
        "news": news_data if news_data else "No news available"
    }
    
    # Add or update current recommendation in all_recommendations
    updated = False
    for i, rec in enumerate(all_recommendations):
        if rec["date"] == date_str:
            all_recommendations[i] = recommendation
            updated = True
            break
    if not updated:
        all_recommendations.append(recommendation)

    # Sort recommendations by date
    all_recommendations.sort(key=lambda x: x["date"])

    # Update start_date and end_date based on all recommendations
    if all_recommendations:
        decisions_data["start_date"] = all_recommendations[0]["date"]
        decisions_data["end_date"] = all_recommendations[-1]["date"]
        decisions_data["recommendations"] = all_recommendations
    
    # Save to file
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(decisions_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Decision saved to {json_file} (sorted by quarter and date)")
        return True
    except Exception as e:
        print(f"❌ Failed to save decision: {e}")
        return False

def get_previous_valid_price(symbol, date_str):
    """Find the most recent valid price before the specified date from historical data"""
    json_file = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file does not exist: {json_file}")
        return None
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print(f"❌ JSON file is empty: {json_file}")
            return None
        
        # Get all dates and sort them
        all_dates = sorted(data.keys())
        
        # Find all dates before the specified date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        previous_dates = []
        
        for date_key in all_dates:
            check_date = datetime.strptime(date_key, '%Y-%m-%d').date()
            if check_date < target_date:
                previous_dates.append(date_key)
        
        # Start looking for valid prices from the most recent date
        previous_dates.sort(reverse=True)
        
        for prev_date in previous_dates:
            prev_price = data[prev_date].get('prices')
            if prev_price is not None:
                print(f"📊 {symbol} price is null on {date_str}, using price from {prev_date}: ${prev_price}")
                return prev_price
        
        print(f"⚠️ {symbol} cannot find valid historical price")
        return None
        
    except Exception as e:
        print(f"❌ Failed to find historical price: {str(e)}")
        return None

# Trading Strategy Definitions
TRADING_STRATEGIES = {
    "aggressive": {
        "name": "Aggressive Trading Strategy",
        "description": "This is an aggressive daily frequency trading strategy focused on predicting next-day price movements with decisive action.",
        "rules": {
            "BUY": "Open a long position when expecting price to rise",
            "SELL": "Open a short position when expecting price to fall",
            "HOLD": "Close all positions and stay neutral"
        },
        "timeframe": "Daily (1 day)",
        "objective": "Predict tomorrow's price trend and take corresponding positions"
    },
    # More strategies can be added here in the future
    # "prompt2": {...},
    # "prompt3": {...},
}

def get_trading_strategy_prompt(strategy_key="aggressive"):
    """Generate formatted trading strategy prompt"""
    if strategy_key not in TRADING_STRATEGIES:
        strategy_key = "aggressive"  # Default strategy
    
    strategy = TRADING_STRATEGIES[strategy_key]
    
    prompt = f"""Trading Strategy: {strategy['name']}

Description: {strategy['description']}

Trading Rules:
- BUY: {strategy['rules']['BUY']}
- SELL: {strategy['rules']['SELL']}
- HOLD: {strategy['rules']['HOLD']}

Timeframe: {strategy['timeframe']}
Objective: {strategy['objective']}"""
    
    return prompt

def calculate_new_position(current_position, recommended_action, strategy_key):
    """Calculate new position state based on current position, recommended action, and trading strategy"""
    
    # Get strategy definition
    if strategy_key not in TRADING_STRATEGIES:
        strategy_key = "aggressive"
    
    strategy = TRADING_STRATEGIES[strategy_key]
    
    # For aggressive strategy:
    # BUY: Open a long position
    # SELL: Open a short position
    # HOLD: Close all positions
    
    if strategy_key == "aggressive":
        if recommended_action == "BUY":
            # BUY means go long
            new_position = "LONG"
        elif recommended_action == "SELL":
            # SELL means go short
            new_position = "SHORT"
        elif recommended_action == "HOLD":
            # HOLD means close position, return to flat
            new_position = "FLAT"
        else:
            # Unknown action, keep current position
            new_position = current_position
    else:
        # Other strategies can add different conversion logic here
        new_position = current_position
    
    return new_position

def get_current_position(agent_id, symbol, model, strategy):
    """Get current position for the specified agent/model/asset/strategy combination"""
    position_file = "data/positions.json"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # If file does not exist, return default position HOLD
    if not os.path.exists(position_file):
        return "HOLD"
    
    try:
        with open(position_file, 'r', encoding='utf-8') as f:
            positions = json.load(f)
        
        # Build unique key: agent_model_symbol_strategy
        position_key = f"{agent_id}_{model}_{symbol}_{strategy}"
        
        # Return position for this combination, default is HOLD
        return positions.get(position_key, "HOLD")
        
    except Exception as e:
        print(f"⚠️ Failed to read position file: {e}")
        return "HOLD"

def update_current_position(agent_id, symbol, model, strategy, new_position):
    """Update current position for the specified agent/model/asset/strategy combination"""
    position_file = "data/positions.json"
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Read existing position data
    positions = {}
    if os.path.exists(position_file):
        try:
            with open(position_file, 'r', encoding='utf-8') as f:
                positions = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to read existing position file: {e}")
    
    # Build unique key: agent_model_symbol_strategy
    position_key = f"{agent_id}_{model}_{symbol}_{strategy}"
    
    # Update position
    positions[position_key] = new_position
    
    # Save to file
    try:
        with open(position_file, 'w', encoding='utf-8') as f:
            json.dump(positions, f, indent=2, ensure_ascii=False)
        print(f"💾 [{agent_id}] Position updated: {position_key} -> {new_position}")
        return True
    except Exception as e:
        print(f"❌ [{agent_id}] Failed to save position: {e}")
        return False

def get_history_prices(symbol, date_str, days=10):
    """Get historical price data for N days before the specified date"""
    json_file = f"data/paper_trading_{symbol}.json"
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file does not exist: {json_file}")
        return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print(f"❌ JSON file is empty: {json_file}")
            return []
        
        # Parse target date
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        
        # Get all dates and sort them
        all_dates = sorted(data.keys())
        
        # Find dates before target date and get the most recent N days of prices (only dates with valid prices)
        history_prices = []
        for date_key in reversed(all_dates):
            check_date = datetime.strptime(date_key, '%Y-%m-%d').date()
            if check_date < target_date:
                price = data[date_key].get('prices')
                # Only add dates with valid prices
                if price is not None:
                    history_prices.append({
                        "date": date_key,
                        "price": price
                    })
                    # Stop after getting enough data
                    if len(history_prices) >= days:
                        break
        
        # Sort in chronological order (oldest first, newest last)
        history_prices.reverse()
        
        if len(history_prices) < days:
            print(f"📈 Got {len(history_prices)} days of historical price data for {symbol} (less than requested {days} days)")
        else:
            print(f"📈 Got {len(history_prices)} days of historical price data for {symbol}")
        
        return history_prices
        
    except Exception as e:
        print(f"❌ Failed to get historical prices: {str(e)}")
        return []

def call_single_trading_api(url, identifier, symbol, date_str, price, news, model, ten_k, ten_q, original_price, history_price, trading_strategy, strategy_key):
    """Call a single trading_action API endpoint"""
    
    # Get current position
    current_position = get_current_position(identifier, symbol, model, strategy_key)
    print(f"📍 [{identifier}] Current position: {current_position}")
    
    # Prepare request data
    payload = {
        "date": date_str,
        "price": {symbol: price},
        "news": {
            symbol: news if isinstance(news, str) else news
        },
        "symbol": [symbol],
        "model": model,
        #"current_position": current_position
    }
    
    # Add 10K and 10Q data to payload
    if ten_k:
        payload["10k"] = {symbol: ten_k}
    if ten_q:
        payload["10q"] = {symbol: ten_q}
    
    # Add historical price data to payload (enabled by default, can be controlled via environment variable)
    enable_history_price = os.environ.get('ENABLE_HISTORY_PRICE', 'true').lower() == 'true'
    if enable_history_price and history_price:
        payload["history_price"] = {symbol: history_price}
        print(f"📊 [{identifier}] Including {len(history_price)} days of historical price data")
    
    # Add trading strategy to payload
    trading_strategy = False
    if trading_strategy:
        payload["trading_strategy"] = trading_strategy
        print(f"📋 [{identifier}] Using trading strategy: {trading_strategy}")
    
    try:
        print(f"🤖 [{identifier}] Calling trading_action API to get trading recommendations for {symbol}...")
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=None  # Disable timeout, wait indefinitely
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ [{identifier}] Successfully got trading recommendations")
            print(f"📊 [{identifier}] API response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # Calculate new position state based on recommended action and current position
            recommended_action = result.get("recommended_action", "HOLD")
            new_position = calculate_new_position(current_position, recommended_action, strategy_key)
            
            # Display position change
            if new_position != current_position:
                print(f"🔄 [{identifier}] Position changed: {current_position} -> {new_position} (Recommended: {recommended_action})")
            else:
                print(f"📍 [{identifier}] Position unchanged: {new_position} (Recommended: {recommended_action})")
            
            # Update position state
            update_current_position(identifier, symbol, model, strategy_key, new_position)
            
            # Save decision directly to JSON file (using original price, including null values)
            if original_price is None:
                print(f"💾 [{identifier}] Saving decision with original price: null (market closed that day)")
            else:
                print(f"💾 [{identifier}] Saving decision with original price: ${original_price}")
            save_decision_to_json(symbol, date_str, original_price, result, model, identifier, symbol)
            
            return {"identifier": identifier, "status": "success", "result": result}
        else:
            print(f"❌ [{identifier}] API call failed: HTTP {response.status_code}")
            print(f"Error message: {response.text}")
            return {"identifier": identifier, "status": "error", "error": f"HTTP {response.status_code}"}
            
    except requests.exceptions.ConnectionError:
        print(f"❌ [{identifier}] Cannot connect to trading_action API")
        print(f"Please ensure {identifier} service is running")
        return {"identifier": identifier, "status": "error", "error": "Connection failed"}
    except Exception as e:
        print(f"❌ [{identifier}] API call error: {str(e)}")
        return {"identifier": identifier, "status": "error", "error": str(e)}

def load_agents_config():
    """Load agent configuration from configs/agents.json"""
    config_file = "configs/agents.json"
    
    # Default configuration (fallback)
    default_agents = [
        {"name": "InvestorAgent", "url": "http://localhost:62233/trading_action/"},
        {"name": "TradeAgent", "url": "http://localhost:62234/trading_action/"},
        {"name": "HedgeFundAgent", "url": "http://localhost:62235/trading_action/"},
        {"name": "DeepFundAgent", "url": "http://localhost:62236/trading_action/"}
    ]
    
    # Try to load from file
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                agents = config_data.get("agents", default_agents)
                print(f"✅ Loaded {len(agents)} agent configurations from {config_file}")
                return agents
        except Exception as e:
            print(f"⚠️ Failed to read agent config file: {e}, using default configuration")
            return default_agents
    else:
        print(f"⚠️ Config file {config_file} does not exist, using default configuration")
        return default_agents

def call_trading_action_api(symbol, date_str, price, news, model="gpt-4o", identifier='Finagent', ten_k=None, ten_q=None, trading_strategy=None):
    """Call trading_action API to get trading recommendations (parallelized)"""
    
    # Save original price (including null values) for logging
    original_price = price
    
    # Check if price is null, if so try to fill using historical price for API call
    if price is None:
        print(f"⚠️ {symbol} price is null on {date_str}, trying to fill with historical price...")
        price = get_previous_valid_price(symbol, date_str)
        
        if price is None:
            print(f"❌ {symbol} cannot get valid price, skipping trading decision generation")
            return None
    
    # Get historical price data for the past 10 days
    history_price = get_history_prices(symbol, date_str, days=10)
    
    # Set trading strategy - read from environment variable or use default
    strategy_key = os.environ.get('TRADING_STRATEGY', 'aggressive')
    if trading_strategy is None:
        trading_strategy = get_trading_strategy_prompt(strategy_key)
        print(f"📋 Using trading strategy: {strategy_key}")
    
    # Print SEC report information
    if ten_k:
        print(f"📄 Including {len(ten_k)} 10K reports")
    if ten_q:
        print(f"📈 Including {len(ten_q)} 10Q reports")
    if not ten_k and not ten_q:
        print(f"📁 No SEC report data passed")
    
    # Load agent configuration from file
    agents_config = load_agents_config()
    agent_map = {agent["name"]: (agent["url"], agent["name"]) for agent in agents_config}
    
    # Read agents to use from environment variable, default to all
    agents_env = os.environ.get('AGENTS', 'all').strip()
    if agents_env.lower() == 'all':
        api_endpoints = list(agent_map.values())
    else:
        selected_agents = [a.strip() for a in agents_env.split(',')]
        api_endpoints = [agent_map[a] for a in selected_agents if a in agent_map]
        if not api_endpoints:
            print(f"⚠️ No valid agent found: {agents_env}, using all agents")
            api_endpoints = list(agent_map.values())
    
    print(f"\n🚀 Starting parallel calls to {len(api_endpoints)} trading agents...")
    
    # Use ThreadPoolExecutor to parallelize API calls
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_agent = {
            executor.submit(
                call_single_trading_api,
                url, agent_id, symbol, date_str, price, news, 
                model, ten_k, ten_q, original_price, history_price, trading_strategy, strategy_key
            ): agent_id
            for url, agent_id in api_endpoints
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_agent):
            agent_id = future_to_agent[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ [{agent_id}] Task execution exception: {str(e)}")
                results.append({"identifier": agent_id, "status": "error", "error": str(e)})
    
    # Summary
    success_count = sum(1 for r in results if r.get("status") == "success")
    print(f"\n✨ Parallel calls completed: {success_count}/{len(api_endpoints)} agents successful")
    
    return results

def main():
    if len(sys.argv) == 5:
        # Format: python get_daily_action.py <SYMBOL> <DATE> <MODEL> <IDENTIFIER>
        symbol = sys.argv[1].upper()
        date_str = sys.argv[2]
        model = sys.argv[3]
        identifier = sys.argv[4]
        
        print(f"🤖 Using model: {model}")
        print(f"🏷️ Using identifier: {identifier}")
        
        # 1. Run get_daily_news.py to fetch data
        success = run_get_daily_news(symbol, date_str)
        
        if success:
            # 2. Read data for specified date
            target_data = get_last_day_data(symbol, date_str)
            
            if target_data:
                # 3. Call trading_action API to get trading recommendations
                call_trading_action_api(
                    symbol, 
                    target_data["date"], 
                    target_data["data"]["prices"], 
                    target_data["data"]["news"],
                    model,
                    identifier,
                    target_data["data"].get("10k", []),
                    target_data["data"].get("10q", [])
                )
        else:
            print("❌ Data fetch failed, cannot read data for specified date")
            
    elif len(sys.argv) == 4:
        # Format: python get_daily_action.py <SYMBOL> <DATE> <MODEL>
        symbol = sys.argv[1].upper()
        date_str = sys.argv[2]
        model = sys.argv[3]
        
        print(f"🤖 Using model: {model}")
        
        # 1. Run get_daily_news.py to fetch data
        success = run_get_daily_news(symbol, date_str)
        
        if success:
            # 2. Read data for specified date
            target_data = get_last_day_data(symbol, date_str)
            
            if target_data:
                # 3. Call trading_action API to get trading recommendations
                call_trading_action_api(
                    symbol, 
                    target_data["date"], 
                    target_data["data"]["prices"], 
                    target_data["data"]["news"],
                    model,
                    "Finagent",
                    target_data["data"].get("10k", []),
                    target_data["data"].get("10q", [])
                )
        else:
            print("❌ Data fetch failed, cannot read data for specified date")
            
    elif len(sys.argv) == 3:
        # Format: python get_daily_action.py <SYMBOL> <DATE> (using default model gpt-4o)
        symbol = sys.argv[1].upper()
        date_str = sys.argv[2]
        model = "gpt-4o"  # Default model
        
        print(f"🤖 Using default model: {model}")
        
        # 1. Run get_daily_news.py to fetch data
        success = run_get_daily_news(symbol, date_str)
        
        if success:
            # 2. Read data for specified date
            target_data = get_last_day_data(symbol, date_str)
            
            if target_data:
                # 3. Call trading_action API to get trading recommendations
                call_trading_action_api(
                    symbol, 
                    target_data["date"], 
                    target_data["data"]["prices"], 
                    target_data["data"]["news"],
                    model,
                    "Finagent",
                    target_data["data"].get("10k", []),
                    target_data["data"].get("10q", [])
                )
        else:
            print("❌ Data fetch failed, cannot read data for specified date")
            
    elif len(sys.argv) == 2:
        # Format: python get_daily_action.py <SYMBOL> (using today's date and default model)
        symbol = sys.argv[1].upper()
        today = datetime.now().strftime('%Y-%m-%d')
        model = "gpt-4o"  # Default model
        
        print(f"📅 Using today's date: {today}")
        print(f"🤖 Using default model: {model}")
        
        # 1. Run get_daily_news.py to fetch today's data
        success = run_get_daily_news(symbol, today)
        
        if success:
            # 2. Read the last day's data
            last_data = get_last_day_data(symbol)
            
            if last_data:
                # 3. Call trading_action API to get trading recommendations
                call_trading_action_api(
                    symbol, 
                    last_data["date"], 
                    last_data["data"]["prices"], 
                    last_data["data"]["news"],
                    model,
                    "Finagent",
                    last_data["data"].get("10k", []),
                    last_data["data"].get("10q", [])
                )
        else:
            print("❌ Data fetch failed, cannot read last day's data")
    
    else:
        print("📖 Usage:")
        print("  # Get data for specified date and get trading recommendations")
        print("  python get_daily_action.py <SYMBOL> <DATE> <model>")
        print("  python get_daily_action.py BTC 2025-08-15 gpt-4o")
        print()

if __name__ == "__main__":
    main()
