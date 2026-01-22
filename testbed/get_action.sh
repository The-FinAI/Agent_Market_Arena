#!/bin/bash

# ===========================================
# Configuration Parameters - Modify here
# ===========================================

# Specify which Agent to use (TradeAgent, InvestorAgent, HedgeFundAgent, DeepFundAgent)
# You can specify multiple agents separated by comma, e.g.: "TradeAgent,DeepFundAgent"
# Or use "all" to call all agents
export AGENTS="HedgeFundAgent" 

# Asset symbol list (e.g.: BTC, TSLA, AAPL, etc.) BMRN
# ASSETS="ETH"
ASSETS="BTC ETH TSLA MSFT BMRN MRNA"
# ASSETS="BTC ETH TSLA"

# Start date (format: YYYY-MM-DD)
# START_DATE="2025-10-09"
START_DATE="2025-10-22" 
END_DATE="2025-12-06"

# Model name list (e.g.: gpt-4.1, gpt-4o)
# MODELS="claude-3-5-haiku-20241022"
MODELS="claude-sonnet-4-20250514 claude-haiku-4-5-20251001 claude-3-5-haiku-20241022" # gpt-4o gemini-2.0-flash deepseek-v3-1 qwen3-235b
#"deepseek-v3-1" # gpt-4o gemini-2.0-flash claude-3-5-haiku-20241022 claude-sonnet-4-20250514"

# ===========================================
# Script Logic - No modification needed
# ===========================================

echo "Generating trading decisions from $START_DATE to $END_DATE for $ASSETS..."
echo "Using models: $MODELS"
echo "=========================================="

# Convert dates to timestamps for iteration
current_date=$START_DATE
end_timestamp=$(date -d "$END_DATE" +%s)

while [ $(date -d "$current_date" +%s) -le $end_timestamp ]; do
    echo "📅 Processing date: $current_date"
    
    # Execute command for each asset and each model
    for asset in $ASSETS; do
        for model in $MODELS; do
            echo "🤖 Using model: $model"
            echo "🏷️ Using identifier: $asset"
            echo "🚀 Running get_daily_action.py $asset $current_date $model..."
            echo "python get_daily_action.py $asset $current_date $model"
            python get_daily_action.py $asset $current_date $model
            echo "----------------------------------------"
        done
    done
    
    # Increment by one day
    current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
done

echo "=========================================="
echo "All trading decisions generated!"
