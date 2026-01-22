#!/bin/bash

# Set timezone to US Eastern Time
export TZ="America/New_York"

# Get the absolute path of the script directory
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

# Initialize and activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
echo "Activating conda environment: agent"
conda activate agent

# API Keys - Load from environment variables or .env file, do not hardcode here
# You can create a .env.local file and use: source .env.local to load
export OPENAI_API_KEY=${OPENAI_API_KEY:-""}
export TOGETHER_API_KEY=${TOGETHER_API_KEY:-""}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-""}
export GEMINI_API_KEY=${GEMINI_API_KEY:-""}
export GUARDRAILS_API_KEY=${GUARDRAILS_API_KEY:-""}
export CRYPTONEWS_API_KEY=${CRYPTONEWS_API_KEY:-""}
export NEWSDATA_API_KEY=${NEWSDATA_API_KEY:-""}
export FINNHUB_API_KEY=${FINNHUB_API_KEY:-""}
export ARB_RECORD_HASH_PK=${ARB_RECORD_HASH_PK:-""}
export PUBLIC_DECISION_KEY_HEX=${PUBLIC_DECISION_KEY_HEX:-""}
export FINAI_AUTH_HEADER=${FINAI_AUTH_HEADER:-""}

# Get trading date (YYYY-MM-DD format)
# Use Eastern Time zone to calculate current date (script runs after US market close)
YESTERDAY=$(TZ="America/New_York" date +%Y-%m-%d)

# Read trading symbols and models from config files
# Read data/asset_cache.json to get asset list
if [ -f "data/asset_cache.json" ]; then
    SYMBOLS=$(python3 -c "import json; data=json.load(open('data/asset_cache.json')); print(' '.join(data.keys()))")
    echo "✅ Loaded $(echo $SYMBOLS | wc -w) assets from data/asset_cache.json: $SYMBOLS"
else
    echo "⚠️ data/asset_cache.json not found, using default assets"
    SYMBOLS="TSLA BTC ETH MSFT BMRN MRNA"
fi

# Read configs/models.json to get model list
if [ -f "configs/models.json" ]; then
    MODELS=$(python3 -c "import json; data=json.load(open('configs/models.json')); print(' '.join([m['name'] for m in data['models']]))")
    echo "✅ Loaded $(echo $MODELS | wc -w) models from configs/models.json: $MODELS"
else
    echo "⚠️ configs/models.json not found, using default models"
    MODELS="gpt-4o gpt-4.1 gemini-2.0-flash claude-3-5-haiku-20241022 claude-sonnet-4-20250514"
fi

# Main function: run daily action
run_daily_action() {
    echo "=== Starting daily action process for date $YESTERDAY ==="
    echo "📊 Assets: $(echo $SYMBOLS | wc -w)"
    echo "🤖 Models: $(echo $MODELS | wc -w)"
    echo "📈 Total combinations: $(($(echo $SYMBOLS | wc -w) * $(echo $MODELS | wc -w)))"
    echo ""
    
    # Default: call all agents (read from configs/agents.json)
    # To specify specific agents, set: export AGENTS="TradeAgent,DeepFundAgent"
    
    # Run action retrieval for all trading symbols and models
    for symbol in $SYMBOLS; do
        for model in $MODELS; do
            echo "Running action for $symbol on date $YESTERDAY with model $model"
            python get_daily_action.py "$symbol" "$YESTERDAY" "$model"
            
            # Add delay to avoid potential API rate limits
            sleep 5
        done
    done
    
    echo "=== All actions completed, generating vote files ==="
    
    # Generate vote aggregation files
    echo "Generating vote aggregation files..."
    python vote_aggregator.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Vote files generated successfully"
    else
        echo "❌ Vote file generation failed"
    fi
    
    echo "=== Starting data upload ==="
    
    # Upload all data to database (full version)
    echo "Uploading all data to Supabase..."
    python sync_to_supabase.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Data upload completed successfully"
    else
        echo "❌ Data upload failed"
    fi
    
    # Upload filtered data (only specified assets and models)
    echo "Uploading filtered data to Supabase (BMRN, TSLA, BTC, ETH with selected models)..."
    python sync_to_supabase_filtered.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Filtered data upload completed successfully"
    else
        echo "❌ Filtered data upload failed"
    fi
    
    echo "=== Starting HuggingFace upload ==="
    
    # Install required dependencies (if not already installed)
    pip install -q huggingface_hub
    
    # Upload paper_trading files to HuggingFace
    echo "Uploading paper trading files to HuggingFace..."
    python upload_to_hf.py
    
    if [ $? -eq 0 ]; then
        echo "✅ HuggingFace upload completed successfully"
    else
        echo "❌ HuggingFace upload failed"
    fi
    
    echo "=== Starting Hash upload ==="
    
    # Upload daily decisions to Hash
    echo "Uploading daily decisions to Hash..."
    python upload_to_hash.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Hash upload completed successfully"
    else
        echo "❌ Hash upload failed"
    fi
    
    echo "=== Daily process completed ==="
}

# Setup crontab (run daily at UTC 0:01)
setup_cron() {
    # Remove old crontab entries
    crontab -l 2>/dev/null | grep -v "auto_daily_action.sh" | crontab -
    
    # Create new crontab entry - UTC 0:01
    (crontab -l 2>/dev/null; echo "1 0 * * * cd $SCRIPT_DIR && $SCRIPT_DIR/auto_daily_action.sh run >> $SCRIPT_DIR/cron.log 2>&1") | crontab -
    echo "Crontab has been set up. The script will run daily at 00:01 UTC"
}

# Execute different operations based on command line arguments
case "$1" in
    "run")
        run_daily_action
        ;;
    "setup")
        setup_cron
        ;;
    *)
        echo "Usage: $0 {run|setup}"
        echo "  run   - Run the daily action manually"
        echo "  setup - Set up automatic daily runs at 00:01 ET"
        exit 1
        ;;
esac

# sh auto_daily_action.sh setup
# crontab -l
# crontab -r
