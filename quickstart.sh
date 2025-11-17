#!/bin/bash

echo "🚀 Quant Trading System - Quick Start"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p data/historical data/models data/results logs

# Create .env template
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env template..."
    cat > .env << 'EOF'
# Optional: Alpaca Paper Trading
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true

# Optional: Enhanced Data Sources
POLYGON_API_KEY=

# Optional: Notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
    echo "✓ .env file created. Edit with your API keys if needed."
fi

# Test run
echo ""
echo "🧪 Running test screening..."
python -m src.screening.screener_engine

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys (optional)"
echo "2. Run: streamlit run streamlit_app/app.py"
echo "3. View dashboard at http://localhost:8501"
echo ""
echo "For deployment, see SETUP-GUIDE.md"
