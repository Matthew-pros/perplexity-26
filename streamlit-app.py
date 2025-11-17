"""
Streamlit Dashboard - Minimal Version for Cloud Deployment
Works even without complete data
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import glob
from pathlib import Path
import os

# Page config
st.set_page_config(
    page_title="Quant Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .positive { color: #00ff00; }
    .negative { color: #ff0000; }
    </style>
""", unsafe_allow_html=True)


def create_demo_data():
    """Create demo data if real data doesn't exist"""
    tickers = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM']
    
    data = []
    for ticker in tickers:
        score = np.random.uniform(60, 95)
        confluence = np.random.randint(2, 5)
        price = np.random.uniform(50, 500)
        
        data.append({
            'ticker': ticker,
            'score': score,
            'confluence': confluence,
            'entry_price': price,
            'stop_loss': price * 0.95,
            'target_1': price * 1.08,
            'confidence': 'HIGH' if confluence >= 3 else 'MEDIUM'
        })
    
    return pd.DataFrame(data).sort_values('score', ascending=False)


def create_demo_performance():
    """Create demo performance history"""
    dates = pd.date_range(start='2024-01-01', periods=52, freq='W')
    
    portfolio_values = [100000]
    for _ in range(51):
        ret = np.random.normal(0.01, 0.03)
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    return pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'weekly_return': [0] + [pv / portfolio_values[i-1] - 1 for i, pv in enumerate(portfolio_values[1:], 1)],
        'sharpe_ratio': np.random.uniform(1.5, 2.5, 52),
        'max_drawdown': -np.random.uniform(0.05, 0.15, 52),
        'win_rate': np.random.uniform(0.55, 0.70, 52)
    })


@st.cache_data(ttl=300)
def load_latest_recommendations():
    """Load latest screening results or demo data"""
    try:
        files = glob.glob('data/results/recommendations_*.csv')
        if files:
            latest_file = max(files)
            return pd.read_csv(latest_file)
    except Exception:
        pass
    
    return create_demo_data()


@st.cache_data(ttl=300)
def load_performance_history():
    """Load performance history or demo data"""
    try:
        files = glob.glob('data/results/performance_*.csv')
        if files:
            dfs = [pd.read_csv(f) for f in files]
            return pd.concat(dfs, ignore_index=True)
    except Exception:
        pass
    
    return create_demo_performance()


def main():
    # Header
    st.title("🎯 Automated Quant Trading Dashboard")
    st.markdown("Real-time monitoring s 5-strategy framework + RL agent")
    
    # Info banner
    if not os.path.exists('data/results'):
        st.info("📊 Zobrazuji demo data. Pro reálná data spusť screening engine.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        refresh = st.button("🔄 Refresh Data")
        if refresh:
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        min_score = st.slider("Minimum Score", 0, 100, 70)
        
        st.markdown("---")
        st.info(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        st.markdown("---")
        st.markdown("### 📚 Resources")
        st.markdown("[GitHub Repo](https://github.com/yourusername/quant-trading-rl)")
        st.markdown("[Documentation](https://github.com/yourusername/quant-trading-rl/blob/main/README.md)")
    
    # Load data
    try:
        recommendations = load_latest_recommendations()
        performance_history = load_performance_history()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_opportunities = len(recommendations)
        st.metric("📈 Total Opportunities", total_opportunities)
    
    with col2:
        high_confidence = len(recommendations[recommendations['confidence'] == 'HIGH'])
        st.metric("⭐ High Confidence", high_confidence)
    
    with col3:
        avg_score = recommendations['score'].mean()
        st.metric("📊 Avg Score", f"{avg_score:.1f}")
    
    with col4:
        if not performance_history.empty:
            weekly_return = performance_history['weekly_return'].iloc[-1]
            st.metric("💰 Last Week Return", 
                     f"{weekly_return:.2%}",
                     delta=f"{weekly_return:.2%}")
    
    st.markdown("---")
    
    # Top Recommendations Table
    st.subheader("🏆 Top Recommendations")
    
    # Filter by score
    filtered = recommendations[recommendations['score'] >= min_score]
    
    # Display table
    display_cols = ['ticker', 'score', 'confluence', 'entry_price', 
                   'stop_loss', 'target_1', 'confidence']
    
    styled_df = filtered[display_cols].head(15).style.format({
        'score': '{:.1f}',
        'entry_price': '${:.2f}',
        'stop_loss': '${:.2f}',
        'target_1': '${:.2f}'
    }).background_gradient(subset=['score'], cmap='RdYlGn')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Download button
    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download Recommendations CSV",
        data=csv,
        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Score Distribution")
        
        fig = px.histogram(
            recommendations,
            x='score',
            nbins=20,
            title='Distribution of Opportunity Scores',
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="Score",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Confluence Analysis")
        
        confluence_counts = recommendations['confluence'].value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=confluence_counts.index,
                y=confluence_counts.values,
                marker_color=['#ff6b6b', '#ffa06b', '#ffe66d', '#4ecdc4', '#95e1d3']
            )
        ])
        fig.update_layout(
            title='Strategies Confluence Count',
            xaxis_title='Number of Strategies',
            yaxis_title='Count',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Performance
    st.markdown("---")
    st.subheader("🔬 Strategy Breakdown")
    
    cols = st.columns(5)
    strategies = ['Earnings', 'Institutional', 'Technical', 'Short Squeeze', 'Sentiment']
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    
    for col, name, weight in zip(cols, strategies, weights):
        with col:
            st.metric(name, f"{weight*100:.0f}%")
            if not recommendations.empty:
                top_ticker = recommendations.iloc[0]['ticker']
                st.caption(f"Top: {top_ticker}")
    
    # Performance History
    if not performance_history.empty:
        st.markdown("---")
        st.subheader("📈 Portfolio Performance History")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_history['date'],
            y=performance_history['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2ecc71', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = (performance_history['portfolio_value'].iloc[-1] / 
                          performance_history['portfolio_value'].iloc[0] - 1)
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col2:
            sharpe = performance_history['sharpe_ratio'].iloc[-1]
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            max_dd = performance_history['max_drawdown'].min()
            st.metric("Max Drawdown", f"{max_dd:.2%}")
        
        with col4:
            win_rate = performance_history['win_rate'].mean()
            st.metric("Win Rate", f"{win_rate:.1%}")
    
    # Footer
    st.markdown("---")
    st.caption("🤖 Powered by RL Agent | Automated Weekly Rebalancing")
    
    # Expander with system info
    with st.expander("ℹ️ System Information"):
        st.markdown("""
        **System Components:**
        - 5 Strategy Framework (Earnings, Institutional, Technical, Short Squeeze, Sentiment)
        - Reinforcement Learning Agent (PPO/A2C)
        - Automated GitHub Actions workflow
        - Real-time data from yfinance
        
        **Next Rebalance:** Every Friday 18:00 CET
        
        **Risk Management:**
        - Max 2% risk per trade
        - 2 ATR stop loss
        - Position sizing via Kelly criterion
        """)


if __name__ == "__main__":
    main()
