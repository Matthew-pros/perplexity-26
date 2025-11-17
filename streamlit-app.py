"""
Streamlit Dashboard - Real-time monitoring a vizualizace výsledků
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import glob
from pathlib import Path

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
    .positive {
        color: #00ff00;
    }
    .negative {
        color: #ff0000;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache 5 minut
def load_latest_recommendations():
    """Load nejnovější screening results"""
    files = glob.glob('data/results/recommendations_*.csv')
    if not files:
        return pd.DataFrame()
    
    latest_file = max(files)
    df = pd.read_csv(latest_file)
    return df


@st.cache_data(ttl=300)
def load_performance_history():
    """Load historickou performance"""
    files = glob.glob('data/results/performance_*.csv')
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def main():
    # Header
    st.title("🎯 Automated Quant Trading Dashboard")
    st.markdown("Real-time monitoring s 5-strategy framework + RL agent")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        refresh = st.button("🔄 Refresh Data")
        if refresh:
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        show_strategy = st.multiselect(
            "Show Strategies",
            ['earnings_catalyst', 'institutional_flow', 'technical_breakout', 
             'short_squeeze', 'sentiment'],
            default=['earnings_catalyst', 'institutional_flow', 'technical_breakout']
        )
        
        min_score = st.slider("Minimum Score", 0, 100, 70)
        
        st.markdown("---")
        st.info(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Load data
    recommendations = load_latest_recommendations()
    performance_history = load_performance_history()
    
    if recommendations.empty:
        st.warning("⚠️ No data available. Run screening first.")
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
            color = "positive" if weekly_return > 0 else "negative"
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
    
    # Top stock per strategy
    cols = st.columns(5)
    
    strategies = ['earnings_catalyst', 'institutional_flow', 'technical_breakout',
                 'short_squeeze', 'sentiment']
    strategy_names = ['Earnings', 'Institutional', 'Technical', 'Short Squeeze', 'Sentiment']
    
    for i, (col, strategy, name) in enumerate(zip(cols, strategies, strategy_names)):
        with col:
            # Simuluj score pro každou strategii (v praxi by bylo v datech)
            st.metric(name, "✅")
            
            # Top ticker pro tuto strategii
            if not recommendations.empty:
                top_ticker = recommendations.iloc[i % len(recommendations)]['ticker']
                st.caption(f"Top: {top_ticker}")
    
    # Performance History Chart
    if not performance_history.empty:
        st.markdown("---")
        st.subheader("📈 Portfolio Performance History")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_history['date'],
            y=performance_history['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#2ecc71', width=2)
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
    st.caption("🤖 Powered by RL Agent | Last GitHub Action: See workflow logs")


if __name__ == "__main__":
    main()
