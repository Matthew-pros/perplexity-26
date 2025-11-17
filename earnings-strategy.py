"""
Earnings Catalyst Strategy - Identifikuje stocks s vysokou pravděpodobností earnings beat
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict
from loguru import logger


class EarningsStrategy:
    """
    Strategy založená na earnings surprise patterns
    
    Scoring factors:
    - Consecutive earnings beats (3+): 30 bodů
    - Positive Earnings ESP > 2%: 25 bodů
    - Pre-earnings IV expansion: 20 bodů
    - Historical vs implied move arbitrage: 25 bodů
    """
    
    def __init__(self):
        self.name = "earnings_catalyst"
        logger.info(f"Initialized {self.name} strategy")
    
    def calculate_score(self, 
                       ticker: str,
                       stock: yf.Ticker,
                       hist: pd.DataFrame) -> float:
        """
        Calculate earnings catalyst score (0-100)
        
        Args:
            ticker: Stock ticker symbol
            stock: yfinance Ticker object
            hist: Historical price data
            
        Returns:
            Score 0-100
        """
        score = 0
        
        try:
            # 1. Check for consecutive earnings beats
            earnings_history = self._get_earnings_history(stock)
            beat_streak = self._count_consecutive_beats(earnings_history)
            
            if beat_streak >= 3:
                score += 30
                logger.debug(f"{ticker}: {beat_streak} consecutive beats (+30)")
            elif beat_streak >= 2:
                score += 20
            elif beat_streak >= 1:
                score += 10
            
            # 2. Check earnings surprise prediction (ESP)
            # V production by se použil Zacks ESP nebo podobný
            # Pro demo simulujeme
            esp = self._estimate_earnings_surprise(stock)
            
            if esp > 5:
                score += 25
                logger.debug(f"{ticker}: ESP {esp:.1f}% (+25)")
            elif esp > 2:
                score += 15
            elif esp > 0:
                score += 5
            
            # 3. Check IV expansion (pokud máme options data)
            # Pro yfinance free data nemáme IV, simulujeme z volatility
            iv_expansion = self._check_iv_expansion(hist)
            
            if iv_expansion > 0.15:
                score += 20
                logger.debug(f"{ticker}: IV expansion {iv_expansion:.1%} (+20)")
            elif iv_expansion > 0.05:
                score += 10
            
            # 4. Historical vs implied move arbitrage
            # Pokud historický průměrný earnings move je větší než současný implied
            move_arbitrage = self._calculate_move_arbitrage(hist)
            
            if move_arbitrage > 0:
                score += 25
                logger.debug(f"{ticker}: Positive move arbitrage (+25)")
            elif move_arbitrage > -0.05:
                score += 10
            
        except Exception as e:
            logger.warning(f"{ticker} earnings strategy error: {str(e)}")
            return 0
        
        return min(score, 100)  # Cap at 100
    
    def _get_earnings_history(self, stock: yf.Ticker) -> pd.DataFrame:
        """Get historical earnings data"""
        try:
            # yfinance earnings data
            earnings = stock.earnings_dates
            if earnings is None or earnings.empty:
                return pd.DataFrame()
            
            # Filter to last 8 quarters
            earnings = earnings.head(8)
            return earnings
            
        except Exception as e:
            logger.warning(f"Error getting earnings history: {str(e)}")
            return pd.DataFrame()
    
    def _count_consecutive_beats(self, earnings: pd.DataFrame) -> int:
        """Count consecutive earnings beats"""
        if earnings.empty:
            return 0
        
        streak = 0
        
        # Check if EPS Estimate and Reported EPS columns exist
        if 'EPS Estimate' not in earnings.columns or 'Reported EPS' not in earnings.columns:
            return 0
        
        for _, row in earnings.iterrows():
            estimate = row['EPS Estimate']
            reported = row['Reported EPS']
            
            if pd.isna(estimate) or pd.isna(reported):
                break
            
            if reported > estimate:
                streak += 1
            else:
                break
        
        return streak
    
    def _estimate_earnings_surprise(self, stock: yf.Ticker) -> float:
        """
        Estimate earnings surprise prediction (ESP)
        
        V production by používal:
        - Zacks ESP
        - Estimize data
        - Whisper numbers
        
        Pro demo: simulujeme z trendu
        """
        try:
            info = stock.info
            
            # Zkontroluj growth metrics
            earnings_growth = info.get('earningsGrowth', 0)
            revenue_growth = info.get('revenueGrowth', 0)
            
            if earnings_growth and revenue_growth:
                # Pokud oboje rostou, higher ESP
                avg_growth = (earnings_growth + revenue_growth) / 2
                esp = avg_growth * 100  # Convert to percentage
                return esp
            
            return 0
            
        except Exception:
            return 0
    
    def _check_iv_expansion(self, hist: pd.DataFrame) -> float:
        """
        Check for implied volatility expansion
        
        V production by používal skutečnou IV z options
        Pro demo: používáme realized volatility jako proxy
        """
        if len(hist) < 30:
            return 0
        
        # Calculate 20-day vs 5-day volatility
        returns = hist['Close'].pct_change()
        
        vol_20d = returns.tail(20).std() * np.sqrt(252)
        vol_5d = returns.tail(5).std() * np.sqrt(252)
        
        # Pokud recent volatility je vyšší = IV expansion
        iv_expansion = (vol_5d - vol_20d) / vol_20d
        
        return iv_expansion
    
    def _calculate_move_arbitrage(self, hist: pd.DataFrame) -> float:
        """
        Calculate historical earnings move vs current implied move
        
        V production:
        - Actual earnings dates a moves
        - Options IV implied move
        
        Pro demo: Simulujeme z volatility patterns
        """
        if len(hist) < 60:
            return 0
        
        # Approximate earnings moves (každý ~90 dnů)
        # Find largest moves in rolling 5-day windows
        returns = hist['Close'].pct_change()
        rolling_max = returns.rolling(5).sum().abs()
        
        # Top 4 moves (assumuji 4 earnings v roce)
        top_moves = rolling_max.nlargest(4).mean()
        
        # Current volatility
        current_vol = returns.tail(20).std() * np.sqrt(5)  # 5-day move
        
        # Arbitrage = historical average - current expectation
        arbitrage = top_moves - current_vol
        
        return arbitrage


def main():
    """Test strategy"""
    strategy = EarningsStrategy()
    
    # Test ticker
    ticker = "NVDA"
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    
    score = strategy.calculate_score(ticker, stock, hist)
    
    print(f"\n{ticker} Earnings Catalyst Score: {score:.1f}/100")


if __name__ == "__main__":
    main()
