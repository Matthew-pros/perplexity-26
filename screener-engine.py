"""
QuantScreeningEngine - Hlavní screening engine s multi-strategy framework
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from loguru import logger
import yaml

from src.strategies.earnings_strategy import EarningsStrategy
from src.strategies.institutional_flow import InstitutionalFlowStrategy
from src.strategies.technical_breakout import TechnicalBreakoutStrategy
from src.strategies.short_squeeze import ShortSqueezeStrategy
from src.strategies.sentiment_analysis import SentimentStrategy
from src.utils.config import Config


class QuantScreeningEngine:
    """
    Hlavní screening engine pro identifikaci top trading opportunities
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.strategies = self._initialize_strategies()
        logger.info("QuantScreeningEngine initialized")
        
    def _initialize_strategies(self) -> List[Tuple[str, object, float]]:
        """Initialize všechny strategie s jejich vahami"""
        return [
            ('earnings_catalyst', EarningsStrategy(), 0.30),
            ('institutional_flow', InstitutionalFlowStrategy(), 0.25),
            ('technical_breakout', TechnicalBreakoutStrategy(), 0.20),
            ('short_squeeze', ShortSqueezeStrategy(), 0.15),
            ('sentiment', SentimentStrategy(), 0.10)
        ]
    
    def get_stock_universe(self, universe: str = 'SP500') -> List[str]:
        """
        Získej seznam tickerů pro screening
        
        Args:
            universe: 'SP500', 'NASDAQ100', 'RUSSELL2000' nebo 'ALL'
        """
        logger.info(f"Fetching stock universe: {universe}")
        
        if universe == 'SP500':
            # SP500 tickers
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(url)[0]
            return df['Symbol'].str.replace('.', '-').tolist()
        
        elif universe == 'NASDAQ100':
            # NASDAQ 100
            url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
            df = pd.read_html(url)[4]
            return df['Ticker'].tolist()
        
        elif universe == 'RUSSELL2000':
            # Russell 2000 - simplified (top liquid ones)
            # V produkci by se stahoval kompletní seznam
            return []  # Would need separate data source
        
        elif universe == 'ALL':
            sp500 = self.get_stock_universe('SP500')
            nasdaq = self.get_stock_universe('NASDAQ100')
            return list(set(sp500 + nasdaq))
        
        else:
            raise ValueError(f"Unknown universe: {universe}")
    
    def screen_universe(self, 
                       universe: str = 'SP500',
                       min_volume: float = 1e6,
                       min_price: float = 5.0) -> pd.DataFrame:
        """
        Proveď kompletní screening celého universe
        
        Returns:
            DataFrame s top opportunities
        """
        tickers = self.get_stock_universe(universe)
        logger.info(f"Screening {len(tickers)} stocks...")
        
        results = []
        
        for ticker in tickers:
            try:
                result = self._score_ticker(ticker)
                
                # Filtruj podle volume a ceny
                if (result['avg_volume'] >= min_volume and 
                    result['price'] >= min_price):
                    results.append(result)
                    
            except Exception as e:
                logger.warning(f"Error screening {ticker}: {str(e)}")
                continue
        
        # Konvertuj na DataFrame a seřaď
        df = pd.DataFrame(results)
        df = df.sort_values('total_score', ascending=False)
        
        logger.info(f"Screening complete. Found {len(df)} valid opportunities")
        return df
    
    def _score_ticker(self, ticker: str) -> Dict:
        """
        Spočítej skóre pro jeden ticker napříč všemi strategiemi
        """
        total_score = 0
        strategy_scores = {}
        
        # Získej základní data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='3mo')
        
        if hist.empty:
            raise ValueError(f"No data for {ticker}")
        
        # Spočítej skóre pro každou strategii
        for strategy_name, strategy, weight in self.strategies:
            try:
                score = strategy.calculate_score(ticker, stock, hist)
                weighted_score = score * weight
                total_score += weighted_score
                
                strategy_scores[strategy_name] = {
                    'score': score,
                    'weighted': weighted_score
                }
            except Exception as e:
                logger.warning(f"{ticker} - {strategy_name} failed: {str(e)}")
                strategy_scores[strategy_name] = {'score': 0, 'weighted': 0}
        
        # Confluence bonus: pokud 3+ strategie mají score > 60
        high_scoring = sum(1 for s in strategy_scores.values() 
                          if s['score'] > 60)
        if high_scoring >= 3:
            total_score *= 1.25  # 25% bonus
        
        # Základní metriky
        current_price = hist['Close'].iloc[-1]
        avg_volume = hist['Volume'].mean()
        
        return {
            'ticker': ticker,
            'total_score': total_score,
            'confluence_count': high_scoring,
            'strategy_scores': strategy_scores,
            'price': current_price,
            'avg_volume': avg_volume,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown'),
            'timestamp': datetime.now()
        }
    
    def generate_recommendations(self, 
                                top_n: int = 20,
                                min_score: float = 70) -> pd.DataFrame:
        """
        Vygeneruj top N doporučení s risk management parametry
        """
        # Screen universe
        results = self.screen_universe()
        
        # Filter by minimum score
        results = results[results['total_score'] >= min_score]
        
        # Take top N
        top_results = results.head(top_n)
        
        # Add risk management parameters
        recommendations = []
        for _, row in top_results.iterrows():
            ticker = row['ticker']
            rec = self._add_risk_params(ticker, row)
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def _add_risk_params(self, ticker: str, row: pd.Series) -> Dict:
        """
        Přidej risk management parametry (entry, stop, targets)
        """
        # Získej ATR pro stop loss calculation
        hist = yf.Ticker(ticker).history(period='1mo')
        atr = self._calculate_atr(hist)
        
        current_price = row['price']
        
        return {
            'ticker': ticker,
            'score': row['total_score'],
            'confluence': row['confluence_count'],
            'entry_price': current_price,
            'stop_loss': current_price - (2 * atr),
            'target_1': current_price + (3 * atr),  # 1.5:1 RR
            'target_2': current_price + (5 * atr),  # 2.5:1 RR
            'position_size': self._calculate_position_size(atr),
            'risk_amount': self._calculate_position_size(atr) * 2 * atr,
            'confidence': 'HIGH' if row['confluence_count'] >= 3 else 'MEDIUM',
            'strategy_breakdown': row['strategy_scores']
        }
    
    def _calculate_atr(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = hist['High']
        low = hist['Low']
        close = hist['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _calculate_position_size(self, atr: float, 
                                 account_size: float = 100000,
                                 risk_pct: float = 0.02) -> int:
        """
        Calculate position size based on risk
        
        Args:
            atr: Average True Range
            account_size: Total account size
            risk_pct: % of account to risk per trade (default 2%)
        """
        risk_amount = account_size * risk_pct
        stop_distance = 2 * atr  # 2 ATR stop
        
        position_size = int(risk_amount / stop_distance)
        return max(1, position_size)


def main():
    """Main screening workflow"""
    logger.info("Starting weekly screening workflow...")
    
    # Initialize engine
    engine = QuantScreeningEngine()
    
    # Generate recommendations
    recommendations = engine.generate_recommendations(top_n=20, min_score=70)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'data/results/recommendations_{timestamp}.csv'
    recommendations.to_csv(output_file, index=False)
    
    logger.info(f"✅ Screening complete! Results saved to {output_file}")
    logger.info(f"Found {len(recommendations)} high-quality opportunities")
    
    # Print top 5
    print("\n🎯 TOP 5 RECOMMENDATIONS:\n")
    for _, row in recommendations.head(5).iterrows():
        print(f"{row['ticker']}: Score {row['score']:.1f} | "
              f"Entry ${row['entry_price']:.2f} | "
              f"Stop ${row['stop_loss']:.2f} | "
              f"Target ${row['target_1']:.2f}")
    
    return recommendations


if __name__ == "__main__":
    main()
