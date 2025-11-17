"""
Reinforcement Learning Agent - FIXED for Streamlit Cloud
Uses gymnasium instead of gym (updated API)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from loguru import logger
from typing import Dict, Tuple


class TradingEnv(gym.Env):
    """
    Custom Trading Environment for RL agent
    Compatible with gymnasium API
    """
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 initial_balance: float = 100000):
        super(TradingEnv, self).__init__()
        
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # State space: 15 features
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(15,), dtype=np.float32
        )
        
        # Action space: 5 strategy weights
        self.action_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment - gymnasium API"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {}
        self.trades_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state"""
        if self.current_step >= len(self.historical_data):
            return np.zeros(15, dtype=np.float32)
        
        row = self.historical_data.iloc[self.current_step]
        
        # Strategy scores (5)
        strategy_scores = [
            row.get('earnings_score', 50),
            row.get('institutional_score', 50),
            row.get('technical_score', 50),
            row.get('short_squeeze_score', 50),
            row.get('sentiment_score', 50)
        ]
        
        # Portfolio metrics (5)
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance * 100
        sharpe_ratio = self._calculate_sharpe()
        max_drawdown = self._calculate_max_drawdown()
        win_rate = self._calculate_win_rate()
        avg_trade = self._calculate_avg_trade()
        
        portfolio_metrics = [
            portfolio_return,
            sharpe_ratio * 10,
            abs(max_drawdown) * 100,
            win_rate * 100,
            avg_trade
        ]
        
        # Market indicators (5)
        market_indicators = [
            row.get('vix', 15),
            (row.get('spy_trend', 0) + 1) * 50,
            row.get('volume_ratio', 1) * 50,
            row.get('market_breadth', 50),
            row.get('sector_rotation', 50)
        ]
        
        observation = np.array(
            strategy_scores + portfolio_metrics + market_indicators,
            dtype=np.float32
        )
        
        return observation
    
    def step(self, action: np.ndarray):
        """Execute one time step - gymnasium API"""
        # Normalize action
        action = action / (action.sum() + 1e-8)
        
        # Execute trades
        recommendations = self._get_weekly_recommendations()
        portfolio_return = self._execute_trades(recommendations, action)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return)
        
        # Move forward
        self.current_step += 1
        terminated = self.current_step >= len(self.historical_data)
        truncated = False
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weekly_return': portfolio_return,
            'action': action.tolist()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_weekly_recommendations(self) -> pd.DataFrame:
        """Get simulated recommendations"""
        row = self.historical_data.iloc[self.current_step]
        return row.get('recommendations', pd.DataFrame())
    
    def _execute_trades(self, recommendations: pd.DataFrame, strategy_weights: np.ndarray) -> float:
        """Execute trades and return weekly return"""
        if recommendations.empty:
            return 0.0
        
        # Simplified: just return simulated weekly return
        weekly_return = np.random.normal(0.01, 0.05)  # 1% avg, 5% std
        
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + weekly_return)
        
        self.trades_history.append({
            'step': self.current_step,
            'return': weekly_return,
            'portfolio_value': self.portfolio_value,
            'strategy_weights': strategy_weights.copy()
        })
        
        return weekly_return
    
    def _calculate_reward(self, weekly_return: float) -> float:
        """Calculate reward"""
        reward = weekly_return * 100
        
        if weekly_return > 0:
            reward *= 1.5
        
        max_dd = self._calculate_max_drawdown()
        reward -= abs(max_dd) * 200
        
        sharpe = self._calculate_sharpe()
        reward += sharpe * 10
        
        if len(self.trades_history) > 4:
            recent_returns = [t['return'] for t in self.trades_history[-4:]]
            volatility = np.std(recent_returns)
            reward -= volatility * 50
        
        return float(reward)
    
    def _calculate_sharpe(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trades_history) < 2:
            return 0.0
        
        returns = [t['return'] for t in self.trades_history]
        return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(52)
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.trades_history) < 2:
            return 0.0
        
        portfolio_values = [t['portfolio_value'] for t in self.trades_history]
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (value - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if len(self.trades_history) == 0:
            return 0.5
        
        wins = sum(1 for t in self.trades_history if t['return'] > 0)
        return wins / len(self.trades_history)
    
    def _calculate_avg_trade(self) -> float:
        """Calculate average trade return"""
        if len(self.trades_history) == 0:
            return 0.0
        
        returns = [t['return'] for t in self.trades_history]
        return np.mean(returns) * 100


class RLTradingAgent:
    """RL Trading Agent wrapper"""
    
    def __init__(self, algorithm: str = 'PPO', model_path: str = None):
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        
    def train(self,
             historical_data: pd.DataFrame,
             total_timesteps: int = 10000,
             save_path: str = 'data/models/rl_agent.zip'):
        """Train RL agent"""
        logger.info(f"Training {self.algorithm} agent...")
        
        # Create environment
        env = DummyVecEnv([lambda: TradingEnv(historical_data)])
        
        # Initialize model
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                env,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                verbose=1
            )
        
        # Train
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        return self.model
    
    def load(self, model_path: str):
        """Load trained model"""
        logger.info(f"Loading model from {model_path}")
        
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path)
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        action, _ = self.model.predict(observation, deterministic=True)
        action = action / (action.sum() + 1e-8)
        
        return action


def main():
    """Test training"""
    logger.info("Starting RL agent training...")
    
    # Create dummy data
    historical_data = pd.DataFrame({
        'week': range(52),
        'earnings_score': np.random.uniform(40, 90, 52),
        'institutional_score': np.random.uniform(40, 90, 52),
        'technical_score': np.random.uniform(40, 90, 52),
        'short_squeeze_score': np.random.uniform(30, 80, 52),
        'sentiment_score': np.random.uniform(35, 85, 52),
        'vix': np.random.uniform(12, 30, 52),
        'spy_trend': np.random.uniform(-0.5, 0.5, 52)
    })
    
    # Train
    agent = RLTradingAgent(algorithm='PPO')
    agent.train(historical_data, total_timesteps=1000)
    
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
