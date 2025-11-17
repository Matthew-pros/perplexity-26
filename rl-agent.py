"""
Reinforcement Learning Agent pro optimalizaci trading strategie
Používá PPO (Proximal Policy Optimization) pro kontinuální učení
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch
from loguru import logger
from typing import Dict, Tuple


class TradingEnv(gym.Env):
    """
    Custom Trading Environment pro RL agent
    
    State Space:
        - Strategy scores (5 hodnot, 0-100)
        - Portfolio metrics (return, sharpe, drawdown)
        - Market regime (VIX, trend)
        - Previous week performance
    
    Action Space:
        - Strategy weights (5 hodnot, sum=1)
        - Position sizes (continuous, 0-1)
    """
    
    def __init__(self, 
                 historical_data: pd.DataFrame,
                 initial_balance: float = 100000):
        super(TradingEnv, self).__init__()
        
        self.historical_data = historical_data
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # State space: 15 features
        # [5 strategy scores, 5 performance metrics, 5 market indicators]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(15,), dtype=np.float32
        )
        
        # Action space: 5 strategy weights (normalized to sum=1)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(5,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {}
        self.trades_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Sestaví aktuální state pro agenta
        """
        if self.current_step >= len(self.historical_data):
            return np.zeros(15)
        
        row = self.historical_data.iloc[self.current_step]
        
        # Strategy scores (5 hodnot)
        strategy_scores = [
            row.get('earnings_score', 50),
            row.get('institutional_score', 50),
            row.get('technical_score', 50),
            row.get('short_squeeze_score', 50),
            row.get('sentiment_score', 50)
        ]
        
        # Portfolio metrics (5 hodnot)
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance * 100
        sharpe_ratio = self._calculate_sharpe()
        max_drawdown = self._calculate_max_drawdown()
        win_rate = self._calculate_win_rate()
        avg_trade = self._calculate_avg_trade()
        
        portfolio_metrics = [
            portfolio_return,
            sharpe_ratio * 10,  # Scale to 0-100
            abs(max_drawdown) * 100,
            win_rate * 100,
            avg_trade
        ]
        
        # Market indicators (5 hodnot)
        vix = row.get('vix', 15)
        trend = row.get('spy_trend', 0)  # -1 to 1
        volume_ratio = row.get('volume_ratio', 1)
        
        market_indicators = [
            vix,
            (trend + 1) * 50,  # Convert -1,1 to 0,100
            volume_ratio * 50,
            row.get('market_breadth', 50),
            row.get('sector_rotation', 50)
        ]
        
        observation = np.array(
            strategy_scores + portfolio_metrics + market_indicators,
            dtype=np.float32
        )
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step
        
        Args:
            action: Numpy array of 5 strategy weights
        
        Returns:
            observation, reward, done, info
        """
        # Normalize action to sum=1 (strategy weights)
        action = action / (action.sum() + 1e-8)
        
        # Apply strategy weights to screening results
        recommendations = self._get_weekly_recommendations()
        portfolio_return = self._execute_trades(recommendations, action)
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return)
        
        # Move to next week
        self.current_step += 1
        done = self.current_step >= len(self.historical_data)
        
        observation = self._get_observation()
        info = {
            'portfolio_value': self.portfolio_value,
            'weekly_return': portfolio_return,
            'action': action
        }
        
        return observation, reward, done, info
    
    def _get_weekly_recommendations(self) -> pd.DataFrame:
        """Simuluj weekly screening results"""
        # V reálném použití by se loadovaly skutečné screening results
        row = self.historical_data.iloc[self.current_step]
        return row.get('recommendations', pd.DataFrame())
    
    def _execute_trades(self, 
                       recommendations: pd.DataFrame,
                       strategy_weights: np.ndarray) -> float:
        """
        Simuluj execute trades based on RL agent decisions
        
        Returns:
            Weekly return %
        """
        if recommendations.empty:
            return 0.0
        
        # Apply strategy weights to score každé opportunity
        weighted_scores = np.zeros(len(recommendations))
        
        for i, (strategy_name, weight) in enumerate([
            ('earnings_score', strategy_weights[0]),
            ('institutional_score', strategy_weights[1]),
            ('technical_score', strategy_weights[2]),
            ('short_squeeze_score', strategy_weights[3]),
            ('sentiment_score', strategy_weights[4])
        ]):
            if strategy_name in recommendations.columns:
                weighted_scores += recommendations[strategy_name].values * weight
        
        # Select top trades based on weighted scores
        recommendations['weighted_score'] = weighted_scores
        top_trades = recommendations.nlargest(10, 'weighted_score')
        
        # Simuluj weekly return (simplified)
        weekly_return = top_trades['weekly_return'].mean()
        
        # Update portfolio
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + weekly_return)
        
        # Record trade
        self.trades_history.append({
            'step': self.current_step,
            'return': weekly_return,
            'portfolio_value': self.portfolio_value,
            'strategy_weights': strategy_weights.copy()
        })
        
        return weekly_return
    
    def _calculate_reward(self, weekly_return: float) -> float:
        """
        Reward function pro RL agent
        
        Prioritizuje:
        - Pozitivní returns
        - High Sharpe ratio
        - Low drawdown
        - Consistency
        """
        # Base reward from return
        reward = weekly_return * 100
        
        # Bonus for positive return
        if weekly_return > 0:
            reward *= 1.5
        
        # Penalty for drawdown
        max_dd = self._calculate_max_drawdown()
        reward -= abs(max_dd) * 200
        
        # Bonus for Sharpe ratio
        sharpe = self._calculate_sharpe()
        reward += sharpe * 10
        
        # Penalty for volatility
        if len(self.trades_history) > 4:
            recent_returns = [t['return'] for t in self.trades_history[-4:]]
            volatility = np.std(recent_returns)
            reward -= volatility * 50
        
        return reward
    
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
    """
    Wrapper pro RL agent s training a inference
    """
    
    def __init__(self, 
                 algorithm: str = 'PPO',
                 model_path: str = None):
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        
    def train(self,
             historical_data: pd.DataFrame,
             total_timesteps: int = 10000,
             save_path: str = 'data/models/rl_agent.zip'):
        """
        Train RL agent on historical data
        """
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
                verbose=1,
                tensorboard_log='./tensorboard/'
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
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=f"{self.algorithm}_trading"
        )
        
        # Save model
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
        """
        Predict action (strategy weights) for current state
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call train() or load() first.")
        
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Normalize to sum=1
        action = action / (action.sum() + 1e-8)
        
        return action


def main():
    """Main training workflow"""
    logger.info("Starting RL agent training...")
    
    # Load historical data
    # V praxi by se loadovala skutečná historická data z backtestů
    historical_data = pd.DataFrame({
        'week': range(52),  # 1 rok dat
        'earnings_score': np.random.uniform(40, 90, 52),
        'institutional_score': np.random.uniform(40, 90, 52),
        'technical_score': np.random.uniform(40, 90, 52),
        'short_squeeze_score': np.random.uniform(30, 80, 52),
        'sentiment_score': np.random.uniform(35, 85, 52),
        'vix': np.random.uniform(12, 30, 52),
        'spy_trend': np.random.uniform(-0.5, 0.5, 52)
    })
    
    # Initialize agent
    agent = RLTradingAgent(algorithm='PPO')
    
    # Train
    agent.train(
        historical_data=historical_data,
        total_timesteps=10000,
        save_path='data/models/rl_agent_ppo.zip'
    )
    
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()
