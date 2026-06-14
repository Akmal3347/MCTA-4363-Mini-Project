"""
Course: MCTA 4362 Machine Learning: Mini Project
Module: Reinforcement Learning Policy Training
Objective: Optimizing a stochastic Proximal Policy Optimization (PPO) agent
           to achieve stable spacecraft descent dynamics.
"""

import gymnasium as gym
from stable_baselines3 import PPO

def train_agent():
    # Initialize environment without rendering to maximize CPU/GPU training throughput
    env = gym.make("LunarLander-v3")
    
    print("Initializing Deep Reinforcement Learning Agent...")
    
    # --- HYPERPARAMETER AND ARCHITECTURE CONFIGURATION ---
    # Policy Network: Multi-Layer Perceptron (MlpPolicy) for coordinate-based inputs.
    # Learning Rate: Set to 0.0003 (Adam optimizer standard) for stable convergence.
    # Tensorboard Log: Enabled to monitor reward tracking and policy loss curves.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./ppo_lunar_tensorboard/"
    )
    
    # --- TRAINING PARAMETERS ---
    # Total Timesteps: 100,000 steps balances execution time and policy success rates.
    TIMESTEPS = 100000
    print(f"Starting policy training loop for {TIMESTEPS} timesteps...")
    
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save the trained policy weights as a compressed zip file
    model_path = "ppo_lunar_lander"
    model.save(model_path)
    print(f"Policy network optimization complete. Model saved as '{model_path}.zip'")
    
    env.close()

if __name__ == "__main__":
    train_agent()