# LunarLander PD Controller vs PPO AI

This project compares a classical PD controller with a Machine Learning-based PPO reinforcement learning controller using the LunarLander-v3 simulation environment.

## Project Objective

The objective is to evaluate whether a PPO AI controller can perform better than a traditional PD controller when landing a rocket under wind disturbance.

## System

- Environment: LunarLander-v3
- Classical controller: PD Controller
- Machine Learning controller: PPO Reinforcement Learning
- Policy: MlpPolicy
- Training timesteps: 100,000
- Disturbance: Wind power applied during evaluation

## Files

- `train_rl.py` - trains the PPO AI agent
- `evaluate_comparison.py` - runs the split-screen PD vs PPO GUI demo
- `evaluate_batch.py` - runs multiple evaluation tests under different wind levels
- `ppo_lunar_lander.zip` - trained PPO model
- `controller_comparison_metrics.png` - comparison graph from demo
- `avg_reward_vs_wind.png` - average reward graph
- `success_rate_vs_wind.png` - landing success rate graph
- `results_summary.csv` - summary of evaluation results

## How to Run

Create virtual environment:

```bash
python -m venv .venv
