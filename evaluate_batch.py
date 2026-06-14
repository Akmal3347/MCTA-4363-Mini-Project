import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


MODEL_PATH = "ppo_lunar_lander"
EPISODES_PER_WIND = 10
WIND_LEVELS = [0, 8, 12]


def pd_controller(obs):
    pos_x, pos_y, vel_x, vel_y, angle, angular_vel = obs[:6]

    target_vel_y = -0.3 * (pos_y + 0.1)
    vel_y_error = vel_y - target_vel_y

    target_angle = -0.5 * pos_x - 0.5 * vel_x
    angle_error = angle - target_angle

    control_up = -1.5 * vel_y_error
    control_side = -1.0 * angle_error - 0.5 * angular_vel

    if control_up > 0.5 and abs(control_side) < 0.3:
        return 2  # Main engine
    elif control_side > 0.1:
        return 3  # Right engine
    elif control_side < -0.1:
        return 1  # Left engine
    else:
        return 0  # Do nothing


def run_episode(controller_name, wind_power, model=None, seed=0):
    env = gym.make(
        "LunarLander-v3",
        enable_wind=(wind_power > 0),
        wind_power=float(wind_power)
    )

    obs, _ = env.reset(seed=seed)

    total_reward = 0
    tracking_errors = []
    steps = 0
    final_reward = 0

    terminated = False
    truncated = False

    while not terminated and not truncated:
        pos_x, pos_y = obs[0], obs[1]
        tracking_error = np.sqrt(pos_x**2 + pos_y**2)
        tracking_errors.append(tracking_error)

        if controller_name == "PD Controller":
            action = pd_controller(obs)
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        final_reward = reward
        steps += 1

    env.close()

    # Reward-based landing/crash approximation
    success = final_reward > 50 or total_reward > 150
    crash = final_reward < -50 or total_reward < -100

    return {
        "controller": controller_name,
        "wind_power": wind_power,
        "total_reward": total_reward,
        "steps": steps,
        "avg_tracking_error": float(np.mean(tracking_errors)),
        "success": int(success),
        "crash": int(crash)
    }


def main():
    print("Loading trained PPO model...")
    model = PPO.load(MODEL_PATH)

    all_results = []

    for wind in WIND_LEVELS:
        print(f"\nTesting wind power = {wind}")

        for episode in range(EPISODES_PER_WIND):
            seed = 1000 + wind * 100 + episode

            pd_result = run_episode(
                controller_name="PD Controller",
                wind_power=wind,
                seed=seed
            )

            ppo_result = run_episode(
                controller_name="PPO AI",
                wind_power=wind,
                model=model,
                seed=seed
            )

            all_results.append(pd_result)
            all_results.append(ppo_result)

            print(
                f"Episode {episode + 1}/{EPISODES_PER_WIND} done | "
                f"PD reward: {pd_result['total_reward']:.1f} | "
                f"PPO reward: {ppo_result['total_reward']:.1f}"
            )

    df = pd.DataFrame(all_results)
    df.to_csv("results_by_episode.csv", index=False)

    summary = df.groupby(["wind_power", "controller"]).agg(
        avg_reward=("total_reward", "mean"),
        avg_steps=("steps", "mean"),
        avg_tracking_error=("avg_tracking_error", "mean"),
        success_rate=("success", "mean"),
        crash_rate=("crash", "mean")
    ).reset_index()

    summary["success_rate"] = summary["success_rate"] * 100
    summary["crash_rate"] = summary["crash_rate"] * 100

    summary.to_csv("results_summary.csv", index=False)

    print("\n=== SUMMARY RESULTS ===")
    print(summary)

    # Plot 1: Success rate vs wind
    pivot_success = summary.pivot(
        index="wind_power",
        columns="controller",
        values="success_rate"
    )

    pivot_success.plot(kind="bar", figsize=(9, 5))
    plt.title("Landing Success Rate under Different Wind Disturbances")
    plt.xlabel("Wind Power")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig("success_rate_vs_wind.png", dpi=300)
    plt.show()

    # Plot 2: Average reward vs wind
    pivot_reward = summary.pivot(
        index="wind_power",
        columns="controller",
        values="avg_reward"
    )

    pivot_reward.plot(kind="bar", figsize=(9, 5))
    plt.title("Average Reward under Different Wind Disturbances")
    plt.xlabel("Wind Power")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig("avg_reward_vs_wind.png", dpi=300)
    plt.show()

    print("\nFiles saved:")
    print("1. results_by_episode.csv")
    print("2. results_summary.csv")
    print("3. success_rate_vs_wind.png")
    print("4. avg_reward_vs_wind.png")


if __name__ == "__main__":
    main()