"""
Course: MCTA 4362 Machine Learning: Mini Project
Module: Parallel GUI Split-Screen Simulation Evaluation
Objective: Real-time side-by-side benchmarking of a classical PD loop
           against the optimized ML policy under active wind turbulence.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
from stable_baselines3 import PPO

def main():
    # Initialize Pygame setup for custom split-screen display
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 20, bold=True)
    
    # Window dimensions: 1200x450 canvas allows side-by-side visual analysis
    window_width = 1200
    window_height = 450
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Parallel Intelligent Control Systems Benchmark Sandbox")
    clock = pygame.time.Clock()

    # --- ENVIRONMENT DISTURBANCE INJECTION ---
    # Active Wind: Enabled with a magnitude of 12.0 to evaluate system robustness.
    # Render Mode: Capture pixel matrices using rgb_array for Pygame UI injection.
    print("Spawning evaluation tracks with cross-wind turbulence active...")
    env_pid = gym.make("LunarLander-v3", enable_wind=True, wind_power=12.0, render_mode="rgb_array")
    env_ml = gym.make("LunarLander-v3", enable_wind=True, wind_power=12.0, render_mode="rgb_array")
    
    # Load the optimized reinforcement learning model file
    model = PPO.load("ppo_lunar_lander")
    
    # Reset tracking environments
    obs_pid, _ = env_pid.reset()
    obs_ml, _ = env_ml.reset()
    
    frame_pid = env_pid.render()
    frame_ml = env_ml.render()
    
    # Metrics collection arrays for report plot export
    pid_alt, pid_err = [], []
    ml_alt, ml_err = [], []
    pid_rew, ml_rew = 0, 0
    pid_steps, ml_steps = 0, 0
    
    pid_active = True
    ml_active = True
    running = True

    print("Parallel execution loop active. Press 'Q' inside the GUI window to terminate...")

    while running:
        # Listen for keyboard interrupts
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # --- 1. CLASSICAL PD CONTROLLER PIPELINE (LEFT PANEL) ---
        if pid_active:
            # STATE-SPACE MAPPING:
            # pos_x: Horizontal displacement error vector
            # pos_y: Vertical altitude tracking parameter
            # vel_x / vel_y: Linear derivative tracking parameters
            # angle / angular_vel: Rotational coordinates for posture balancing
            pos_x, pos_y, vel_x, vel_y, angle, angular_vel = obs_pid[:6]
            
            pid_alt.append(pos_y)
            # EVALUATION METRIC: Calculate current Euclidean tracking error distance
            pid_err.append(np.sqrt(pos_x**2 + pos_y**2))
            
            # Mathematical PD Control Formulas
            target_vel_y = -0.3 * (pos_y + 0.1)
            vel_y_error = vel_y - target_vel_y
            target_angle = -0.5 * pos_x - 0.5 * vel_x
            angle_error = angle - target_angle
            
            control_up = -1.5 * vel_y_error
            control_side = -1.0 * angle_error - 0.5 * angular_vel
            
            # Map analytical continuous calculations to discrete actuator switches
            if control_up > 0.5 and abs(control_side) < 0.3:
                action_pid = 2  # Fire Main Thruster
            elif control_side > 0.1:
                action_pid = 3  # Fire Right Orientation Engine
            elif control_side < -0.1:
                action_pid = 1  # Fire Left Orientation Engine
            else:
                action_pid = 0  # Atmospheric Drift Mode
                
            obs_pid, reward, terminated, truncated, _ = env_pid.step(action_pid)
            pid_rew += reward
            pid_steps += 1
            frame_pid = env_pid.render()
            
            if terminated or truncated:
                pid_active = False

        # --- 2. INTELLIGENT ML POLICY PIPELINE (RIGHT PANEL) ---
        if ml_active:
            pos_x, pos_y = obs_ml[0], obs_ml[1]
            ml_alt.append(pos_y)
            ml_err.append(np.sqrt(pos_x**2 + pos_y**2))
            
            # Predict action values deterministically using the trained network policy
            action_ml, _ = model.predict(obs_ml, deterministic=True)
            obs_ml, reward, terminated, truncated, _ = env_ml.step(action_ml)
            ml_rew += reward
            ml_steps += 1
            frame_ml = env_ml.render()
            
            if terminated or truncated:
                ml_active = False

        # --- 3. DUAL GRAPHICAL USER INTERFACE RENDERING ---
        screen.fill((30, 30, 35))
        
        # Display dynamic tracking status tags
        pid_status = "RUNNING" if pid_active else "FINISHED"
        ml_status = "RUNNING" if ml_active else "FINISHED"
        
        text_pid = font.render(f"CLASSICAL PD PROFILE ({pid_status})", True, (255, 140, 0))
        text_ml = font.render(f"INTELLIGENT ML PPO AGENT ({ml_status})", True, (0, 191, 255))
        
        screen.blit(text_pid, (20, 15))
        screen.blit(text_ml, (620, 15))
        
        # Process visual environment matrices into interactive Pygame surfaces
        surf_pid = pygame.surfarray.make_surface(np.transpose(frame_pid, (1, 0, 2)))
        surf_ml = pygame.surfarray.make_surface(np.transpose(frame_ml, (1, 0, 2)))
        
        # Project frames side by side below the text headers
        screen.blit(surf_pid, (0, 50))
        screen.blit(surf_ml, (600, 50))
        
        pygame.display.flip()
        clock.tick(60) # Lock simulation runtime framework at 60 FPS
        
        if not pid_active and not ml_active:
            break

    # Keep display open post flight run for final analysis inspection
    print("\nSimulations completed. Focus the GUI window and press 'Q' to export comparative data charts.")
    waiting_for_input = True
    while waiting_for_input:
        screen.fill((30, 30, 35), rect=(0, 0, 1200, 50))
        prompt_text = font.render("EVALUATION COMPLETE: Press 'Q' to exit and export analytical graphs.", True, (255, 255, 255))
        screen.blit(prompt_text, (320, 15))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting_for_input = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                waiting_for_input = False

    env_pid.close()
    env_ml.close()
    pygame.quit()

    # --- 4. EXPORT COMPREHENSIVE OVERLAPPED METRIC VISUALIZATIONS ---
    print("\nGenerating overlapped dual axis evaluation charts...")
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    color_pid = "tab:orange"
    color_ml = "tab:blue"
    
    # Configure Primary Y Axis for Descent Curve Profiles
    line1, = ax1.plot(pid_alt, color=color_pid, linestyle="-", linewidth=2.5, label=f"PD Altitude ({pid_steps} steps)")
    line2, = ax1.plot(ml_alt, color=color_ml, linestyle="-", linewidth=2.5, label=f"PPO Altitude ({ml_steps} steps)")
    ax1.set_xlabel("Simulation Frame Steps", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Normalized Altitude (y)", color="black", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Configure Secondary Y Axis for Target Tracking Deviation Error Profiles
    ax2 = ax1.twinx()
    line3, = ax2.plot(pid_err, color=color_pid, linestyle="--", linewidth=1.5, alpha=0.6, label=f"PD Tracking Error (Score: {pid_rew:.1f})")
    line4, = ax2.plot(ml_err, color=color_ml, linestyle="--", linewidth=1.5, alpha=0.6, label=f"PPO Tracking Error (Score: {ml_rew:.1f})")
    ax2.set_ylabel("Euclidean Target Tracking Error Distance", color="black", fontsize=11, fontweight="bold")
    
    # Generate consolidated legend array mapping boundaries cleanly
    all_lines = [line1, line2, line3, line4]
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc="upper right", frameon=True, shadow=True)
    
    plt.title("Overlapped Controller Performance Profiles under Atmospheric Turbulence", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("controller_comparison_metrics.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()