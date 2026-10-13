import argparse
import pathlib
import numpy as np
import mlflow
import torch
import gymnasium as gym

def load_champion_policy():
    model_uri = "models:/lapzero-policy@champion"
    print(f"Loading champion policy from {model_uri}...")
    local_path = mlflow.artifacts.download_artifacts(model_uri=model_uri)
    policy_path = pathlib.Path(local_path) / "model.pt"
    if not policy_path.exists():
        policy_path = pathlib.Path("models/ppo_v1.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = torch.load(policy_path, map_location=device)
    policy.eval()
    return policy, device

def main():
    parser = argparse.ArgumentParser(description="Evaluate champion policy noise-free.")
    parser.add_argument("--map", type=str, required=True, help="Path to track map yaml")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation laps")
    args = parser.parse_args()

    # Initialize environment directly via gym/lapzero wrapper if registered, 
    # or fallback to standard gym make if map is passed
    try:
        env = gym.make("lapzero/f1tenth-v0", map_path=args.map)
    except Exception:
        from lapzero.sim.env import LapZeroEnv
        env = LapZeroEnv(map_path=args.map)

    policy, device = load_champion_policy()

    successes = 0
    lap_times = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        truncated = False
        steps = 0
        total_reward = 0.0

        while not (done or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Handle policy output structure (actor action)
                output = policy(obs_tensor)
                action = output[0] if isinstance(output, tuple) else output
            
            action = action.cpu().numpy().squeeze(0)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            total_reward += reward

        crashed = info.get("collision", False) if isinstance(info, dict) else False
        if not crashed:
            successes += 1
            lap_time = steps * 0.04
            lap_times.append(lap_time)
            print(f"Episode {ep + 1}: SUCCESS | Time: {lap_time:.2f}s | Reward: {total_reward:.2f}")
        else:
            print(f"Episode {ep + 1}: CRASHED at step {steps} | Reward: {total_reward:.2f}")

    print("\n--- Evaluation Summary ---")
    print(f"Success Rate: {successes}/{args.episodes} ({successes / args.episodes * 100}%)")
    if lap_times:
        print(f"Mean Lap Time: {np.mean(lap_times):.2f}s (Min: {np.min(lap_times):.2f}s)")

if __name__ == "__main__":
    main()
