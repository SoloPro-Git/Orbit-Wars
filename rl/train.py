"""
Orbit Wars PPO Training Script.

Main training loop:
1. Load BC model for warm start (optional)
2. Create environment and opponent pool
3. PPO training with periodic evaluation
4. Save checkpoints
"""
import os
import sys
import torch
import argparse
from pathlib import Path
import swanlab

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import OrbitWarsEnv
from agent import PPOAgent
from ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Orbit Wars PPO Agent')
    
    # Game mode
    parser.add_argument('--mode', type=str, default='1v1', choices=['1v1', '4p'],
                        help='Game mode: 1v1 or 4-player')
    
    # Training parameters
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total training timesteps')
    parser.add_argument('--rollout-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--eval-freq', type=int, default=10,
                        help='Evaluate every N iterations')
    parser.add_argument('--eval-games', type=int, default=50,
                        help='Number of evaluation games')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N iterations')
    
    # Model parameters
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--bc-checkpoint', type=str, 
                        default='checkpoints/bc_1v1_v2_best.pt',
                        help='BC model checkpoint for warm start')
    parser.add_argument('--no-warm-start', action='store_true',
                        help='Skip BC warm start')
    
    # Opponent pool
    parser.add_argument('--opponents', nargs='+', 
                        default=['starter', 'random', 'aggressive', 'replay'],
                        help='Opponent pool for training')
    parser.add_argument('--replay-dir', type=str, default='datasets',
                        help='Directory containing replay JSON files')
    parser.add_argument('--replay-prob', type=float, default=0.7,
                        help='Probability of using replay opponent vs fallback')
    
    # Device
    parser.add_argument('--device', type=str, default='cpu',
                        help='Training device (cpu/cuda)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--exp-name', type=str, default='ppo_v1',
                        help='Experiment name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SwanLab
    swanlab.init(
        project="orbit-wars-ppo",
        experiment_name=args.exp_name,
        config=vars(args),
        logdir=str(output_dir / "swanlab_logs")
    )
    
    print("=" * 60)
    print("Orbit Wars PPO Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Opponents: {args.opponents}")
    print(f"Output: {output_dir}")
    print(f"SwanLab: https://swanlab.cn/@lance/orbit-wars-ppo")
    print("=" * 60)
    
    # Set num_players based on mode
    num_players = 2 if args.mode == "1v1" else 4
    
    # Create environment
    env = OrbitWarsEnv(num_players=num_players)
    print(f"✓ Environment created ({num_players} players)")
    
    # Create agent
    agent = PPOAgent(
        embed_dim=args.embed_dim,
        max_planets=40,
        max_fleets=100,
        num_players=num_players,
    )
    print(f"✓ Agent created (embed_dim={args.embed_dim}, num_players={num_players})")
    
    # Load BC checkpoint for warm start
    if not args.no_warm_start and os.path.exists(args.bc_checkpoint):
        try:
            num_loaded = agent.load_bc_encoder(args.bc_checkpoint)
            print(f"✓ BC warm start: loaded {num_loaded} encoder parameters")
        except Exception as e:
            print(f"⚠ BC warm start failed: {e}")
            print("  Training from scratch")
    else:
        if not args.no_warm_start:
            print(f"⚠ BC checkpoint not found: {args.bc_checkpoint}")
        print("  Training from scratch")
    
    # Load replay opponents if 'replay' is in opponent list
    replay_opponents = {}
    if 'replay' in args.opponents:
        try:
            from replay_opponent import load_replay_opponents
            replay_opponents = load_replay_opponents(
                replay_dir=args.replay_dir,
                mode=args.mode,
                replay_prob=args.replay_prob
            )
            print(f"✓ Loaded {len(replay_opponents)} replay opponents from {args.replay_dir}")
            for name in list(replay_opponents.keys())[:5]:
                print(f"  - {name}")
        except Exception as e:
            print(f"⚠ Failed to load replay opponents: {e}")
            print("  Continuing without replay opponents")
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        agent=agent,
        opponent_pool=args.opponents,
        device=device,
        replay_opponents=replay_opponents,
    )
    print(f"✓ Trainer created")
    print()
    
    # Initial evaluation
    print("Initial evaluation...")
    for opp in args.opponents:
        stats = trainer.evaluate(opponent=opp, num_games=10)
        print(f"  vs {opp:12s}: win_rate={stats['win_rate']:.1%}, "
              f"avg_reward={stats['avg_reward']:.3f}")
    print()
    
    # Training loop
    num_iterations = args.total_timesteps // args.rollout_steps
    total_steps = 0
    best_win_rate = 0.0
    
    print(f"Starting training ({num_iterations} iterations)...")
    print("-" * 60)
    
    for iteration in range(1, num_iterations + 1):
        # Collect rollouts
        steps = trainer.collect_rollouts(num_steps=args.rollout_steps)
        total_steps += steps
        
        # Train
        stats = trainer.train_step()
        
        # Logging
        if iteration % 1 == 0:
            print(f"Iter {iteration:4d} | Steps {total_steps:8,} | "
                  f"Loss {stats['loss']:.4f} | "
                  f"Policy {stats['policy_loss']:.4f} | "
                  f"Value {stats['value_loss']:.4f} | "
                  f"Entropy {stats['entropy']:.4f} | "
                  f"Reward {stats['mean_reward']:.3f}")
            
            # Log to SwanLab
            swanlab.log({
                "iteration": iteration,
                "total_steps": total_steps,
                "loss": stats['loss'],
                "policy_loss": stats['policy_loss'],
                "value_loss": stats['value_loss'],
                "entropy": stats['entropy'],
                "mean_reward": stats['mean_reward'],
            })
        
        # Evaluation
        if iteration % args.eval_freq == 0:
            print()
            print(f"Evaluation at iteration {iteration}...")
            
            avg_win_rate = 0
            for opp in args.opponents:
                eval_stats = trainer.evaluate(opponent=opp, num_games=args.eval_games)
                print(f"  vs {opp:12s}: win_rate={eval_stats['win_rate']:.1%} "
                      f"({eval_stats['wins']}/{eval_stats['games']}), "
                      f"avg_reward={eval_stats['avg_reward']:.3f}")
                
                # Log to SwanLab
                swanlab.log({
                    f"eval/win_rate_vs_{opp}": eval_stats['win_rate'],
                    f"eval/avg_reward_vs_{opp}": eval_stats['avg_reward'],
                    f"eval/wins_vs_{opp}": eval_stats['wins'],
                })
                
                avg_win_rate += eval_stats['win_rate']
            
            avg_win_rate /= len(args.opponents)
            
            # Log average win rate
            swanlab.log({
                "eval/avg_win_rate": avg_win_rate,
                "iteration": iteration,
            })
            
            # Save best model
            if avg_win_rate > best_win_rate:
                best_win_rate = avg_win_rate
                best_path = output_dir / 'best.pt'
                torch.save({
                    'iteration': iteration,
                    'total_steps': total_steps,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'win_rate': avg_win_rate,
                    'args': vars(args),
                }, best_path)
                print(f"  ✓ New best model saved (win_rate={avg_win_rate:.1%})")
            print()
        
        # Save checkpoint
        if iteration % args.save_freq == 0:
            ckpt_path = output_dir / f'checkpoint_{iteration}.pt'
            torch.save({
                'iteration': iteration,
                'total_steps': total_steps,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'stats': stats,
                'args': vars(args),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")
    
    # Final evaluation
    print()
    print("=" * 60)
    print("Final evaluation...")
    print("=" * 60)
    
    for opp in args.opponents:
        stats = trainer.evaluate(opponent=opp, num_games=args.eval_games)
        print(f"vs {opp:12s}: win_rate={stats['win_rate']:.1%} "
              f"({stats['wins']}/{stats['games']})")
    
    # Save final model
    final_path = output_dir / 'final.pt'
    torch.save({
        'iteration': num_iterations,
        'total_steps': total_steps,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}")
    print(f"✓ Best model (win_rate={best_win_rate:.1%}): {output_dir / 'best.pt'}")
    
    print()
    print("Training complete!")


if __name__ == '__main__':
    main()
