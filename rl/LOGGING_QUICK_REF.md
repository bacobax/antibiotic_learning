# Quick Reference: Understanding Your Training Logs

## What You'll See in Console

```
[INFO] UPDATE    0/100 | Reward:    50.34 (±12.45) | Episodes:   5 | Actor Loss: 0.5234 | Critic Loss: 0.3821
[INFO] UPDATE   10/100 | Reward:    65.12 (±18.92) | Episodes:   6 | Actor Loss: 0.3421 | Critic Loss: 0.2156
[INFO] UPDATE   20/100 | Reward:    75.89 (±22.10) | Episodes:   7 | Actor Loss: 0.2890 | Critic Loss: 0.1654
```

## Is My Training Working? Checklist

**✅ GOOD SIGNS** (Training is working):
- Reward value increasing each update
- Actor Loss **decreasing** trend
- Critic Loss **decreasing** trend  
- 2-5 episodes per update (shows environment is interactive)
- No ERROR or WARNING messages

**⚠️ WARNING SIGNS** (May need adjustment):
```
[WARNING] High clipping fraction at update 15: 0.85
→ Solution: Reduce learning rate (--lr 1e-4)

[WARNING] No episodes completed at update 10!
→ Solution: Increase rollout steps (--steps-per-rollout 4096)

[DEBUG]   Entropy: 0.0012 | Clip Frac: 0.312 | Grad Norm: 18.234
→ Entropy too low? Add exploration
→ Grad Norm too high? Reduce learning rate
```

**❌ BAD SIGNS** (Stop training, fix issue):
```
[ERROR] NaN detected in actor loss at update 25!
→ Solution: Reduce learning rate significantly (try --lr 1e-5)

Reward becoming negative/stagnant while losses don't decrease
→ Environment issue or reward signal problem
```

## Key Metrics Explained

| Metric | Meaning | Target Behavior |
|--------|---------|-----------------|
| **Reward** | How well agent performs | ↑ Increasing |
| **±** | Variability | ↓ Decreasing (more stable) |
| **Episodes** | Training data collected | 2-10 per update is typical |
| **Actor Loss** | Policy network performance | ↓ Decreasing |
| **Critic Loss** | Value function performance | ↓ Decreasing |
| **Entropy** | Exploration level | Starts high, slowly ↓ |
| **Clip Frac** | PPO clipping usage | 0.1-0.5 is good, >0.5 is bad |
| **Grad Norm** | Training stability | 0.1-2.0 is normal |

## File Locations

```
checkpoints/
├── training.log          ← All detailed logs (read with tail -f)
├── training_log.json     ← Metrics for analysis
├── checkpoint_50.pt      ← Model weights at update 50
└── config.json          ← Your run configuration
```

## Common Commands

```bash
# Watch training in real-time
tail -f checkpoints/training.log

# Check for problems
grep -E "ERROR|WARNING" checkpoints/training.log

# See training summary
tail -50 checkpoints/training.log

# Check final performance
tail -5 checkpoints/training.log
```

## When to Stop Training

- **Good end**: Reward plateaued at high value, losses stable
- **Stop early if**: Losses become NaN/Inf, or gradient norm exploding
- **Safe to interrupt**: Hit desired reward, losses stable, then save final checkpoint

## Performance Interpretation

```
Update 0: Reward 20.5 → Learning from scratch
Update 10: Reward 35.2 → Initial progress (75% improvement)
Update 20: Reward 42.1 → Slowing improvement (19% improvement) - EXPECTED
Update 50: Reward 48.3 → Near convergence (1% improvement) - CONVERGED
Update 50-100: Reward ~48 ± 5 → Fine-tuning phase - good time to stop
```

**Interpretation**:
- Large % improvement early = ✅ Normal
- Small % improvement later = ✅ Convergence (expected)
- Negative improvement = ⚠️ Check training parameters
- Huge fluctuations = ⚠️ High variance, may need more episodes/lower LR
