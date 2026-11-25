# Run 4: `v2_wide_step` (from scratch, calibrated library v2, 99 clips incl. side-steps)

- `model_9750.pt`: final checkpoint (training stopped at iteration 9,857 of 10,000; last save 9,750).
- `run4_hits.mp4`: 20 s, one robot, training-matched serves (`scripts/eval_ball.py --video`).
- `heatmap_run4.png` / `heatmap_run3.png`: return rate over the hitting plane, full-width serves (bounce y -0.75..0.75).

Eval, training-matched serves (bounce y -0.8..0.0, strikes within 8 cm of a clip; 3,614 swings): 51.2% returned
(53.2% hit). Side-step swings 63%, no-step 60%. Swings starting within 10 cm of home 68%, displaced >10 cm ~47%.

Full plane vs run 3 (returns): centre (y -0.25..0) 81% vs 50%; wide right (-0.85..-0.65) 51% vs 38%;
very wide (< -0.85) 18% vs 1%; low (< 0.70 m) 29% vs 15%; core forehand 76% vs 89%.
