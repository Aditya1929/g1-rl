# g1-rl: HITTER-style table tennis on the G1, with SONIC tokens as actions

PPO trains a whole-body striking policy as in HITTER (arXiv 2508.21043): the
environment samples racket targets, and there is no ball. Every 20 ms the policy
outputs a **residual on a SONIC token** from a reference swing library (GRAIL-style,
arXiv 2606.05160). A frozen SONIC decoder inside `env.step` turns the token into PD
targets, the same thing the deploy controller does on the robot.

```
offline: motion-kit forehand clips -> IK face correction -> SONIC encoder -> swing library (z_ref, q_ref, racket path)
train:   sampled (racket pos/vel/normal at impact, t_strike, base target)
            -> policy -> dz -> z = quantize(z_ref + 0.1 clip(dz)) -> frozen SONIC decoder -> PD -> G1 (Isaac Lab)
deploy:  stereo vision tracker -> strike planner -> policy -> token --ZMQ--> gear_sonic_deploy -> G1
```

## Layout
| Path | What |
|---|---|
| `source/g1_sonic_tt/g1_sonic_tt/sonic/` | decoder (batched torch MLP + 10-frame history), FSQ, swing library, SONIC conventions |
| `.../assets/g1_racket.py` | G1 + racket with SONIC's exact actuator model |
| `.../tasks/token_hitter/` | the Isaac Lab task: `env_cfg.py`, `mdp/` (commands, token action, obs, rewards, curriculum), PPO cfg |
| `.../ball/model.py`, `.../planner/strike.py` | analytic ball (drag, bounce, racket impact) and HITTER planner (Eqs. 1–6) |
| `scripts/` | data builders, replay check, BC, train, play/export, ball-in-the-loop eval |
| `deploy/` | vision → planner → policy → token bridge, ZMQ wire format, frame conversion |
| `tests/` | simulator-free tests |
| `data/` | generated artifacts (library, decoder weights, racket URDF) |

Tasks: `G1-SonicTT-Token-v0` (residual, default), `-Play-v0`, `-Replay-v0` (library tokens, no policy);
`G1-SonicTT-TokenAbs-v0` / `-TokenAbs-Play-v0` (previous absolute-token + BC + token-matching recipe, for comparison).

## Setup (data generation, no simulator)
```bash
~/sonic-dataset/.venv/bin/python scripts/build_library.py --face-normal 0.85 0.18 0.5   # IK open-face fix + re-encode
~/sonic-dataset/.venv/bin/python scripts/convert_decoder.py   # ONNX -> npz, parity vs onnxruntime
python3 scripts/make_racket_urdf.py                           # SONIC main.urdf, Dex3 removed, bracket racket (0.20 kg)
~/miniconda3/envs/isaaclab/bin/python tests/test_core.py && ~/miniconda3/envs/isaaclab/bin/python tests/test_deploy.py
```
External inputs (override with env vars, see `paths.py`): `~/GR00T-WholeBodyControl`,
`~/sonic-dataset` (the `g1-forehand-sonic-tokens-v1` dataset), `~/table-tennis-motion-kit`.

## Running Isaac Lab scripts
```bash
source ~/miniconda3/bin/activate isaaclab
export PYTHONUNBUFFERED=1            # app.close() hard-exits; unbuffered keeps the printed results
I=~/IsaacLab/isaaclab.sh
$I -p scripts/replay_reference.py --num_envs 64 --headless        # 1. closed-loop sanity check
$I -p scripts/train.py --headless --num_envs 2048                  # 2. PPO (actor output zeroed = starts as replay)
$I -p scripts/view.py --num_envs 8                                 #    live GUI, hot-reloads checkpoints
tensorboard --logdir logs/rsl_rl/g1_sonic_token_hitter             #    curves
$I -p scripts/play.py --checkpoint logs/rsl_rl/g1_sonic_token_hitter/<run>/model_N.pt   # 3. export
$I -p scripts/eval_ball.py --checkpoint <model.pt> --num_envs 64 --headless   # hits / returns
$I -p scripts/eval_ball.py --replay --num_envs 64 --headless                   # library baseline
```
GPU: RTX 5070 Laptop, 8 GB. Use 2048 envs or fewer.

Absolute-token recipe instead: `bc_pretrain.py` (defaults to `-TokenAbs-v0`), then
`train.py --task G1-SonicTT-TokenAbs-v0 --init_checkpoint <bc>/model_0.pt`.

### Replay check (library tokens, no policy; 64 envs × 10 s, 256 impacts, 0 falls in all)
| Robot model | Clips | Racket error at impact | Face error |
|---|---|---|---|
| Dex3 hands + 0.16 kg disk | raw | 5.5 cm (2026-11-19, 16 envs) | – |
| bracket (hands removed, 0.20 kg) | raw | 7.7 cm | 11.3° |
| Dex3 hands + 0.16 kg disk | face-corrected | 7.8 cm | 14.0° |
| **bracket (current)** | **face-corrected** | **12.9 cm** | **16.1°** |

The current error is mostly a consistent offset, +10.7 cm up (+3.9 fwd, −3.4 lateral),
with 4.6 cm scatter. Likely cause: SONIC was trained with ~0.7 kg Dex3 hands, so its
targets over-lift the lighter bracket arm. PPO's residual has to remove the offset.
If racket error stalls near 10 cm, raise `residual_scale` (λ) to 0.2.

## Design notes
- **FSQ.** SONIC tokens are 2 × 32 dims with 32 levels each: k/16 for k ∈ [−16, 15], i.e. [−1, 0.9375] (probed from the encoder ONNX). An earlier version wrongly clamped to ±0.5.
- **Action (residual, default).** `z = quantize(z_ref + λ·clip(a, −1, 1))`, λ = 0.2 (run 1 used 0.1), quantized after the sum. The actor's output layer starts at zero, so the env starts as library replay and no BC is needed. Exploration std is 0.3, since a one-step change needs |a| > 0.16. Entropy coefficient is 0.001; with 0.003 the noise std doubled once the task saturated.
- **Training targets (`target_source="planner"`).** At startup, 200k analytic serves aimed at the forehand are run through the strike planner. The ~71k strikes that land inside the library window become a pool. Each swing draws one: strike point, racket velocity (~1.34 m/s, forward-up), normal = velocity direction, and the clip start set so impact lands at the planned time. Run 1 used targets around the clips' own contacts (`target_source="library"`): a flat sideways sweep with a near-vertical face. It hit those targets to 1.1 cm but returned 0 balls, since the planner needs a ~30° open face moving forward and up. `absolute`: `z = quantize(tanh a)`. `replay`: `z = z_ref`.
- **Library.** The raw clips hit with the face at about 45°. `build_library.py` re-solves the right arm per frame (contact point kept on the clip's path, face normal blended toward `--face-normal` around impact), smooths, then re-encodes with SONIC's encoder. With the open face (0.85, 0.18, 0.5), about the planner's mean normal: face error at impact 0.3°, contact point within 0.8 cm at impact, and up to 10.7 cm during follow-through (0.16–0.28 s after impact).
- **Imitation (pose space).**
  - `pose_joint` (1.0 → 0.5): joints vs the reference at frame+1, ±2-frame slack, right arm/waist ×2.
  - `pose_joint_vel` (0.2 → 0.1).
  - `racket_path` (0.5 → 0.2): racket point vs the clip's racket path.
  - `decoder_match` (0.3 → 0.1).
  - The right arm and racket path are down-weighted to 20% within ±10 frames (±0.2 s) of impact.
- **Prior.** `residual_l2` −0.5 (mean dz²) and `residual_rate` −0.5. Tokens are never compared directly: with a residual, token distance is just λ‖dz‖.
- **Goal.** HITTER's sparse racket position/velocity/face terms at impact with SMASH adaptive σ, plus a success bonus. The face term tracks a commanded normal: in training, the library normal plus ~0.1 rad tilt; from the planner, the racket velocity direction.
- **Actor obs (208).** HITTER commands plus the normal target, proprio, z_ref and the previous dz. **Critic** adds reference tokens (t, t+5, t+10), reference joints, racket errors, base linear velocity and time left.
- **Robot/racket.** SONIC's main.urdf with both Dex3 hands removed, plus the bracket-v4 mount from vision_robotic_table_tennis:
  - bracket box 0.035 kg
  - handle 0.025 kg
  - blade 0.14 kg at +0.2365 m, normal +Y; its collider is a box because the asset turns cylinders into capsules
  
  Racket mass is randomized by −0.04 to +0.06 kg. `make_racket_urdf.py --mount dex3` restores the previous model.
- **Timing.** Actions at clip frame `f` act over the next 20 ms, so impact rewards fire when `f+1 == hit_frame`.

## Known gaps / TODO
- **Backhand.** There is no backhand token library yet. The command term samples forehand only, and `swing_type` is fixed to forehand.
- **Command range vs. planner.** Library contacts cover y ∈ [−0.53, −0.33] m and z ∈ [0.08, 0.23] m (pelvis frame). Planner strikes on simulated serves span y ∈ [−0.96, −0.19] m. Either aim the serves in `sample_serves` or widen `target_margin` and base offsets once the policy can step.
- **Base movement.** The library pelvis never moves, so base-target offsets default to 0.
- **Imitation vs. goal at impact.** When a sampled target is away from the clip's contact point, pose tracking pulls the arm toward the clip and the racket rewards pull toward the target. Mitigations: the impact window (right arm at 20%, ±0.2 s) and the curriculum decay. If impact errors plateau, widen the window, lower the floor, or decay faster. Also note that training velocity targets keep the clips' sideways sweep (~50° from forward) while the face target is now forward. The planner's velocities point mostly forward, so the training distribution of velocity targets does not yet match deployment.
- **Hand removal.** SONIC was trained with Dex3 hands. Without them, replay rides about 11 cm high (see above). This is only right if the real robot uses the bracket with no hands.
- **Not yet run end to end:** BC + absolute training past a smoke test, eval_ball, the live bridge. The residual task trains (2048 envs, ~39k steps/s); play/export and the bridge's obs/token path have been checked against an exported policy.
- **Deploy bridge.** The robot pose in the table frame is static (from CLI arguments) and is not yet read from vision. Verify the tracker-frame axis signs on the real table. The bridge is dry-run by default; validate sim-to-sim in MuJoCo with the real SONIC controller before `--live`.
- **Ball and racket parameters** (drag, restitution, racket mass) are nominal. Fit them on recorded bounces.
- **Command noise.** Training uses exact commands. If real swings jitter, add SMASH Eq. 14 noise to the actor's commands.

## Third-party
`data/sonic_decoder_release.npz` is converted from NVIDIA's SONIC (GR00T-WholeBodyControl) weights.
Licensed by NVIDIA Corporation under the NVIDIA Open Model License; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses/).
