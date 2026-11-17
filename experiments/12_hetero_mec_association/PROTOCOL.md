# Stage 12 — Heterogeneous MEC compute and association RL

Frozen protocol (2026-11-16). Navigation experts are the five frozen Stage-11
models; no navigation model is trained, fine-tuned or modified. Only the hosted
task compute and the node compute capacity change.

## 1. Expert compute (user-frozen, modelling assumptions)

| Model | GFLOPs / decision |
|---|---:|
| Small | 5 |
| Medium | 15 |
| Transformer-S | 40 |
| Transformer-M | 100 |
| Large | 200 |

These are **not** runtime measurements. The Stage-11 measured model FLOPs
(4.5M / 55.1M / 556.0M) are kept only as a historical reference.

## 2. UAV encoding

`T_encode = measured encoder FLOPs / 1.5 TFLOPS`.

`measure_encoder_flops.py` reports **8.174 GFLOPs** (frozen ResNet-50 layer4,
224×224, 2×MACs convention) → **5.450 ms** at 1.5 TFLOPS. The RTX 5090 wall
clock for the same forward is 3.10 ms (recorded only as a cross-check).

## 3. Communication (unchanged)

Payload 49×2048×16 = 1,605,632 bit/decision; 2 GHz, 5 MHz, 0.1 W, 500 m
coverage, LOS/NLOS probabilistic path loss + Shannon; downlink ignored.
`T_E2E = T_encode + T_uplink + T_inference`.

## 4. Deployments: 10 nodes per map, 24 maps

Model placement:

- 2-model groups: `G1,S; G2,TM; G3,S; G4,TM; ...` (5 + 5).
- 5-model equal groups: `S, M, TS, TM, L, S, M, TS, TM, L` (2 each).

Homogeneous compute: 10 TFLOPS on every node.

Heterogeneous tiers (pre-registered; no earlier 5/10/20/30 table exists in the
repository):

| Node | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | G9 | G10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 2-model model | S | TM | S | TM | S | TM | S | TM | S | TM |
| 2-model TFLOPS | 5 | 10 | 20 | 30 | 10 | 5 | 30 | 20 | 10 | 10 |
| 5-model model | S | M | TS | TM | L | S | M | TS | TM | L |
| 5-model TFLOPS | 5 | 10 | 20 | 30 | 10 | 30 | 20 | 10 | 5 | 20 |

2-model balance: both models see `{5,10,10,20,30}` (mean 15). 5-model balance:
each model has one low (5/10) and one high (20/30) node.

## 5. Four experiment groups

| Group | Models | Compute |
|---|---|---|
| `two_homo` | 5 Small + 5 Transformer-M | 10 TFLOPS |
| `two_hetero` | 5 Small + 5 Transformer-M | 5/10/20/30 |
| `five_homo` | 2 each of five experts | 10 TFLOPS |
| `five_hetero` | 2 each of five experts | 5/10/20/30 |

## 6. Association baselines

Every run is a **fresh real closed loop**; association changes the acting expert
and therefore the trajectory.

| Strategy | Rule | Information |
|---|---|---|
| random | uniform over the 500 m coverage set; per-episode RNG `[seed, task_index]` | availability |
| max_rate | maximum Shannon uplink rate | distance/channel |
| max_quality | maximum quality prior (training-side Val Macro-F1), E2E tie-break | hosted model type |
| oracle | privileged teacher-action agreement, lowest E2E among agreeing nodes | **true target + obstacle map + teacher rule + all expert outputs** |

**Oracle upper bound.** At each step the teacher action `a*` is computed from
the true state (Stage-03B direct teacher, with its acquisition/cruise deadband).
The original Stage-12 rule (`oracle_e2e_tiebreak_v1`, kept for history) picks an
exact-agreement node with the lowest instantaneous E2E; if none agrees, it picks
the highest `P(a*)` (E2E tie-break). It bounds the action correctness achievable
by *association* given the frozen expert pool and the teacher preference. It is
not a strict SR upper bound (a non-agreeing action can still succeed) and not an
E2E lower bound. It is non-deployable and lives in `oracle_teacher.py`, imported
only under `--strategy oracle`.

### 6.1 Oracle v2 — action agreement only (2026-11-17 correction)

The corrected rule (`oracle_action_agreement_v2`) removes every latency,
distance and quality term from association:

```
agreement_i = int(argmax(P_i) == a*)
score_i     = (agreement_i, P_i(a*))
select argmax score_i; final ties by fixed node order G1 < G2 < ... < G10
```

No E2E, communication latency, inference latency, distance, Macro-F1/SR or RL
signal is used for node selection. `selected_teacher_action_probability` and
`selected_agreement` are only audit fields; the latency columns in the step
journal are computed after selection and are never fed back.

Per-step audit fields: `teacher_action`, `selected_node`, `selected_model`,
`selected_action`, `selected_teacher_action_probability`, `selected_agreement`,
`available_candidate_count`, `exact_agreement_candidate_count`,
`selected_node_compute_tflops`, `selected_node_comm_ms`,
`selected_node_inf_ms`, `selected_node_e2e_ms`, `tie_break_used`.

v2 results are isolated under
`results/oracle_action_agreement_v2/<group>/`; the v1 results remain under
`results/baselines/<group>/oracle/seed42/`. The two versions must not be
mixed as repeated measurements of the same rule.

## 7. RL

- State: 4 + 10×19 = 194-D. Re-scaled: compute `/30e12`, inference `/25ms`,
  E2E `/100ms` (clipped 1.0), distance `/500m`, rate `/200e6`.
- Action: 10 nodes, coverage mask.
- Reward: Stage-2r scheme D — `u_t = 1 − e2e_t/100`, `r_t = u_t/400`, terminal
  correction `(1−T/400)·J`; SR/OSR/SPL never enter the reward.
- Algorithms: standard DQN (194→256→256→10) and PPO (194→256→256→{10,1}),
  hyperparameters unchanged from Stage-2r.
- Training pool: 11A Train 18,521 → Dev-60 + Train-4,000 (seed 20261022).
  Val-Seen is never used for training, priors or checkpoint selection.
- Two seeds per group/algorithm: 42 and 43.
- `best.pt` = lowest Dev-60 mean E2E (tie: higher SR).
- Order: all PPO runs first (homogeneous groups, then heterogeneous), each
  evaluated on Val-Seen as soon as it finishes; then DQN in the same order.

## 8. Evaluation

Val-Seen all 2,498 tasks; `step_cap=200`; STOP radius 20 m; deterministic
greedy association. Reported per policy: SR, OSR, SPL, NE, E2E mean/P50/P95/P99,
MEC inference, communication, UAV encoding, expert/node step share, failure
taxonomy. Paired per-task comparisons (McNemar for SR; paired bootstrap for
E2E) are produced by `summarize.py` when both sides exist.

## 9. Known scope limits

- No inter-UAV MEC queueing and no handover latency (same as Stage-11).
- Localisation matching (layer-2 encoder) is UAV-side control overhead and is
  not part of the modelled E2E, matching the Stage-11 accounting.
- The five expert GFLOPs are modelling assumptions, not measurements.
- The 5/10/20/30 node table is pre-registered here; it was not present in any
  earlier report or session artifact.
