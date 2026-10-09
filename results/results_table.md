**Overall**

| Method | Episodes | Success | Mean deviation (cm) | Final error (cm) | Completion time (s) |
|---|--:|--:|--:|--:|--:|
| scripted | 7 | 7/7 (100%) | 0.61 | 0.39 | 5.4 |
| residual_dev5 | 7 | 7/7 (100%) | 0.63 | 0.55 | 4.1 |
| residual_s0_1M | 7 | 6/7 (86%) | 0.88 | 1.03 | 3.8 |
| ppo | 7 | 0/7 (0%) | 5.70 | 24.57 | n/a |

**By category**

| Method | Category | Episodes | Success | Mean deviation (cm) | Final error (cm) | Completion time (s) |
|---|---|--:|--:|--:|--:|--:|
| scripted | curve | 2 | 2/2 (100%) | 0.80 | 0.43 | 4.6 |
| scripted | multi | 1 | 1/1 (100%) | 0.57 | 0.38 | 4.0 |
| scripted | straight | 2 | 2/2 (100%) | 0.36 | 0.34 | 4.4 |
| scripted | turn | 2 | 2/2 (100%) | 0.68 | 0.39 | 8.0 |
| residual_dev5 | curve | 2 | 2/2 (100%) | 0.52 | 0.45 | 4.6 |
| residual_dev5 | multi | 1 | 1/1 (100%) | 0.56 | 0.41 | 4.4 |
| residual_dev5 | straight | 2 | 2/2 (100%) | 0.34 | 0.31 | 2.7 |
| residual_dev5 | turn | 2 | 2/2 (100%) | 1.08 | 0.97 | 4.9 |
| residual_s0_1M | curve | 2 | 2/2 (100%) | 0.95 | 0.70 | 3.5 |
| residual_s0_1M | multi | 1 | 1/1 (100%) | 0.90 | 0.44 | 3.2 |
| residual_s0_1M | straight | 2 | 2/2 (100%) | 0.42 | 0.30 | 2.7 |
| residual_s0_1M | turn | 2 | 1/2 (50%) | 1.25 | 2.38 | 5.6 |
| ppo | curve | 2 | 0/2 (0%) | 7.31 | 25.39 | n/a |
| ppo | multi | 1 | 0/1 (0%) | 0.67 | 8.86 | n/a |
| ppo | straight | 2 | 0/2 (0%) | 3.87 | 24.90 | n/a |
| ppo | turn | 2 | 0/2 (0%) | 8.44 | 31.29 | n/a |

Completion time is the mean over successful episodes. Source: `results/scripted_test.csv`, `results/residual_dev5_test.csv`, `results/residual_s0_1M_test.csv`, `results/ppo_test.csv`.
