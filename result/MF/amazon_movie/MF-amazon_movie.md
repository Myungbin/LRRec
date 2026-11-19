# MF on Movie (`amazon_movie`)

Tuning `tune2026`: seed 2026, split of the files (about 4:3:3), AlphaFree evaluation protocol, selection on valid Recall@20; metrics at the best validation epoch.

- Search: `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
- Best: `reg_weight` 0.01 (best epoch 734)
- Log: [`MF-amazon_movie.log`](MF-amazon_movie.log) (5 configurations, MMRec format)

| | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Best | 0.0633 | 0.0982 | 0.0748 | 0.0870 | 0.0627 | 0.0967 | 0.0740 | 0.0859 |

## All configurations (best valid first)

| # | Stage | Configuration | Best epoch | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | 0 | `reg_weight` 0.01 | 734 | 0.0633 | 0.0982 | 0.0748 | 0.0870 | 0.0627 | 0.0967 | 0.0740 | 0.0859 |
| 2 | 0 | `reg_weight` 0.001 | 989 | 0.0544 | 0.0819 | 0.0644 | 0.0740 | 0.0540 | 0.0804 | 0.0640 | 0.0731 |
| 3 | 0 | `reg_weight` 0.0001 | 684 | 0.0391 | 0.0597 | 0.0466 | 0.0535 | 0.0386 | 0.0578 | 0.0461 | 0.0526 |
| 4 | 0 | `reg_weight` 1e-05 | 209 | 0.0345 | 0.0535 | 0.0419 | 0.0485 | 0.0337 | 0.0521 | 0.0411 | 0.0475 |
| 5 | 0 | `reg_weight` 1e-06 | 104 | 0.0344 | 0.0532 | 0.0419 | 0.0484 | 0.0336 | 0.0520 | 0.0408 | 0.0472 |
