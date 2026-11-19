# MF on Video (`amazon_video`)

Tuning `tune2026`: seed 2026, split of the files (about 4:3:3), AlphaFree evaluation protocol, selection on valid Recall@20; metrics at the best validation epoch.

- Search: `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
- Best: `reg_weight` 0.01 (best epoch 934)
- Log: [`MF-amazon_video.log`](MF-amazon_video.log) (5 configurations, MMRec format)

| | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Best | 0.0615 | 0.0946 | 0.0416 | 0.0522 | 0.0619 | 0.0939 | 0.0421 | 0.0524 |

## All configurations (best valid first)

| # | Stage | Configuration | Best epoch | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | 0 | `reg_weight` 0.01 | 934 | 0.0615 | 0.0946 | 0.0416 | 0.0522 | 0.0619 | 0.0939 | 0.0421 | 0.0524 |
| 2 | 0 | `reg_weight` 0.001 | 979 | 0.0531 | 0.0803 | 0.0360 | 0.0446 | 0.0527 | 0.0798 | 0.0358 | 0.0444 |
| 3 | 0 | `reg_weight` 1e-06 | 114 | 0.0115 | 0.0184 | 0.0078 | 0.0100 | 0.0113 | 0.0182 | 0.0078 | 0.0099 |
| 4 | 0 | `reg_weight` 1e-05 | 124 | 0.0113 | 0.0181 | 0.0075 | 0.0097 | 0.0110 | 0.0178 | 0.0075 | 0.0097 |
| 5 | 0 | `reg_weight` 0.0001 | 29 | 0.0107 | 0.0170 | 0.0071 | 0.0091 | 0.0102 | 0.0167 | 0.0068 | 0.0089 |
