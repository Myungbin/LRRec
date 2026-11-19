# MF on Book (`amazon_book_2014`)

Tuning `tune2026`: seed 2026, split of the files (about 4:3:3), AlphaFree evaluation protocol, selection on valid Recall@20; metrics at the best validation epoch.

- Search: `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
- Best: `reg_weight` 0.01 (best epoch 784)
- Log: [`MF-amazon_book_2014.log`](MF-amazon_book_2014.log) (5 configurations, MMRec format)

| | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Best | 0.0496 | 0.0787 | 0.0557 | 0.0647 | 0.0494 | 0.0775 | 0.0550 | 0.0637 |

## All configurations (best valid first)

| # | Stage | Configuration | Best epoch | Valid R@10 | R@20 | N@10 | N@20 | Test R@10 | R@20 | N@10 | N@20 |
|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **1** | 0 | `reg_weight` 0.01 | 784 | 0.0496 | 0.0787 | 0.0557 | 0.0647 | 0.0494 | 0.0775 | 0.0550 | 0.0637 |
| 2 | 0 | `reg_weight` 0.001 | 984 | 0.0488 | 0.0765 | 0.0557 | 0.0641 | 0.0483 | 0.0758 | 0.0546 | 0.0631 |
| 3 | 0 | `reg_weight` 0.0001 | 979 | 0.0396 | 0.0621 | 0.0456 | 0.0525 | 0.0391 | 0.0616 | 0.0446 | 0.0515 |
| 4 | 0 | `reg_weight` 1e-05 | 989 | 0.0307 | 0.0481 | 0.0363 | 0.0415 | 0.0302 | 0.0472 | 0.0357 | 0.0408 |
| 5 | 0 | `reg_weight` 1e-06 | 194 | 0.0248 | 0.0396 | 0.0311 | 0.0353 | 0.0245 | 0.0388 | 0.0306 | 0.0347 |
