# Baseline tuning results

Tuning `tune2026`: seed 2026, split of the files (about 4:3:3), AlphaFree evaluation protocol, selection on valid Recall@20; metrics at the best validation epoch.

Each `<MODEL>/<dataset>/` folder holds the merged log of all configurations (MMRec format), the model code of the best run and a table of every configuration. The paper column is Table 1 of the AlphaFree paper (test Recall@20 / NDCG@20).

Search spaces (original papers):

- **MF**: `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
- **LightGCN**: `n_layers` in {1, 2, 3, 4} x `reg_weight` in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
- **SimGCL**: (a) `lambda_cl` in {0.01, 0.05, 0.1, 0.2, 0.5, 1, 2}, `eps` 0.1, 2 layers -> (b) `eps` in {0, 0.01, 0.05, 0.1, 0.2, 0.5} -> (c) `n_layers` in {1, 2, 3}
- **XSimGCL**: (a) `lambda_cl` in {0.01, 0.05, 0.1, 0.2, 0.5, 1} x `eps` in {0, 0.01, 0.05, 0.1, 0.2, 0.5}, 2 layers, `layer_cl` 1 -> (b) `n_layers` in {1, ..., 5} x `layer_cl` in {1, ..., `n_layers`}

## Movie (`amazon_movie`)

| Model | Best configuration | Best epoch | Test R@10 | R@20 | N@10 | N@20 | Valid R@10 | R@20 | N@10 | N@20 | Paper R@20 / N@20 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [MF](MF/amazon_movie/MF-amazon_movie.md) | `reg_weight` 0.01 | 734 | 0.0627 | 0.0967 | 0.0740 | 0.0859 | 0.0633 | 0.0982 | 0.0748 | 0.0870 | 0.0580 / 0.0533 |
| [LightGCN](LightGCN/amazon_movie/LightGCN-amazon_movie.md) | `n_layers` 3, `reg_weight` 0.001 | 704 | 0.0647 | 0.0998 | 0.0771 | 0.0888 | 0.0653 | 0.1020 | 0.0780 | 0.0906 | 0.0860 / 0.0754 |
| [SimGCL](SimGCL/amazon_movie/SimGCL-amazon_movie.md) | `n_layers` 3, `lambda_cl` 0.05, `eps` 0.1 | 54 | 0.0697 | 0.1063 | 0.0862 | 0.0979 | 0.0710 | 0.1076 | 0.0875 | 0.0993 | – |
| [XSimGCL](XSimGCL/amazon_movie/XSimGCL-amazon_movie.md) | `n_layers` 5, `layer_cl` 1, `lambda_cl` 0.1, `eps` 0.1 | 69 | 0.0735 | 0.1104 | 0.0907 | 0.1023 | 0.0748 | 0.1121 | 0.0922 | 0.1039 | 0.0967 / 0.0866 |
| AlphaFree (paper) | | | | 0.1267 | | 0.1194 | | | | | 0.1267 / 0.1194 |

## Book (`amazon_book_2014`)

| Model | Best configuration | Best epoch | Test R@10 | R@20 | N@10 | N@20 | Valid R@10 | R@20 | N@10 | N@20 | Paper R@20 / N@20 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [MF](MF/amazon_book_2014/MF-amazon_book_2014.md) | `reg_weight` 0.01 | 784 | 0.0494 | 0.0775 | 0.0550 | 0.0637 | 0.0496 | 0.0787 | 0.0557 | 0.0647 | 0.0436 / 0.0387 |
| [LightGCN](LightGCN/amazon_book_2014/LightGCN-amazon_book_2014.md) | `n_layers` 4, `reg_weight` 0.0001 | 764 | 0.0540 | 0.0847 | 0.0621 | 0.0711 | 0.0549 | 0.0858 | 0.0633 | 0.0724 | 0.0712 / 0.0597 |
| [SimGCL](SimGCL/amazon_book_2014/SimGCL-amazon_book_2014.md) | `n_layers` 3, `lambda_cl` 0.05, `eps` 0.1 | 39 | 0.0543 | 0.0847 | 0.0631 | 0.0717 | 0.0550 | 0.0860 | 0.0644 | 0.0734 | – |
| [XSimGCL](XSimGCL/amazon_book_2014/XSimGCL-amazon_book_2014.md) | `n_layers` 5, `layer_cl` 1, `lambda_cl` 0.1, `eps` 0.05 | 149 | 0.0573 | 0.0884 | 0.0676 | 0.0763 | 0.0587 | 0.0902 | 0.0693 | 0.0781 | 0.0818 / 0.0690 |
| AlphaFree (paper) | | | | 0.1014 | | 0.0861 | | | | | 0.1014 / 0.0861 |

## Video (`amazon_video`)

| Model | Best configuration | Best epoch | Test R@10 | R@20 | N@10 | N@20 | Valid R@10 | R@20 | N@10 | N@20 | Paper R@20 / N@20 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| [MF](MF/amazon_video/MF-amazon_video.md) | `reg_weight` 0.01 | 934 | 0.0619 | 0.0939 | 0.0421 | 0.0524 | 0.0615 | 0.0946 | 0.0416 | 0.0522 | 0.0177 / 0.0095 |
| [LightGCN](LightGCN/amazon_video/LightGCN-amazon_video.md) | `n_layers` 2, `reg_weight` 0.01 | 964 | 0.0660 | 0.1022 | 0.0450 | 0.0565 | 0.0669 | 0.1033 | 0.0452 | 0.0567 | 0.0732 / 0.0409 |
| [SimGCL](SimGCL/amazon_video/SimGCL-amazon_video.md) | `n_layers` 3, `lambda_cl` 0.01, `eps` 0.01 | 24 | 0.0622 | 0.0951 | 0.0429 | 0.0533 | 0.0631 | 0.0961 | 0.0431 | 0.0535 | – |
| [XSimGCL](XSimGCL/amazon_video/XSimGCL-amazon_video.md) | `n_layers` 5, `layer_cl` 5, `lambda_cl` 0.01, `eps` 0.01 | 34 | 0.0690 | 0.1041 | 0.0474 | 0.0586 | 0.0695 | 0.1050 | 0.0473 | 0.0586 | 0.0897 / 0.0500 |
| AlphaFree (paper) | | | | 0.1111 | | 0.0615 | | | | | 0.1111 / 0.0615 |
