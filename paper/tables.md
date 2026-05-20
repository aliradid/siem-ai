# Tables — OpenSIEM-AI paper

Source: `paper/Article-SIEM-v32.docx`. Numbers in these tables are regenerated from the JSON files in `results/` by the scripts in `scripts/`.

## Table 1. Integrated ML SIEM studies (2021–2026).

| Study (year) | Detection | Datasets | Automation | Reported Benefits |
|---|---|---|---|---|
| Page et al. (2024) [9] | Random Forest + XGBoost | SIEM event logs | Email alerts | Reduced false-alarm rate (paper-reported) |
| Sheeraz et al. (2024) [4] | Signature engine (Hyperscan) | Synthetic HTTP traffic | — | Correlation throughput improvement |
| Goldstein & Uchida (2016) [6] | Unsupervised (IF, LOF, AE) | 10 multivariate corpora | — | Survey of 19 algorithms |
| Kolosnjaji et al. (2016) [10] | CNN + LSTM | Malware system-call traces | — | Detection on system-call sequences |
| Liu et al. (2023) [11] | Transformer (graph) | Dynamic graphs | — | Anomaly detection in evolving graphs |
| Guo et al. (2021) [12] | BERT (LogBERT) | HDFS, BGL log corpora | — | Pre-trained transformer for log anomaly |
| This Work (2026) | XGBoost + Bi-LSTM | NSL-KDD, UNSW-NB15, GUIDE, HDFS | iptables (real, root) + TheHive 5 case (real, REST) + Azure AD disable (simulated stub), 500 cycles | NSL F1 0.7748 (canonical KDDTest+); UNSW F1 0.8625 (canonical Moustafa split); HDFS F1 0.9945 (block-level split); GUIDE F1 0.6797 (canonical Microsoft Train→Test, leaky cols removed); ablation: ML adds 0 ms p95 |

## Table 2. Dataset summary.

| Dataset | Raw events | Labelled records | Classes/Labels |
|---|---|---|---|
| GUIDE | 13M | 9.46M | Benign, Suspicious, Malicious |
| NSL-KDD | 148 517 flows | 125 973 train / 22 544 test (KDDTest+) | Normal vs. attack (4 attack families: DoS, Probe, R2L, U2R) |
| HDFS | 11M | 575K | Normal, Anomalous |
| UNSW-NB15 | 257 673 flows | 82 332 train / 175 341 test | Normal vs. Attack (9 attack families) |

## Table 3. Overall GUIDE test set metrics (XGBoost).

| Metric | Value |
|---|---|
| Accuracy | 0.7214 |
| Precision | 0.7321 |
| Recall | 0.6692 |

## Table 4. Per-class classification report for XGBoost on the GUIDE dataset (canonical Microsoft GUIDE_Train → GUIDE_Test split, leaky action/verdict columns removed).

| Class / Metric | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (BenignPositive) | 0.678 | 0.899 | 0.773 | 1 752 940 |
| 1 (FalsePositive) | 0.720 | 0.404 | 0.518 | 902 698 |
| 2 (TruePositive) | 0.798 | 0.704 | 0.748 | 1 492 354 |
| Accuracy | — | — | 0.721 | 4 147 992 |
| Macro avg | 0.732 | 0.669 | 0.6797 | 4 147 992 |
| Weighted avg | 0.730 | 0.721 | 0.709 | 4 147 992 |

## Table 5. Summary metrics – XGBoost on NSL-KDD (KDDTrain+ → KDDTest+).

| Metric | Value |
|---|---|
| Accuracy | 0.7756 |
| Precision | 0.8145 |
| Recall | 0.7993 |
| F1score | 0.7748 |
| ROC AUC | 0.9694 |

## Table 6. NSL-KDD baseline comparison on the same KDDTest+ split and 15-feature pipeline.

| Classifier | Accuracy | Macro F1 | Macro Recall | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | 0.7336 | 0.7327 | 0.7562 | 0.8773 |
| Random Forest (200 trees) | 0.7619 | 0.7606 | 0.7871 | 0.9551 |
| LightGBM (default hyperparameters) | 0.7569 | 0.7553 | 0.7829 | 0.9636 |
| CatBoost (default hyperparameters) | 0.7657 | 0.7645 | 0.7905 | 0.9677 |
| XGBoost (tuned, this work) | 0.7756 | 0.7748 | 0.7993 | 0.9694 |

## Table 7. UNSW-NB15 binary classification metrics on the canonical Moustafa & Slay testing split (n = 82 332), same XGBoost configuration.

| Metric | Value |
|---|---|
| Accuracy | 0.8685 |
| Precision (macro) | 0.8922 |
| Recall (macro) | 0.8560 |
| F1-score (macro) | 0.8625 |
| ROC AUC | 0.9814 |

## Table 8. HDFS baselines vs Bi-LSTM on the same 70/10/20 block-level (session) split (n test = 115 013), no BlockId straddles partitions.

| Classifier | Accuracy | Macro F1 | ROC AUC |
|---|---|---|---|
| Isolation Forest (unsupervised, normals only) | 0.9460 | 0.5341 | 0.7184 |
| Logistic Regression (bag-of-templates) | 0.9989 | 0.9901 | 0.9992 |
| Random Forest (bag-of-templates, 200 trees) | 0.9999 | 0.9993 | 1.0000 |
| Bi-LSTM (sequence, this work) | 0.9994 | 0.9945 | — |

## Table 9. Streaming and SOAR validation on the open-source stack (Docker Compose, Apple Silicon, single-node ES + Logstash + ML inference). Streaming latencies are means across 5 × 15 s repetitions at the 100 events/s working point (cf. Table 11); SOAR latencies are measured across 500 end-to-end SOAR cycles.

| Metric | Value |
|---|---|
| Target throughput | 100 events/s |
| Achieved throughput | 100.0 events/s |
| Indexed events / sent | 7 424 / 7 504 (99 %) |
| p50 ingest → ML verdict | 210 ms |
| p95 ingest → ML verdict | 619 ms |
| p99 ingest → ML verdict | 885 ms |
| p50 ingest → containment | 946 ms |
| p95 ingest → containment | 1 161 ms |
| p99 ingest → containment | 1 185 ms |
| iptables DROP latency (real netfilter) | mean 9.7 ms |
| TheHive case creation (live instance) | mean 46.8 ms |
| Azure AD account disable (simulated stub) | mean 50.5 ms |
| Successful SOAR cycles | 500 / 500 |

## Table 10. Ablation study isolating ML inference cost from pipeline cost (100 events/s, 15 s).

| Variant | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|
| A. ML inference only (direct /predict, n = 1 000) | 2.7 | 3.5 | 4.2 |
| B. Pipeline without ML (Logstash → ES) | 204 | 606 | 904 |
| C. Full pipeline (Logstash → ML → ES) | 209 | 568 | 814 |

## Table 11. Pipeline scaling on the single-node Docker stack (15 s × 5 repetitions per rate; values are mean ± standard deviation across reps).

| Target rate | Achieved rate | p50 (ms) | p95 (ms) | p99 (ms) | Indexed / sent |
|---|---|---|---|---|---|
| 100 eps | 100 eps | 210±2 | 619±57 | 885±139 | 7 424 / 7 504 (99 %) |
| 250 eps | 250 eps | 2 212±280 | 5 962±296 | 6 660±133 | 18 755 / 18 755 (100 %) |
| 500 eps | 490 eps | 1 770±49 | 3 772±367 | 4 465±667 | 37 000 / 37 000 (100 %) |
| 1 000 eps | 958 eps | 1 205±44 | 2 237±74 | 2 640±87 | not verified (query cap) |

