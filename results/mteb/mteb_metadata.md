---
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- mteb
model-index:
- name: Basinfy/BasinRAG
  results:
  - task:
      type: Retrieval
    dataset:
      type: mteb/scifact
      name: MTEB SciFact
      config: default
      split: test
      revision: d56462d0e63a25450459c4f213e49ffdb866f7f9
    metrics:
    - type: map_at_1
      value: 54.944
    - type: map_at_3
      value: 60.255
    - type: map_at_5
      value: 61.31
    - type: map_at_10
      value: 61.881
    - type: map_at_20
      value: 62.036
    - type: map_at_100
      value: 62.304
    - type: map_at_1000
      value: 62.356
    - type: mrr_at_1
      value: 57.0
    - type: mrr_at_3
      value: 61.7222
    - type: mrr_at_5
      value: 62.5722
    - type: mrr_at_10
      value: 62.9439
    - type: mrr_at_20
      value: 63.0346
    - type: mrr_at_100
      value: 63.2485
    - type: mrr_at_1000
      value: 63.2971
    - type: ndcg_at_1
      value: 57.0
    - type: ndcg_at_3
      value: 62.205
    - type: ndcg_at_5
      value: 63.764
    - type: ndcg_at_10
      value: 64.967
    - type: ndcg_at_20
      value: 65.5
    - type: ndcg_at_100
      value: 67.649
    - type: ndcg_at_1000
      value: 69.118
    - type: precision_at_1
      value: 57.0
    - type: precision_at_3
      value: 23.778
    - type: precision_at_5
      value: 15.267
    - type: precision_at_10
      value: 8.2
    - type: precision_at_20
      value: 4.267
    - type: precision_at_100
      value: 0.993
    - type: precision_at_1000
      value: 0.112
    - type: recall_at_1
      value: 54.944
    - type: recall_at_3
      value: 65.694
    - type: recall_at_5
      value: 69.539
    - type: recall_at_10
      value: 73.078
    - type: recall_at_20
      value: 75.056
    - type: recall_at_100
      value: 87.033
    - type: recall_at_1000
      value: 98.667
    - type: accuracy
      value: 54.944
    - type: hit_rate_at_1
      value: 57.0
    - type: hit_rate_at_10
      value: 74.667
    - type: hit_rate_at_100
      value: 87.333
    - type: hit_rate_at_1000
      value: 98.667
    - type: hit_rate_at_20
      value: 76.333
    - type: hit_rate_at_3
      value: 67.667
    - type: hit_rate_at_5
      value: 71.333
    - type: main_score
      value: 64.967
---
