---
language:
- en
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 32107
    num_examples: 416
  - name: test
    num_bytes: 7937.0
    num_examples: 104
  download_size: 20481
  dataset_size: 40044.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
