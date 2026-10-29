---
language:
- en
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 1589987
    num_examples: 25058
  - name: test
    num_bytes: 397546
    num_examples: 6265
  download_size: 1215139
  dataset_size: 1987533
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
