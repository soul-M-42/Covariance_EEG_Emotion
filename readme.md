# mdJPT: multi-dataset Joint Pretrain Transformer for emotion decoder from EEG

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Brief project description (1-2 sentences).

## Features
- Core feature 1
- Core feature 2
- End-to-end pipeline via `run_pipeline.sh`

## Run
```bash
bash run_pipeline.sh
```

#### Note: To modify the run parameter, adjust cfgs_multi/config_multi.yaml.

to pretrain and test on your own dataset, you should add a new data config yaml file under cfg_multi/data folder, then assign it in cfgs_multi/config_multi.yaml, in defaults.data_0~4 (for pretrain) and data_val (for validation).
