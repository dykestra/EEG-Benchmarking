# EEG Benchmarking

Implementation for IEEE MLSP 2025 Paper [Assessing the Capabilities of Large Brainwave Foundation Models](https://openreview.net/forum?id=Qwn2a1uIpx).

## Abstract
Over the last decade, deep learning models have been widely used for automatic feature extraction and classification in various Brain-Computer Interface (BCI) tasks. However, their performance and generalization capabilities are often not adequately assessed, as these models are frequently trained and tested under flawed setups and / or influenced by spurious correlations. Recently, these limitations have also been observed in the training and evaluation of Large Brainwave Foundation Models (LBMs). In this work, we employ causal reasoning and careful consideration for task-discriminative artifacts in various EEG datasets covering diverse BCI paradigms and propose a benchmarking protocol to properly evaluate the decoding performance and generalization capabilities of LBMs. Utilising a subject-independent cross-validation approach for each curated benchmark dataset, we showcase that LBMs achieve marginal performance gains over conventional deep learning baselines.

## Environment Setup

Install [PyTorch](https://pytorch.org/get-started/locally/).

Install other requirements:

```commandline
pip install -r requirements.txt
```

## Data Pre-processing

### Download Data

The following 5 EEG datasets are selected for benchmarking:
| Dataset | Paradigm | Number of Classes | Type(s) | Tasks |
|-|-|-|-|-|
| [High Gamma](https://github.com/robintibor/high-gamma-dataset) | Executed Movement | 4 | | `no_action`, `left_fist`, `right_fist`, `both_feet` |
| [OpenBMI-ERP](http://gigadb.org/dataset/100542) | ERP | 2 | | `target`, `nontarget` |
| [Pavlov 2022](https://openneuro.org/datasets/ds003838/versions/1.0.2) | Working Memory | 2 | `13_digits` | `memory`, `control` |
| [Sleep-EDF](https://www.physionet.org/content/sleep-edfx/1.0.0/) | Sleep Stage | 6 | | `Sleep stage W`, `Sleep stage 1`, `Sleep stage 2`, `Sleep stage 3`, `Sleep stage 4`, `Sleep stage R` |
| [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) | Eyes Open-Closed | 2 | `eye_open`, `eye_closed` | |

### Pre-process EEG Signal Data
To reproduce the results presented in the paper, the raw EEG signals for each dataset should be:
- resampled to 200Hz
- bandpass filter at 0.5-45Hz
- cut into trials:
  - **High Gamma:** 0s-4s after each cue
  - **OpenBMI-ERP:** 0.2s before - 0.8s after each cue
  - **Pavlov 2022:** 14s-18s after each 13 digits trial cue (corresponding to the peak in pupil size reported in the dataset publication)
  - **Sleep-EDF:** 30s epochs from the original continuous recording, and discard any data from the awake condition except for the 30 minutes before and after sleep
  - **PhysioNet:** 4s epochs from the original continuous recording (only runs 1 and 2)
- saved in numpy format as an array with shape (N_trials, Channels, Time)

### Save Metadata
For compatability with our data loading functions, metadata about each dataset should be saved as a pandas DataFrame where each row corresponds to a single trial
- each row should contain the subject ID
- each row should give the 'task' or 'type' of the trial
- the attributes of the file should include the list of channel names

### Subject-Independent Cross-Validation
We provide the train/validation splits used for 10-fold cross validation on each dataset. The `splits/` folder includes a text file for each dataset which contains the validation subject IDs for each fold.

## Run Benchmarking
Once data has been pre-processed to the expected format and saved under `{DATA_PATH}`, the benchmarking script can be run as below:

```commandline
python main.py \
  --data-root={DATA_PATH} \
  --batch-size=64 \
  --n-epochs=100 \
  --model-name='EEGNetv1' \
  --output-dir='results'
```

## Citation

```
@inproceedings{
lee2025assessing,
title={Assessing the Capabilities of Large Brainwave Foundation Models},
author={Na Lee and Stylianos Bakas and Konstantinos Barmpas and Yannis Panagakis and Dimitrios Adamos and Nikolaos Laskaris and Stefanos Zafeiriou},
booktitle={IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025, Special Sessions},
year={2025},
url={https://openreview.net/forum?id=Qwn2a1uIpx}
}
```