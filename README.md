# TVDiag

### TVDiag: A Task-oriented and View-invariant Failure Diagnosis Framework with Multimodal Data

*TVDiag* is a multimodal failure diagnosis framework designed to locate the root cause and identify the failure type in microservice-based systems. This repository offers the core implementation of *TVDiag*.

![](./imgs/structure.png)

## Project Structure

```
.
├── core
│   ├── loss
│   │   ├── AutomaticWeightedLoss.py
│   │   ├── SupervisedContrastiveLoss.py
│   │   └── UnsupervisedContrastiveLoss.py
│   ├── model
│   │   ├── backbone
│   │   │   ├── FC.py
│   │   │   ├── sage.py
│   │   │   └── cnn1d.py
│   │   ├── Classifier.py
│   │   ├── Voter.py
│   │   ├── Encoder.py
│   │   └── MainModel.py
│   ├── aug.py
│   ├── ita.py
│   ├── multimodal_dataset.py
│   └── TVDiag.py
├── data
│   └── gaia
│       ├── tmp
│       ├── raw
│       └── label.csv
├── helper
│   ├── eval.py
│   ├── io_uitl.py
│   ├── logger.py
│   ├── scaler.py
│   └── time_util.py
├── process
│   ├── events
│   │   ├── fasttext_w2v.py
│   │   ├── cnn1d_w2v.py
│   │   └── lda_w2v.py
│   └── EventProcess.py
├── requirements.txt
├── README.md
├── train.sh
└── main.py
```

## Dataset

We conducted experiments on two datasets:

- [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA dataset records metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists of ten microservices and some middleware such as Redis, MySQL, and Zookeeper. The extracted events of GAIA can be accessible on [DiagFusion](https://arxiv.org/abs/2302.10512).
- [AIOps-22](https://competition.aiops-challenge.com). The AIOps-22 dataset is derived from the training data released by the AIOps 2022 Challenge, where failures at three levels (node, service, and instance) were injected into a Web-based e-commerce platform [Online-boutique](https://github.com/GoogleCloudPlatform/microservices-demo).

## Getting Started

<B>Install Dependencies</B>

```
pip install -r requirements.txt
```

<B>Run</B>

You can directly run the below commands:

```python
sh train.sh
```

The parameters in `main.py` are described as follows:

<B>Common args</B>

- `dataset`: The dataset that you want to use.
- `reconstruct`: This parameter represents whether the events should be regenerated. (Default: False)

<B>Model</B>

- `TO`: TO denotes whether the task-oriented learning module should be loaded. (Default: True)
- `CM`: CM denotes whether the cross-modal association should be established. (Default: True)
- `dynamic_weight`: dynamic_weight denotes whether weights are dynamically assigned for each loss. (Default: True)
- `guide_weight`: This parameter adjusts the scale of the contrastive loss. (Default: 0.1)
- `temperature`: This parameter adjusts the temprature parameter $\tau$, controlling the the attention to difficult samples. (Default: 0.3)
- `patience`: This parameter adjusts the patience used in early break. (Default: 10)
- `aug_percent`:  The inactivation probability. (Default: 0.2)
