# TVDiag

### TVDiag: A Task-oriented and View-invariant Failure Diagnosis Framework with Multimodal Data

TVDiag is a multimodal failure diagnosis framework designed to locate the root cause and identify the failure type in microservice-based systems. This repository is the basic implementation of TVDiag.


## Project Structure
```
├──requirements.txt
├──main.py
├──README.md
├──TVDiag.py
├──config
│  ├──experiment.yaml
├─dataset
│  ├──dataset.py
├─draw
│  ├──gaia_dependency.py
├──helper
│  ├──aug.py
│  ├──eval.py
│  ├──io.py
│  ├──paths.py
├──loss
│  ├──AutomaticWeightedLoss.py
│  ├──SupervisedContrastiveLoss.py
│  ├──UnsupervisedContrastiveLoss.py
└──model
   ├──backbone
   │  └──FC.py
   ├──Classifier.py
   ├──Encoder.py
   └──MainModel.py
```

## Dataset
We conducted experiments on two dataset:
- [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA dataset records metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists of ten microservices and some middleware such as Redis, MySQL, and Zookeeper. The extracted events of GAIA can be accessible on [DiagFusion](https://arxiv.org/abs/2302.10512).
- [AIOps-22](https://competition.aiops-challenge.com). The AIOps-22 dataset is derived from the training data released by the AIOps 2022 Challenge, where failures at three levels (node, service, and instance) were injected into a Web-based e-commerce platform [Online-boutique](https://github.com/GoogleCloudPlatform/microservices-demo).


## Getting Started


<B>Install Dependencies</B>
```
pip install -r requirements.txt
```

<B>Run</B>

You can directly run the below commands (the default config path is `config/experiment.yaml`):
``` python
python main.py
```
Or you can modify the `config_path` in helper/paths.py.

## Parameters

The parameters in `config/experiment.yaml` are describe as follows:

<B>Common args</B>
- `dataset_name`: The dataset that you want to use.
- `reconstruct`: This parameter reprsents whether the events should be regenerate. (Default: False)

<B>Model</B>
- `TO`: TO denotes whether the task-oriented learning module should be loaded. (Default: True)
- `CM`: CM denotes whether the cross-modal assciation should be establish . (Default: True)
- `guide_weight`: This parameter adjusts the scale of the contrastive loss. (Default: 0.1)
- `aug`: This parameter reprsents whether the data augmentation strategy should be use. (Default: True)
- `aug_method`: Two data augmentation strategies are available: node_drop and random_walk. (Default: node_drop)
- `aug_percent`:  The inactivation probability. (Default: 0.2)
