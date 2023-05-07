# TVDiag

### TVDiag: A Task-oriented and View-invariant Failure Diagnosis Framework with Multimodal Data

TVDiag is a multimodal failure diagnosis framework designed to locate the root cause and identify the failure type in microservice-based systems. This repository is the basic implementation of TVDiag. We have released the core code of TVDiag, and the remaining parts will be added in the final version.


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
- [GAIA](https://github.com/CloudWise-OpenSource/GAIA-DataSet). GAIA dataset records metrics, traces, and logs of the MicroSS simulation system in July 2021, which consists of ten microservices and some middleware such as Redis, MySQL, and Zookeeper. The extracted events of GAIA can be accessible on [DiagFusion](https://arxiv.org/abs/2302.10512)
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
- `dataset_name`: Which dataset you want to use.
- `reconstruct`: This parameter reprsents Whether to regenerate the events. (default: False)

<B>model</B>
- `TO`: TO denotes Whether to load the task-oriented learning module. (default: True)
- `CM`: CM denotes Whether to establish the cross-modal assciation. (default: True)
- `guide_weight`: This parameter adjust the scale of contrastive loss. (default: 0.1)
- `aug`: This parameter reprsents Whether to augment the dataset. (default: True)
- `aug_method`: You can use two data augmentation strategies: node_drop and random_walk. (default: node_drop)
- `aug_percent`:  The inactivation probability. (default: 0.2)
