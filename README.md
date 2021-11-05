# Machine-learning Based Dynamic Resource Allocation in Network Function Virtualization

Virtual network functions (VNFs), e.g., firewalls or video optimizers, can run on standard x86 servers and can be instantiated, terminated, or migrated according to the current demand. Depending on the load a VNF instance has to process, it requires more or less resources (e.g., CPU) to function properly. Dynamically allocating a suitable amount of resources to each VNF instance is non trivial and depends on the characteristics of the VNF and its current load. Here, we propose to use machine learning regression algorithms to predict VNF resource requirements by learning from raw VNF performance measurements. The resulting machine learning models can be integrated and used in existing VNF placement algorithms to automatically assign the right amount of resources to each new VNF instance or to dynamically adjust allocated resources of already deployed VNF instances.

This repository contains real-world VNF performance data from the [SNDZoo](https://sndzoo.github.io/), code to train machine learning models on this data, as well as trained and saved models, which are ready to use. It also contains all data used to evaluate the impact of using machine learning in VNF placement algorithms. As an example, we adjusted the B-JointSP algorithm for joint scaling and placement of VNFs to use our machine learning models ([see B-JointSP repo with our adjustments](https://github.com/CN-UPB/B-JointSP/tree/ml-resource-prediction)).

If you use this repository, please cite our paper:

> "Machine Learning for Dynamic Resource Allocation in Network Function Virtualization" by Stefan Schneider, Narayanan Puthenpurayil Satheeschandran, Manuel Peuster, and Holger Karl. IEEE Conference on Network Softwarization (NetSoft), 2020.

## Setup

Requires Python 3.6+.

```
pip install -r requirements.txt
```

If there are issues with the installation, check `requirements_freeze.txt`, which is a superset of the required dependencies but with specific version numbers.
With this environment, I successfully ran the notebooks.

## Usage

All code in this repository is Python code running in Jupyter notebooks. After installing the requirements above, you can run Jupyter Lab as follows:

```
jupyter lab
```

This starts the Jupyter server and should open a browser tab showing the Jupyter Lab GUI with the available notebooks. The following notebooks are available:

* `example.ipynb`: Code for generating example VNF data, training ML models on this data. Used for motivational example and illustration in paper.
* `synth_data.ipynb`: Code for generating example VNF data, training and evaluating different ML models on this data.
* `vnf_web.ipynb`: Code reading real-world VNF performance data, processing it, training multiple ML models, tuning hyperparamters of these models, and comparing their RMSE.
* `placement_eval.ipynb`: Code parsing and plotting VNF placement results produced with the B-JointSP algorithm and different ML models.
* `runtime.ipynb`: Code for measuring/parsing the runtimes of prediction and VNF placement with different models

In addition, we adjusted and used the [B-JointSP VNF placement algorithm](https://github.com/CN-UPB/B-JointSP/tree/ml-resource-prediction) for applying our machine learning models to dynamic resource allocation in VNF placement.

## Contact

This work was developed as part of the [RealVNF project](https://realvnf.github.io/).

Lead developer: [Stefan Schneider](https://github.com/stefanbschneider/)

For questions or support, please use GitHub's issue system.
