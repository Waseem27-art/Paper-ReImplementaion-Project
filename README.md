# **The major changes implemented in the forked code:**

<br />
<br />


**modified** /tabularbench/benchmark/benchmark.py to run the CAPGD attack instead of CAA

**modified** the tasks/run_benchmark.py each time to test on different datasets and models

**implemented Lowprofool attack on URL dataset** (u may face some small isssues in the final attack)

note: initially we tried to clone the repo and implement our own attack.py with our own parameters and constraints .Due too much fluctuations from the code; we had to switch back to basic one. we used Docker cli using podman.

Due to merging of the task paper with Tabularmenching other attacks and models, we improvised the paper and **attacked on both original model**(for eg:stg_url_default.model) and **adversary one**(stg_url_madry.model)

for viewers: the paper:- CAPGD attack: Towards Adaptive Attacks on Constrained Tabular Machine Learning


<br />

<br />

**REST OF THE README OF ORIGINAL REPO:**



# TabularBench

TabularBench: Adversarial robustness benchmark for tabular data

**Leaderboard**: [https://serval-uni-lu.github.io/tabularbench/](https://serval-uni-lu.github.io/tabularbench/)

**Documentation**: [https://serval-uni-lu.github.io/tabularbench/doc](https://serval-uni-lu.github.io/tabularbench/doc)

**Research papers**:

- Benchmark: [TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases](https://arxiv.org/abs/2408.07579)
- CAA attack: [Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data](https://arxiv.org/abs/2406.00775)
- CAPGD attack: [Towards Adaptive Attacks on Constrained Tabular Machine Learning](https://openreview.net/forum?id=DnvYdmR9OB)
- MOEVA attack: [A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space](https://arxiv.org/abs/2112.01156)

**How to cite**:

Would you like to reference the CAA attack?

Then consider citing our paper, to appear in NeurIPS 2024 (spotlight):

```bibtex
@misc{simonetto2024caa,
    title={Constrained Adaptive Attack: Effective Adversarial Attack Against Deep Neural Networks for Tabular Data},
    author={Thibault Simonetto and Salah Ghamizi and Maxime Cordy},
    booktitle={To appear in Advances in Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2406.00775},
}
```

Would you like to reference the benchmark, the leaderboard or the model zoo?

Then consider citing our paper, to appear in NeurIPS 2024 Datasets and Benchmarks:

```bibtex
@misc{simonetto2024tabularbench,
    title={TabularBench: Benchmarking Adversarial Robustness for Tabular Deep Learning in Real-world Use-cases},
    author={Thibault Simonetto and Salah Ghamizi and Maxime Cordy},
    booktitle={To appear in Advances in Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2408.07579},
}
```

## Installation

### Using Docker (recommended)

1. Clone the repository

2. Build the Docker image

    ```bash
    ./tasks/docker_build.sh
    ```

3. Run the Docker container

    ```bash
    ./tasks/run_benchmark.sh
    ```

Note: The `./tasks/run_benchmark.sh` script mounts the current directory to the `/workspace` directory in the Docker container.
This allows you to edit the code on your host machine and run the code in the Docker container without rebuilding.

### Using Pip

We recommend using Python 3.8.10.

1. Install the package from PyPI

    ```bash
    pip install tabularbench
    ```

### With Pyenv and Poetry

1. Clone the repository

2. Create a virtual environment using [Pyenv](https://github.com/pyenv/pyenv) with Python 3.8.10.

3. Install the dependencies using [Poetry](https://python-poetry.org/).

 ```bash
    poetry install
 ```

### Using conda

1. Clone the repository

2. Create a virtual environment using [Conda](https://docs.anaconda.com/free/miniconda/) with Python 3.8.10.

    ```bash
    conda create -n tabularbench python=3.8.10
    ```

3. Activate the conda environment.

    ```bash
    conda activate tabularbench
    ```

4. Install the dependencies using Pip.

    ```bash
    pip install -r requirements.txt
    ```

## How to use

### Run the benchmark

You can run the benchmark with the following command:

```bash
python -m tasks.run_benchmark
```

or with Docker:

```bash
docker_run_benchmark
```

### Using the API

You can also use the API to run the benchmark. See `tasks/run_benchmark.py` for an example.

```python
clean_acc, robust_acc = benchmark(
    dataset="URL",
    model="STG_Default",
    distance="L2",
    constraints=True,
)
```

### Retrain the models

We provide the models and parameters used in the paper.
You can retrain the models with the following command:

```bash
python -m tasks.train_model
```

Edit the `tasks/train_model.py` file to change the model, dataset, and training method.

## Data availability

Datasets, pretrained models, and synthetic data are publicly available [here](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/thibault_simonetto_uni_lu/EvkG4BI0EqJFu436biA2C_sBpkEKTTjA5PgZU_Z9jwNNSA?e=62a4Dm).
The folder structure on the Shared folder should be followed locally to ensure the code runs correctly.

> [!NOTE]
> We are transitioning to Hugging Face for data storage. The model's data is now available on Huggin Face [here](https://huggingface.co/serval-uni-lu/tabularbench/tree/main).

**Datasets**: Datasets are downloaded automatically in `data/datasets` when used.

**Models** (HuggingFace): Models are now downloaded automatically as needed when running the benchmark. Only the required model for a specific setting will be downloaded. Pretrained models remain available in the `data/models` folder on OneDrive.

**Model parameters**: Optimal parameters (from hyperparameters search) are required to train models and are in `data/model_parameters`.

**Synthetic data**: The synthetic data generated by GANs is available in the folder `data/synthetic`.

## Naming

For technical reasons, the names of datasets, models, and training methods are different from the paper.
The mapping can be found in [docs/naming.md](docs/naming.md).
