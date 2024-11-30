import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union
from tabularbench.attacks.capgd.capgd import CAPGD
from tabularbench.attacks.caa.caa import ConstrainedAutoAttack
from tabularbench.attacks.objective_calculator import ObjectiveCalculator
from tabularbench.benchmark.model_utils import load_model_and_weights
from tabularbench.benchmark.subset_utils import get_x_attack
from tabularbench.datasets.dataset_factory import get_dataset
from tabularbench.metrics.compute import compute_metric
from tabularbench.metrics.metric_factory import create_metric
from tabularbench.models.tab_scaler import TabScaler
from tabularbench.utils.datatypes import to_torch_number
import torch


@dataclass
class BenchmarkSettings:
    n_input: int = 1000
    device: str = "cpu"
    n_gen: int = 100
    n_jobs: int = 4
    n_offspring: int = 100
    steps: int = 10
    seed: int = 0
    eps: float = 0.5
    filter_class: int = 1
    filter_correct: bool = False
    adaptive_step: bool = True  # Add a setting for CAPGD-specific parameters
    momentum: float = 0.9


def benchmark(
    dataset: str,
    model: str,
    distance: str,
    constraints: bool,
    settings: BenchmarkSettings = None,
) -> Union[float, float, float]:

    # Aliases
    bms = settings

    if bms is None:
        bms = BenchmarkSettings()

    # Load objects
    ds = get_dataset(dataset)
    x, y = ds.get_x_y()
    i_test = ds.get_splits()["test"]
    x_test, y_test = x.iloc[i_test].to_numpy(), y[i_test]
    metadata = ds.get_metadata(only_x=True)

    scaler = TabScaler(num_scaler="min_max", one_hot_encode=True)
    scaler.fit(to_torch_number(x), x_type=metadata["type"])

    model_arch = model.split("_")[0]
    model_training = "_".join(model.split("_")[1:])

    model_eval = load_model_and_weights(
        ds.name, model_arch, model_training, metadata, scaler, bms.device
    )
    constraints_o = ds.get_constraints()
    if not constraints:
        constraints_o.relation_constraints = None

    metric = create_metric("accuracy")

    clean_acc = compute_metric(
        model_eval,
        metric,
        x_test,
        y_test,
    )

    # Attack
    attacks_settings = {
        "constraints": constraints_o,  # Ensure this key is properly set
        "scaler": scaler,
        "model": model_eval.wrapper_model,
        "model_objective": model_eval.predict_proba,
        "norm": distance,  # 'L2' or 'Linf'
        "eps": bms.eps,
        "steps": bms.steps,
        "n_restarts": 1,  # Update as required
        "seed": bms.seed,
        "loss": "ce",  # Default loss
        "fix_equality_constraints_end": True,
        "fix_equality_constraints_iter": True,
        "adaptive_eps": bms.adaptive_step,  # If adaptive_step is being used
        "verbose": False,  # Or True for debugging
    }

    attack = CAPGD(**attacks_settings)

    x_att, y_att = get_x_attack(
        x_test,
        y_test,
        constraints_o,
        model_eval,
        bms.filter_class,
        bms.filter_correct,
        bms.n_input,
    )
    x_att = torch.tensor(x_att, dtype=torch.float32).to(bms.device)
    y_att = torch.tensor(y_att, dtype=torch.long).to(bms.device)
    print(f"Attacking {len(x_att)} samples.")

    x_adv = attack(x_att, y_att)

    # Detach the tensor from the computation graph and convert to NumPy
    x_adv = x_adv.detach().cpu().numpy()  # Detach and move to CPU if necessary
    x_adv = np.expand_dims(x_adv, 1)

    # Evaluate
    objective_calculator = ObjectiveCalculator(
        classifier=model_eval.predict_proba,
        constraints=constraints_o,
        thresholds={
            "distance": bms.eps,
        },
        norm=distance,
        fun_distance_preprocess=scaler.transform,
    )
    mdc = objective_calculator.get_success_rate(x_att, y_att, x_adv).mdc
    robust_acc = 1 - mdc

    return clean_acc, robust_acc
