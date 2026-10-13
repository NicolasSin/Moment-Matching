"""Train the scalar K=1500 example: 1000 Adam steps, then 4000 GH steps."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

# Match the original launcher's BLAS settings before importing numerical
# libraries. NumPy's local quadrature otherwise changes in the final bits.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "4"

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from common import read_scalar_sequence, setup, write_json
from empirical_quadrature import build_empirical_quadrature
from gh_objective import gh_rule, weighted_moment_loss, weighted_model_moment
from model import MMGenerator, save_checkpoint
from objective import FeatureBank, data_moment, moment_loss


def new_lbfgs(model, config):
    return torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=1, max_eval=25,
        history_size=config["refinement"]["lbfgs_history"],
        tolerance_grad=0.0, tolerance_change=0.0, line_search_fn="strong_wolfe",
    )


def adam_step(model, optimizer, states, bank, target, generators, config, update):
    """One original float32 Adam update with independent sampling/noise streams."""
    fraction = (update - 1) / max(1, config["adam_max_updates"] - 1)
    learning_rate = config["adam_lr_end"] + 0.5 * (
        config["adam_lr_start"] - config["adam_lr_end"]
    ) * (1 + math.cos(math.pi * fraction))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    indices = torch.randperm(
        len(states), device=states.device, generator=generators["state_sampling"]
    )[:config["moment_samples"]]
    sampled = states[indices]
    noise = torch.randn(
        (len(sampled), config["R_MM"]), device=states.device,
        generator=generators["mm_noise"],
    )
    optimizer.zero_grad(set_to_none=True)
    loss, _ = moment_loss(
        model, sampled, noise, bank, target, chunk=config["moment_chunk"], backward=True
    )
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite Adam loss")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["gradient_clip"], error_if_nonfinite=True
    ))
    optimizer.step()
    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
        raise FloatingPointError("Nonfinite Adam parameters")
    return dict(loss=loss, gradient_norm=gradient_norm, learning_rate=learning_rate)


def train(config, data_directory, output, device):
    setup(device)
    started = time.perf_counter()
    deadline = started + config["total_training_seconds_cap"]
    train_path = data_directory / "train_x.npy"
    observations = read_scalar_sequence(train_path)
    if len(observations) != config["train_points"]:
        raise ValueError("Unexpected training sequence length")
    seeds = config["seeds"]
    normalization = config["normalization_source_float64"]
    # Initialize in float32, including normalization buffers. The GH phase
    # promotes these exact working values to float64; it never recomputes them.
    model = MMGenerator(normalization["mean"], normalization["std"],
                        seed=seeds["initialization"]).to(device)
    canonical = FeatureBank.draw(config["features_per_side"], config["gamma_x"],
                                 seeds["rff_x"], seeds["rff_y"])
    generators = {
        name: torch.Generator(device=device).manual_seed(seeds[name])
        for name in ("state_sampling", "mm_noise")
    }
    adam_updates = gh_updates = restarts = 0
    stage = "adam"
    last_loss = None

    def save(name):
        save_checkpoint(output / name, model, config=config, features=canonical.state_dict(),
                        stage=stage, adam_updates=adam_updates, gh_updates=gh_updates)

    def check_time():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Training time budget reached; the run is incomplete")

    try:
        bank32 = canonical.to(device, torch.float32)
        states32 = torch.tensor(observations[:-1], device=device, dtype=torch.float32)
        next32 = torch.tensor(observations[1:], device=device, dtype=torch.float32)
        target32 = data_moment(states32, next32, bank32, chunk=config["data_chunk"])
        del next32
        optimizer = torch.optim.Adam(model.parameters(), lr=config["adam_lr_start"],
                                     eps=config["adam_eps"], weight_decay=config["weight_decay"])
        for update in range(1, config["adam_max_updates"] + 1):
            check_time()
            # Roll back a failed update so model_partial.pt contains a completed step.
            backup = parameters_to_vector(model.parameters()).detach().clone()
            try:
                values = adam_step(model, optimizer, states32, bank32, target32,
                                   generators, config, update)
            except BaseException:
                vector_to_parameters(backup, model.parameters())
                raise
            adam_updates = update
            last_loss = values["loss"]
        del states32, target32, bank32, optimizer

        stage = "gh"
        model.double()
        bank64 = canonical.to(device, torch.float64)
        states64 = torch.tensor(observations[:-1], device=device, dtype=torch.float64)
        next64 = torch.tensor(observations[1:], device=device, dtype=torch.float64)
        target = data_moment(states64, next64, bank64, chunk=config["data_chunk"])
        del states64, next64
        refinement = config["refinement"]
        nodes, weights = build_empirical_quadrature(
            observations[:-1], width=normalization["std"] / refinement["input_width_divisor"]
        )
        states = torch.tensor(nodes, device=device, dtype=torch.float64)
        input_weights = torch.tensor(weights, device=device, dtype=torch.float64)
        latent, latent_weights = gh_rule(refinement["gaussian_order"], device)

        def objective(backward=False):
            if backward:
                return weighted_moment_loss(
                    model, states, latent, bank64, target, latent_weights,
                    chunk=refinement["chunk"], x_weights=input_weights,
                    backward=True, weight=refinement["optimizer_scale"],
                )[0]
            matrix = weighted_model_moment(model, states, latent, bank64, latent_weights,
                                            chunk=refinement["chunk"], x_weights=input_weights)
            return float((matrix - target).square().mean())

        optimizer = new_lbfgs(model, config)
        last_loss = objective()
        while gh_updates < refinement["max_updates"]:
            check_time()
            backup = parameters_to_vector(model.parameters()).detach().clone()
            optimizer_backup = copy.deepcopy(optimizer.state_dict())

            def closure():
                check_time()
                optimizer.zero_grad(set_to_none=True)
                loss = objective(backward=True)
                if not math.isfinite(loss) or not all(
                    parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                ):
                    raise RuntimeError("Nonfinite GH loss or gradient")
                return torch.tensor(loss * refinement["optimizer_scale"], device=device, dtype=torch.float64)

            try:
                optimizer.step(closure)
                loss = objective()
                movement = float((parameters_to_vector(model.parameters()).detach() - backup).norm())
                if not math.isfinite(loss) or loss > last_loss + 1e-18 or movement == 0:
                    raise RuntimeError("GH update increased the loss or made no parameter movement")
            except BaseException as error:
                vector_to_parameters(backup, model.parameters())
                optimizer.load_state_dict(optimizer_backup)
                if not isinstance(error, RuntimeError) or restarts >= refinement["max_restarts"]:
                    raise
                restarts += 1
                optimizer = new_lbfgs(model, config)
                continue
            gh_updates += 1
            last_loss = loss
        save("model_final.pt")
    except BaseException as error:
        save("model_partial.pt")
        write_json(output / "summary.json", dict(
            status="incomplete", reason=str(error), adam_updates=adam_updates, gh_updates=gh_updates,
            elapsed_seconds=time.perf_counter() - started,
        ))
        raise
    write_json(output / "summary.json", dict(
        status="completed", adam_updates=adam_updates, gh_updates=gh_updates, training_loss=last_loss,
        elapsed_seconds=time.perf_counter() - started,
        selection="Last fixed-step model; validation is not used for training or selection",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing paths are refused")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    if any(output.is_relative_to(ROOT / name) for name in ("code", "configs", "data", "docs", "models", "results", "scripts")):
        parser.error("Choose an output outside the packaged code, data, models, and results")
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    train(config, args.data, output, args.device)


if __name__ == "__main__":
    main()
