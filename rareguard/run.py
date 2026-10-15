import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from .audit import audit_weights, trimmed_mean
from .data import load


METHODS = ("rareguard", "fedavg", "fedprox", "scaffold", "trimmed", "krum", "fltrust",
           "class_balanced_trust", "no_integrity", "no_utility", "no_uncertainty", "no_tail")


class Classifier(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        layers = []
        for width, kernel in ((32, 7), (64, 5), (128, 3)):
            layers.extend([nn.Conv1d(channels, width, kernel, stride=2, padding=kernel // 2),
                           nn.GroupNorm(8, width), nn.GELU()])
            channels = width
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        return self.head(self.features(x))


def gradient_vector(model, x, y, balanced=False):
    logits = model(x)
    losses = F.cross_entropy(logits, y, reduction="none")
    if balanced:
        counts = torch.bincount(y, minlength=logits.shape[1]).clamp_min(1)
        weights = 1 / counts[y].to(losses)
        loss = (losses * weights).sum() / weights.sum()
    else:
        loss = losses.mean()
    return torch.cat([g.detach().reshape(-1) for g in torch.autograd.grad(loss, tuple(model.parameters()))])


def aggregate(updates, counts, model, audit_x, audit_y, args, generator, geometry):
    records = []
    if args.method in ("fedavg", "fedprox", "scaffold"):
        weight = counts / counts.sum()
        return weight @ updates, records
    if args.method == "trimmed":
        return trimmed_mean(updates, args.trim), records
    if args.method == "krum":
        adversaries = int(np.ceil(args.byzantine * len(updates)))
        if len(updates) < 2 * adversaries + 3:
            raise ValueError("Krum requires n >= 2*f+3; reduce the Byzantine fraction")
        distances = torch.cdist(updates, updates).square()
        distances.fill_diagonal_(torch.inf)
        score = distances.sort(1).values[:, :len(updates) - adversaries - 2].sum(1)
        return updates[score.argmin()], records
    if args.method in ("fltrust", "class_balanced_trust"):
        reference = -gradient_vector(model, audit_x, audit_y, args.method == "class_balanced_trust")
        norm = updates.norm(dim=1).clamp_min(1e-12)
        trust = F.relu(F.cosine_similarity(updates, reference[None, :], dim=1))
        if trust.sum() <= 0:
            raise ValueError("No positive client/reference cosine in trust aggregation")
        normalized = updates / norm[:, None] * reference.norm()
        return (trust / trust.sum()) @ normalized, records
    center = trimmed_mean(updates, args.trim)
    base = parameters_to_vector(model.parameters()).detach().clone()
    vector_to_parameters(base + args.server_lr * center, model.parameters())
    selected = torch.isin(audit_y, torch.tensor(args.rare_labels, device=audit_y.device))
    if not selected.any():
        raise ValueError("The audited resource has no registered rare examples")
    weights, records = audit_weights(model, updates, audit_x[selected], audit_y[selected], args, generator, geometry)
    vector_to_parameters(base, model.parameters())
    return weights @ updates, records


@torch.no_grad()
def predict(model, x, device, temperature=1):
    model.eval()
    result = []
    for start in range(0, len(x), 256):
        logits = model(torch.from_numpy(x[start:start + 256]).to(device))
        result.append((logits / temperature).softmax(1).cpu().numpy())
    return np.concatenate(result)


def metrics(model, data, split, args, temperature=1):
    indices = data["split"] == split
    y = data["y"][indices]
    p = predict(model, data["x"][indices], args.device, temperature)
    prediction = p.argmax(1)
    confidence = p.max(1)
    correct = prediction == y
    rare = np.isin(y, args.rare_labels)
    bins = np.minimum((confidence * 15).astype(int), 14)
    ece = sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
              for b in range(15) if np.any(bins == b))
    binary = rare.astype(int)
    rare_score = p[:, args.rare_labels].sum(1)
    auprc = float(average_precision_score(binary, rare_score)) if len(np.unique(binary)) == 2 else None
    client_errors = []
    for client in np.unique(data["client"][indices]):
        selected = data["client"][indices] == client
        client_errors.append(float(np.mean(prediction[selected] != y[selected])))
    return dict(rare_auprc=auprc, macro_f1=float(f1_score(y, prediction,
                labels=range(p.shape[1]), average="macro", zero_division=0)),
                attack_success_rate=float(np.mean(prediction[rare] == args.attack_target)) if rare.any() else None,
                worst_client_error=max(client_errors), ece=float(ece), accuracy=float(correct.mean()))


def calibrate(model, x, y):
    with torch.no_grad():
        logits = model(x).detach()
    log_temperature = torch.zeros((), device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.clamp(-4, 4).exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-4, 4).exp())


def run(args):
    if not 0 <= args.byzantine < 0.5 or not 0 < args.participation <= 1:
        raise ValueError("Require Byzantine fraction in [0,0.5) and participation in (0,1]")
    if args.bootstraps < 2 or args.damping <= 0 or args.projection < 1:
        raise ValueError("At least two bootstraps and positive damping/projection required")
    if min(args.rounds, args.local_epochs, args.batch_size, args.cg_iterations) < 1:
        raise ValueError("Training budgets and solver iterations must be positive")
    if min(args.local_lr, args.server_lr, args.clip_multiplier) <= 0 or not 0 <= args.trim < 0.5:
        raise ValueError("Invalid learning rates, clipping or trimming fraction")
    if not 0 < args.tail_fraction <= 1 or not 0 < args.maximum_weight <= 1:
        raise ValueError("Invalid CVaR fraction or weight cap")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    data = load(args.data)
    classes = int(data["y"].max()) + 1
    if not args.rare_labels or min(args.rare_labels) < 0 or max(args.rare_labels) >= classes:
        raise ValueError("Rare labels must be declared class IDs")
    if args.attack_target in args.rare_labels or not 0 <= args.attack_target < classes:
        raise ValueError("Attack target must be a valid non-rare class")
    device = torch.device(args.device)
    model = Classifier(data["x"].shape[1], classes).to(device)
    clients = np.unique(data["client"][data["split"] == "train"]).tolist()
    participating = max(1, int(np.ceil(len(clients) * args.participation)))
    malicious_count = int(np.floor(len(clients) * args.byzantine))
    malicious = set(rng.choice(clients, malicious_count, replace=False).tolist())
    audit = data["split"] == "audit"
    audit_x = torch.from_numpy(data["x"][audit]).to(device)
    audit_y = torch.from_numpy(data["y"][audit]).to(device)
    is_rare = torch.isin(audit_y, torch.tensor(args.rare_labels, device=device))
    if not is_rare.any() or is_rare.all():
        raise ValueError("The audited resource needs rare and non-rare examples")
    val = data["split"] == "val"
    val_x = torch.from_numpy(data["x"][val]).to(device)
    val_y = torch.from_numpy(data["y"][val]).to(device)
    controls = {client: torch.zeros_like(parameters_to_vector(model.parameters())) for client in clients}
    server_control = torch.zeros_like(parameters_to_vector(model.parameters()))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best = -np.inf
    temperature = 1
    communication = 0
    start = time.perf_counter()
    for round_index in range(1, args.rounds + 1):
        base = parameters_to_vector(model.parameters()).detach().clone()
        selected_clients = sorted(rng.choice(clients, participating, replace=False).tolist())
        deltas, counts, control_deltas, local_steps = [], [], [], []
        for client in selected_clients:
            local = copy.deepcopy(model)
            local.train()
            selected = (data["client"] == client) & (data["split"] == "train")
            x = torch.from_numpy(data["x"][selected])
            y = torch.from_numpy(data["y"][selected].copy())
            if client in malicious and args.attack in ("label_flip", "model_replacement"):
                rare = torch.isin(y, torch.tensor(args.rare_labels))
                y[rare] = args.attack_target
            loader_generator = torch.Generator().manual_seed(args.seed + round_index * 1009 + clients.index(client))
            loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size,
                                shuffle=True, generator=loader_generator)
            optimizer = torch.optim.SGD(local.parameters(), lr=args.local_lr)
            steps = 0
            for _ in range(args.local_epochs):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = F.cross_entropy(local(xb), yb)
                    if args.method == "fedprox":
                        loss = loss + args.proximal / 2 * (parameters_to_vector(local.parameters()) - base).square().sum()
                    loss.backward()
                    if args.method == "scaffold":
                        correction = server_control - controls[client]
                        offset = 0
                        for parameter in local.parameters():
                            size = parameter.numel()
                            parameter.grad.add_(correction[offset:offset + size].view_as(parameter))
                            offset += size
                    optimizer.step()
                    steps += 1
            delta = parameters_to_vector(local.parameters()).detach() - base
            if args.method == "scaffold":
                updated = controls[client] - server_control - delta / (steps * args.local_lr)
                control_deltas.append(updated - controls[client])
                controls[client] = updated
            deltas.append(delta / steps)
            local_steps.append(steps)
            counts.append(len(x))
        updates = torch.stack(deltas)
        norms = updates.norm(dim=1)
        radius = args.clip_multiplier * norms.median().clamp_min(1e-12)
        for slot, client in enumerate(selected_clients):
            if client not in malicious:
                continue
            if args.attack == "sign_flip":
                updates[slot] *= -args.attack_scale
            elif args.attack == "model_replacement":
                updates[slot] *= args.attack_scale
            elif args.attack == "mimicry":
                rare = torch.isin(audit_y, torch.tensor(args.rare_labels, device=device))
                rare_direction = -gradient_vector(model, audit_x[rare], audit_y[rare])
                common_direction = gradient_vector(model, audit_x[~rare], audit_y[~rare])
                direction = rare_direction / rare_direction.norm().clamp_min(1e-12)
                direction = direction + args.mimic_strength * common_direction / common_direction.norm().clamp_min(1e-12)
                updates[slot] = direction / direction.norm().clamp_min(1e-12) * radius
        updates *= (radius / updates.norm(dim=1).clamp_min(1e-12)).clamp_max(1)[:, None]
        model.eval()
        full_updates = updates * torch.tensor(local_steps, device=device)[:, None]
        aggregate_delta, records = aggregate(full_updates, torch.tensor(counts, dtype=updates.dtype, device=device),
                                              model, audit_x, audit_y, args, generator, updates)
        vector_to_parameters(base + args.server_lr * aggregate_delta, model.parameters())
        if args.method == "scaffold":
            server_control += sum(control_deltas) / len(clients)
        audited_method = args.method in ("rareguard", "no_integrity", "no_utility", "no_uncertainty", "no_tail")
        if audited_method:
            anchor = parameters_to_vector(model.parameters()).detach().clone()
            optimizer = torch.optim.SGD(model.parameters(), lr=args.server_refine_lr)
            rare = torch.isin(audit_y, torch.tensor(args.rare_labels, device=device))
            for _ in range(args.server_steps):
                optimizer.zero_grad()
                losses = F.cross_entropy(model(audit_x), audit_y, reduction="none")
                rare_losses = losses[rare]
                count = max(1, int(np.ceil(len(rare_losses) * args.tail_fraction)))
                tail = rare_losses.topk(count).values.mean()
                coefficient = 0 if args.method == "no_tail" else args.tail_weight
                proximal = args.server_proximal / 2 * (parameters_to_vector(model.parameters()) - anchor).square().sum()
                loss = losses.mean() + coefficient * tail + proximal
                loss.backward()
                optimizer.step()
        temperature = calibrate(model, val_x, val_y)
        validation = metrics(model, data, "val", args, temperature)
        if validation["rare_auprc"] is None:
            raise ValueError("Validation requires both rare and non-rare examples")
        for record, client in zip(records, selected_clients):
            record["client"] = client
            record["attack_injected"] = client in malicious
        parameter_bytes = base.numel() * base.element_size()
        communication += 2 * participating * parameter_bytes
        if args.method == "scaffold":
            communication += 2 * participating * parameter_bytes
        communication += 8 * participating
        row = dict(round=round_index, validation=validation, audit=records,
                   temperature=temperature, communication_bytes=communication,
                   elapsed_seconds=time.perf_counter() - start)
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["rare_auprc"] > best:
            best = validation["rare_auprc"]
            torch.save(dict(state=model.state_dict(), temperature=temperature, round=round_index), output / "best.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state"])
    result = dict(config=vars(args), best_round=checkpoint["round"], validation_rare_auprc=best,
                  test=metrics(model, data, "test", args, checkpoint["temperature"]),
                  communication_bytes=communication, elapsed_seconds=time.perf_counter() - start,
                  malicious_clients=sorted(malicious), audit_seconds_included=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
