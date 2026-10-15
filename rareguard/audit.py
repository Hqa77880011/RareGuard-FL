import torch


def trimmed_mean(updates, fraction):
    count = int(len(updates) * fraction)
    if count * 2 >= len(updates):
        raise ValueError("Trimming removes every client")
    return updates.sort(dim=0).values[count:len(updates) - count].mean(0)


def capped_normalization(scores, cap, eligible):
    active = eligible & (scores > 0)
    if active.sum().item() * cap < 1 - 1e-7:
        raise ValueError("Too few eligible clients for the configured maximum weight")
    weights = torch.zeros_like(scores)
    budget = 1.0
    while active.any():
        tentative = budget * scores[active] / scores[active].sum()
        indices = active.nonzero().flatten()
        saturated = tentative > cap
        if not saturated.any():
            weights[indices] = tentative
            break
        fixed = indices[saturated]
        weights[fixed] = cap
        active[fixed] = False
        budget -= len(fixed) * cap
    return weights


def conjugate_gradient(operator, rhs, iterations=30, tolerance=1e-7):
    estimate = torch.zeros_like(rhs)
    residual = rhs.clone()
    direction = residual.clone()
    squared = residual.dot(residual)
    for _ in range(iterations):
        if squared.sqrt() <= tolerance:
            break
        curvature = operator(direction)
        denominator = direction.dot(curvature)
        if denominator <= 0:
            raise ValueError("Nonpositive audit curvature; increase --damping")
        step = squared / denominator
        estimate = estimate + step * direction
        residual = residual - step * curvature
        updated = residual.dot(residual)
        direction = residual + updated / squared * direction
        squared = updated
    return estimate


def influence(model, x, y, updates, center, damping, eta, iterations):
    parameters = tuple(model.parameters())
    loss = torch.nn.functional.cross_entropy(model(x), y)
    gradients = torch.autograd.grad(loss, parameters, create_graph=True)
    gradient = torch.cat([item.reshape(-1) for item in gradients])

    def operator(vector):
        product = torch.dot(gradient, vector)
        curvature = torch.autograd.grad(product, parameters, retain_graph=True)
        return torch.cat([item.detach().reshape(-1) for item in curvature]) + damping * vector

    solved = conjugate_gradient(operator, gradient.detach(), iterations)
    return -eta * ((updates - center) @ solved)


def audit_weights(model, updates, x, y, args, generator, geometry=None):
    center = trimmed_mean(updates, args.trim)
    geometry = updates if geometry is None else geometry
    projection_generator = torch.Generator().manual_seed(args.seed + 991)
    projection = torch.randn(updates.shape[1], args.projection, generator=projection_generator,
                             device="cpu").to(updates) / args.projection ** 0.5
    distances = torch.cdist(geometry @ projection, geometry @ projection).median(dim=1).values
    median = distances.median()
    mad = (distances - median).abs().median().clamp_min(1e-8)
    zscore = (distances - median) / (1.4826 * mad)
    estimates = []
    classes = torch.unique(y)
    for _ in range(args.bootstraps):
        indices = []
        for label in classes:
            rows = (y == label).nonzero().flatten()
            draws = torch.randint(len(rows), (len(rows),), generator=generator)
            indices.append(rows[draws.to(rows.device)])
        indices = torch.cat(indices)
        estimates.append(influence(model, x[indices], y[indices], updates, center,
                                   args.damping, args.eta, args.cg_iterations))
    estimates = torch.stack(estimates)
    mean = estimates.mean(0)
    deviation = estimates.std(0, unbiased=True)
    conservative = mean - args.kappa * deviation
    certainty = ((estimates >= 0) == (mean[None, :] >= 0)).float().mean(0)
    if args.method == "no_utility":
        conservative = torch.zeros_like(conservative)
        certainty = torch.ones_like(certainty)
    if args.method == "no_uncertainty":
        conservative = mean
        certainty = torch.ones_like(certainty)
    integrity = torch.nn.functional.softplus(zscore - args.distance_center)
    eligible = zscore <= args.integrity_veto
    if args.method == "no_integrity":
        integrity = torch.zeros_like(integrity)
        eligible = torch.ones_like(eligible)
    logits = args.utility_temperature * conservative.clamp(-args.influence_clip, args.influence_clip)
    logits -= args.distance_temperature * integrity
    scores = torch.exp(logits - logits.max()) * certainty
    weights = capped_normalization(scores, args.maximum_weight, eligible)
    records = [dict(client_slot=index, distance=float(distances[index]), robust_z=float(zscore[index]),
                    rare_influence_mean=float(mean[index]), rare_influence_sd=float(deviation[index]),
                    conservative_utility=float(conservative[index]), certainty=float(certainty[index]),
                    integrity_eligible=bool(eligible[index]), weight=float(weights[index]))
               for index in range(len(updates))]
    return weights, records
