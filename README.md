# RareGuard-FL

Implementation of **RareGuard-FL: Counterfactual Client-Influence Auditing for Rare-Failure-Preserving Robust Federated Maintenance**.

RareGuard-FL separates three aggregation signals: projected-update integrity, a client's estimated rare-loss removal influence, and bootstrap sign stability. It caps the resulting weights and refines the aggregated model using rare-example CVaR. The repository includes recording preparation, federated training, Byzantine attacks, baselines, component ablations, calibration, evaluation and figures.

## Install

Use Python 3.11 or 3.12 from the repository root:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment before installation (`source .venv/bin/activate`, or `.venv\Scripts\Activate.ps1` in PowerShell). For GPU execution, install the PyTorch build matching your CUDA environment first. `--device cpu` is the default.

## 1. Prepare recordings and partitions

Download the chosen data from the [MIMII release](https://zenodo.org/records/3384388), [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter), or [Paderborn KAt Data Center](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter). Use one dataset per experiment. The preparation command consumes a recording manifest so that machine identity, client membership and rare-class definitions remain explicit:

```csv
path,key,label,client,machine,split
raw/machine_a_001.wav,,0,client00,machine_a,train
raw/machine_b_001.wav,,1,client01,machine_b,train
raw/machine_c_001.wav,,1,audit,machine_c,audit
raw/machine_d_001.wav,,0,validation,machine_d,val
raw/machine_e_001.wav,,1,test00,machine_e,test
```

This example illustrates the schema; supply complete records including both rare and ordinary classes in audit, validation and test partitions. `path` is relative to the manifest. `key` selects a MAT variable; slash-separated fields and zero-based indices support nested simplified MAT structures. WAV recordings are averaged across audio channels, and NPY inputs contain a one-dimensional signal. Input samples retain their native sampling rate.

`label` is an integer starting at zero; `client` assigns training machines to federation members; `machine` identifies a physical machine/bearing or a recording-session grouping supported by the dataset metadata. All recordings from that group must belong to one client and one split. A machine crossing splits is rejected before window extraction. Use only group identities actually provided by the release. For CWRU, record-session grouping does not establish generalization to unseen physical bearings.

Split values are `train`, `audit`, `val`, `test`. Audit examples are a separately declared server resource; validation selects checkpoints and temperature; test examples are accessed only after training. Recordings must already be assigned to splits before windowing. Assign 20 training clients and concentrate rare recordings on four clients for the main paper setting, or create alternative manifests for prevalence and heterogeneity studies. There is no automatic random sample split.

```bash
python -m rareguard prepare --manifest data/mimii.csv --output data/mimii.npz --representation spectrum
python -m rareguard prepare --manifest data/cwru.csv --output data/cwru.npz --representation waveform
```

Defaults use windows of 4096 samples and stride 2048. `waveform` yields `[N,1,L]`; `spectrum` uses log-magnitude STFT features with a 256-sample transform and 128-sample overlap, yielding `[N,frequency,time]`. A shared convolutional classifier processes the resulting channels. Normalization uses training-only channel statistics and is saved in the NPZ. Training-statistic aggregation is an offline preparation convention; the federated loop exchanges parameters and metadata.

## 2. Run the method

```bash
python -m rareguard train --data data/mimii.npz --output outputs/rareguard_0 --rare-labels 1 --method rareguard --attack mimicry --byzantine 0.2 --seed 0 --device cuda
```

The model contains three 1D convolution blocks with widths 32/64/128, kernels 7/5/3, stride two, group normalization, GELU and global pooling. Every method uses the same initialization and client/batch schedule for a given seed. Local optimization uses SGD. Client geometry is computed from updates divided by local step count and clipped to a common median-norm radius; the resulting clipping factors also apply to complete parameter updates. Final aggregation combines full local updates.

The server forms a coordinate-trimmed center, evaluates rare cross-entropy there, and computes the damped Hessian-inverse gradient through Hessian-vector products and conjugate gradients. Stratified bootstrap resampling gives per-client influence means, standard deviations and sign-stability frequencies. Conservative influence subtracts `kappa * standard_deviation`. Integrity and utility remain separate terms in the weighting equation. An integrity veto applies before capped normalization. CVaR uses the largest specified fraction of rare audit losses, with a proximal penalty around the aggregate.

Key parameters:

| Argument | Default | Meaning |
|---|---:|---|
| `--rounds`, `--local-epochs`, `--batch-size` | 100, 2, 64 | Training budgets |
| `--local-lr`, `--server-lr` | 0.01, 1 | Local SGD and server update scale |
| `--participation` | 0.4 | Fraction of training clients sampled each round, rounded up |
| `--byzantine` | 0.2 | Fraction assigned malicious status at startup, rounded down |
| `--rare-labels` | required | One or more rare class IDs |
| `--trim` | 0.2 | Coordinate fraction removed from each end |
| `--projection` | 32 | Fixed random projection dimension |
| `--clip-multiplier` | 2 | Median per-step update norm multiplier |
| `--bootstraps`, `--cg-iterations` | 8, 30 | Audit resamples and maximum solver iterations |
| `--damping` | 10 | Positive Hessian damping |
| `--eta`, `--kappa` | 1, 1 | Influence scale and uncertainty penalty |
| `--utility-temperature`, `--distance-temperature` | 4, 1 | Utility reward and integrity penalty scales |
| `--distance-center`, `--integrity-veto` | 2, 6 | Soft penalty center and hard robust-z veto |
| `--influence-clip` | 1 | Absolute influence clipping level |
| `--maximum-weight` | 0.35 | Per-client aggregation cap |
| `--tail-fraction`, `--tail-weight` | 0.2, 0.5 | Rare CVaR tail mass and coefficient |
| `--server-steps`, `--server-refine-lr`, `--server-proximal` | 1, 0.001, 1 | Post-aggregation refinement |
| `--proximal` | 0.01 | FedProx coefficient |

The audit runs every round. Projection is fixed for the seed, while bootstrap resamples advance across rounds. Evidence-model and simulator details absent from the paper are defined by these arguments. This implementation uses the centralized audit-resource regime described by the main evaluation.

A nonpositive conjugate-gradient curvature raises an error identifying the damping setting; increase `--damping` if needed. A cap infeasible after integrity vetoes also raises an error. In particular, the number of eligible clients times `--maximum-weight` must be at least one. These conditions do not trigger an unreported aggregation fallback.

## 3. Compare baselines and components

```bash
python -m rareguard train --data data/mimii.npz --output outputs/trimmed_0 --rare-labels 1 --method trimmed --attack mimicry --seed 0 --device cuda
python -m rareguard train --data data/mimii.npz --output outputs/no_tail_0 --rare-labels 1 --method no_tail --attack mimicry --seed 0 --device cuda
```

| Method | Behavior |
|---|---|
| `fedavg` | Sample-count weighted local updates |
| `fedprox` | FedAvg with local proximal regularization |
| `scaffold` | Client/server control variates and partial participation |
| `trimmed` | Coordinate-wise trimmed mean |
| `krum` | Selects the minimum nearest-update-distance score |
| `fltrust` | Positive cosine trust and norm matching to an audited gradient |
| `class_balanced_trust` | FLTrust-style comparison using class-balanced audit cross-entropy |
| `rareguard` | Complete integrity/utility/uncertainty/CVaR method |
| `no_integrity` | Removes integrity penalty and veto |
| `no_utility` | Distance-only weighting, without rare utility or sign-stability weighting |
| `no_uncertainty` | Mean influence without standard-deviation penalty or certainty weighting |
| `no_tail` | Removes rare CVaR while retaining ordinary audited server refinement |

`class_balanced_trust` specifies a separate baseline through its actual objective; it does not substitute a differently defined algorithm under the paper's CBR-Agg name. All methods share the clipping protocol. FLTrust's reference is one gradient step on the declared resource, with class-balanced weighting only for its named variant. Krum requires `n >= 2*f+3` for the configured participating Byzantine budget.

## 4. Attack and sensitivity studies

`--attack` supports `sign_flip`, `model_replacement`, `label_flip` and `mimicry`. The target is class zero by default; choose another ordinary class with `--attack-target`. Rare labels cannot be attack targets.

- Sign flip reverses and scales the transmitted update.
- Label flip maps malicious clients' rare labels to the target before local training.
- Model replacement trains the target-label model and scales its transmitted update by `--attack-scale` (default 10).
- Mimicry combines rare-loss descent with ordinary-loss ascent, controlled by `--mimic-strength` (default 0.5), then matches the common clipping radius.

Mimicry is a declared stress attack with access to the audit-gradient directions. It provides an explicit overlap condition rather than assuming geometric separation. Malicious membership is fixed per seed; the sampled malicious fraction can vary by round. `malicious_clients` in the final output and `attack_injected` in audit records preserve the injected ground truth.

For the Byzantine stress curve, repeat fractions 0, 0.1, 0.2, 0.3 and 0.4 with the same seed and data. For rarity/partition studies, prepare new machine-grouped manifests while preserving held-out and server-resource boundaries. Repeat five seeds and three independently prepared client partitions. Keep all non-varied budgets and hyperparameters identical across comparisons.

## 5. Read outputs and generate figures

Every run writes `best.pt`, `history.json` and `metrics.json` under `--output`. Checkpoints maximize validation rare AUPRC. Temperature is fit on validation logits by minimizing cross-entropy; test predictions use the selected checkpoint's temperature.

`metrics.json` reports held-out rare AUPRC, Macro-F1, targeted attack success, worst-client error, 15-bin ECE, accuracy, best round, cumulative communication and elapsed training/audit time. Rare AUPRC treats membership in any registered rare label as the positive event and sums their predicted probabilities. Targeted attack success is the fraction of true rare examples predicted as the designated ordinary target. Worst-client error uses the maximum error among test-client groups. Undefined AUPRC is represented by `null` if a split contains only one binary membership class; validation must contain both.

`history.json` records validation metrics and a client-level audit with projected distance, robust-z score, influence mean/standard deviation, conservative utility, sign certainty, eligibility and final weight. Parameter communication counts upload/download, SCAFFOLD controls when applicable, and two scalar metadata fields per participant. Raw recordings and audit examples are not sent during the federated loop.

```bash
python -m rareguard plot --metrics outputs/rareguard_0/metrics.json outputs/trimmed_0/metrics.json outputs/no_tail_0/metrics.json --output outputs/figures
```

Plotting produces primary-metric comparisons, communication curves and `summary.json` grouped by method and Byzantine fraction. Supply repeated seeds from the same dataset/attack/partition convention for mean and sample-standard-deviation bars. Group fractions separately when constructing a stress curve; the JSON preserves those groups.

Source files implement data preparation, the audit solver and capped weights, model/training/evaluation, and CLI/plotting. GitHub Actions checks dependency installation, compilation and imports without launching training.
