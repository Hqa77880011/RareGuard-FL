# RareGuard-FL

Implementation of **RareGuard-FL: Counterfactual Client-Influence Auditing for Rare-Failure-Preserving Robust Federated Maintenance**.

## Installation

Use Python 3.11 or 3.12 for installation. Run from the repository root in an activated environment.

```bash
python -m pip install -r requirements.txt
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, scikit-learn 1.6.1, matplotlib 3.10.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

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

## Training and inference

```bash
python -m rareguard train --data data/mimii.npz --output outputs/rareguard_0 --rare-labels 1 --method rareguard --attack mimicry --byzantine 0.2 --seed 0
```

Keep enough eligible clients for the aggregation cap (`eligible_clients * maximum_weight >= 1`).

## Evaluation

Training selects `best.pt` on validation rare AUPRC, fits validation temperature and evaluates the held-out test split.

## Comparisons and ablations

```bash
python -m rareguard train --data data/mimii.npz --output outputs/trimmed_0 --rare-labels 1 --method trimmed --attack mimicry --seed 0
python -m rareguard train --data data/mimii.npz --output outputs/no_tail_0 --rare-labels 1 --method no_tail --attack mimicry --seed 0
```

Further methods: `fedavg`, `fedprox`, `scaffold`, `krum`, `fltrust`, `class_balanced_trust`, `no_integrity`, `no_utility`, `no_uncertainty`. Attacks: `sign_flip`, `model_replacement`, `label_flip`, `mimicry`. Keep data, seed, attack and budgets identical.

## Outputs

`best.pt` retains the selected model; `history.json` records rounds and client audit weights; `metrics.json` reports test rare AUPRC, Macro-F1, attack success, calibration and communication/time.

```bash
python -m rareguard plot --metrics outputs/rareguard_0/metrics.json outputs/trimmed_0/metrics.json outputs/no_tail_0/metrics.json --output outputs/figures
```
