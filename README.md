# PIGRAM

PIGRAM is an interpretable patch--motif grammar model for protein--nucleic-acid recognition.

PIGRAM represents proteins as local 3D residue patches and single-stranded nucleic-acid partners as secondary-structure motifs. It learns an explicit coarse-to-fine grammar over patch-side and motif-side attributes, then applies a sparse residual-rescue branch to correct grammar-hard patch--motif pairs.

The final PIGRAM score is:

```text
S_final = H_A + r * Delta_B
```

where:

```text
H_A      = frozen coarse-to-fine grammar foundation
Delta_B  = residual Transformer correction
r        = budgeted sparse rescue router
```

---

## Repository structure

```text
PIGRAM/
├── configs/
│   ├── data.yaml
│   ├── coarse.yaml
│   ├── label.yaml
│   ├── grammar.yaml
│   ├── pigram.yaml
│   ├── baseline.yaml
│   ├── eval.yaml
│   ├── downstream.yaml
│   ├── analysis.yaml
│   ├── design.yaml
│   ├── tiny_baseline.yaml
│   └── tiny_coarse.yaml
│
├── src/
│   └── pigram/
│       ├── data/
│       ├── models/
│       ├── losses/
│       ├── train/
│       ├── eval/
│       ├── analysis/
│       ├── design/
│       └── utils/
│
├── scripts/
│   ├── data/
│   ├── train/
│   ├── eval/
│   ├── analysis/
│   ├── design/
│   └── tests/
│
├── data/
│   └── example/
│       ├── train_tiny.pkl
│       ├── val_tiny.pkl
│       ├── test_tiny.pkl
│       └── summary_tiny.json
│
└── docs/
    ├── reproduce.md
    ├── pipeline.md
    └── data_format.md
```

Generated directories such as `runs/`, `results/`, `outputs/`, and full data directories are intentionally excluded from the supplementary package.

---

## Installation

All commands below assume Windows PowerShell.

Create the environment:

```powershell
cd PIGRAM
conda env create -f environment.yml
conda activate pigram
$env:PYTHONPATH = "$PWD\src"
```

If the environment already exists:

```powershell
cd PIGRAM
conda activate pigram
$env:PYTHONPATH = "$PWD\src"
```

---

## Data included in this supplementary package

The supplementary package includes a small processed example dataset for smoke tests:

```text
data/example/train_tiny.pkl
data/example/val_tiny.pkl
data/example/test_tiny.pkl
data/example/summary_tiny.json
```

The tiny example data contain:

```text
train_tiny: 80 samples
val_tiny:   10 samples
test_tiny:  10 samples
```

The tiny data are intended to verify that the code, loader, model forward pass, loss computation, and short training loops run correctly. They are not intended to reproduce the full benchmark numbers.

To verify the included tiny data:

```powershell
python .\scripts\tests\smoke_processed_forward.py `
  --pkl .\data\example\train_tiny.pkl `
  --sample_index 0 `
  --device cpu
```

Expected output:

```text
SMOKE TEST PASSED: processed pkl loads and model forward works.
```

---

## Full benchmark data

The full processed benchmark data are not included in the supplementary package due to the 100MB NeurIPS supplementary material limit.

The full processed data used in the paper contain:

```text
train: 3680 samples
val:   289 samples
test:  259 samples
patch_attr_dim = 56
motif_attr_dim = 27
```

Each full processed sample follows the same schema as the tiny example data:

```text
patch_attr: [num_patches, 56]
motif_attr: [num_motifs, 27]
patch_motif_structure_energy_v2: [num_patches, num_motifs]
```

The full data are needed to reproduce the exact benchmark numbers in the paper. The tiny data included here are only for code-execution and pipeline smoke tests.

For the full data schema, see:

```text
docs/data_format.md
```

For the full pipeline description, see:

```text
docs/pipeline.md
```

For the complete command list, see:

```text
docs/reproduce.md
```

---

## Quick start with tiny example data

### 1. Processed forward smoke test

```powershell
python .\scripts\tests\smoke_processed_forward.py `
  --pkl .\data\example\train_tiny.pkl `
  --sample_index 0 `
  --device cpu
```

Expected output:

```text
SMOKE TEST PASSED: processed pkl loads and model forward works.
```

---

### 2. Tiny baseline training

```powershell
python .\scripts\train\baseline.py `
  --config .\configs\tiny_baseline.yaml
```

This runs short one-epoch training on the tiny example data for:

```text
Concat-MLP
Bilinear
Transformer
Prototype
```

Expected outputs:

```text
runs/tiny/baselines/concat_mlp/
runs/tiny/baselines/bilinear/
runs/tiny/baselines/transformer/
runs/tiny/baselines/prototype/
runs/tiny/baselines/suite_summary.json
```

---

### 3. Tiny coarse grammar training

```powershell
python .\scripts\train\coarse.py `
  --config .\configs\tiny_coarse.yaml
```

Expected outputs:

```text
runs/tiny/coarse/best.pt
runs/tiny/coarse/last.pt
runs/tiny/coarse/history.json
runs/tiny/coarse/summary.json
```

---

## Full pipeline overview

The complete PIGRAM training pipeline consists of:

```text
raw structural data
→ processed patch--motif data
→ coarse grammar teacher
→ fine pseudo-label generation
→ coarse-to-fine grammar foundation
→ final sparse residual-rescue PIGRAM
```

The main commands are:

```powershell
python .\scripts\data\split.py `
  --config .\configs\data.yaml

python .\scripts\data\build.py `
  --config .\configs\data.yaml

python .\scripts\data\check.py `
  --config .\configs\data.yaml

python .\scripts\train\coarse.py `
  --config .\configs\coarse.yaml

python .\scripts\train\label.py `
  --config .\configs\label.yaml

python .\scripts\train\grammar.py `
  --config .\configs\grammar.yaml

python .\scripts\train\pigram.py `
  --config .\configs\pigram.yaml
```

Expected full-run checkpoints are:

```text
runs/coarse/best.pt
runs/grammar/best.pt
runs/pigram/best_by_composite.pt
runs/pigram/best_by_pearson.pt
```

The full pipeline requires the full processed benchmark data, which are not included in this supplementary package.

---

## Baselines

Single-stage baselines:

```powershell
python .\scripts\train\baseline.py `
  --config .\configs\baseline.yaml
```

This trains:

```text
Concat-MLP
Bilinear
Transformer
Prototype
```

Prototype + residual baseline:

```powershell
python .\scripts\train\proto_res.py `
  --config .\configs\baseline.yaml
```

Expected full-run outputs:

```text
runs/baselines/concat_mlp/best.pt
runs/baselines/bilinear/best.pt
runs/baselines/transformer/best.pt
runs/baselines/prototype/best.pt
runs/baselines/proto_res/best.pt
runs/baselines/suite_summary.json
```

---

## Evaluation

Main validation/test evaluation:

```powershell
python .\scripts\eval\main.py `
  --config .\configs\eval.yaml
```

Expected output:

```text
results/main/metrics.json
```

Attribute-shuffle ablation:

```powershell
python .\scripts\eval\shuffle.py `
  --config .\configs\eval.yaml
```

Expected output:

```text
results/shuffle/attribute_shuffle.json
```

Note:

```text
The shuffle ablation reports Struct-AUPRC by thresholding
patch_motif_structure_energy_v2. It is not directly comparable to the main
benchmark binary AUPRC.
```

---

## Downstream screening

Downstream candidate preprocessing:

```powershell
python .\scripts\data\downstream.py `
  --config .\configs\downstream.yaml
```

PIGRAM downstream evaluation:

```powershell
python .\scripts\eval\downstream.py `
  --config .\configs\downstream.yaml
```

Downstream baseline evaluation:

```powershell
python .\scripts\eval\baseline.py `
  --config .\configs\downstream.yaml
```

Expected outputs:

```text
results/downstream/pigram/
results/downstream/baselines/
```

Downstream evaluation requires the corresponding downstream processed candidate pools, which are not included in this supplementary package.

---

## Analysis figures

Route atlas:

```powershell
python .\scripts\analysis\route.py `
  --config .\configs\analysis.yaml

python .\scripts\analysis\route_fig.py `
  --config .\configs\analysis.yaml
```

Fine-refinement composition:

```powershell
python .\scripts\analysis\fine_fig.py `
  --config .\configs\analysis.yaml
```

Dense matrix case study:

```powershell
python .\scripts\analysis\case.py `
  --config .\configs\analysis.yaml
```

Structure projection:

```powershell
python .\scripts\analysis\pymol.py `
  --config .\configs\analysis.yaml

python .\scripts\analysis\structure_fig.py `
  --config .\configs\analysis.yaml
```

Attribute analyses:

```powershell
python .\scripts\analysis\attr_route.py `
  --config .\configs\analysis.yaml

python .\scripts\analysis\rule_attr.py `
  --config .\configs\analysis.yaml
```

Expected full-run outputs:

```text
results/route/
results/figures/
results/structure/
```

---

## Grammar-guided design

All-target design summary:

```powershell
python .\scripts\design\scan.py `
  --config .\configs\design.yaml
```

GFP suppressive-context design:

```powershell
python .\scripts\design\gfp.py `
  --config .\configs\design.yaml

python .\scripts\design\gfp_fold.py `
  --config .\configs\design.yaml

python .\scripts\design\gfp_order.py `
  --config .\configs\design.yaml
```

NELF favorable Rule-4 design:

```powershell
python .\scripts\design\nelf.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/
```

---

## Core import test

```powershell
python -c "from pigram.data.dataset import process_single_sample; from pigram.data.loader import prepare_single_sample_tensors; from pigram.models.encoders import ProteinPatchEncoder, MotifEncoder; from pigram.train.baseline import train_baselines; from pigram.train.proto_res import train_proto_res; from pigram.train.coarse import train_coarse; from pigram.train.label import build_fine_labels; from pigram.train.grammar import train_grammar; from pigram.train.pigram import train_pigram; from pigram.eval.main import run_main_eval; from pigram.eval.shuffle import run_shuffle_eval; print('all core imports OK')"
```

---

## Core dependency check

The core training and evaluation code should not depend on old experimental script names.

Run:

```powershell
python -c "import inspect; import pigram.train.baseline as b; import pigram.train.proto_res as pr; import pigram.train.coarse as c; import pigram.train.grammar as g; import pigram.train.pigram as p; import pigram.eval.main as em; import pigram.eval.shuffle as es; import pigram.eval.downstream as ed; import pigram.eval.baseline as eb; src=''.join(inspect.getsource(m) for m in [b,pr,c,g,p,em,es,ed,eb]); banned=['legacy','get_legacy_functions','add_legacy_path','src.training','src.models','train_unified','train_dualhead']; print({x:(x in src) for x in banned})"
```

Expected:

```text
{'legacy': False, 'get_legacy_functions': False, 'add_legacy_path': False, 'src.training': False, 'src.models': False, 'train_unified': False, 'train_dualhead': False}
```

---

## Notes for reviewers

The core training and evaluation pipeline is implemented under:

```text
src/pigram/
scripts/data/
scripts/train/
scripts/eval/
```

The included tiny example data allow reviewers to verify that the code executes and that the main model components, loader, losses, and short training loops are functional.

The tiny data are not intended to reproduce the full benchmark numbers. Full benchmark reproduction requires the full processed benchmark splits described above.

Some paper-figure and wet-lab-design scripts may require additional result files generated from full training runs. These scripts are separated from the core training and evaluation pipeline.

---

## Files intentionally excluded

The following large or generated files are intentionally excluded from the supplementary package:

```text
data/raw/
data/split/
data/processed/
data/downstream_processed/
runs/
results/
outputs/
*.pt
*.pth
*.ckpt
*.npy
*.npz
```

The package only includes the small `data/example/` processed dataset for smoke testing.