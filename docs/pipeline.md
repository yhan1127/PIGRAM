# PIGRAM Pipeline

This document summarizes the full PIGRAM pipeline from raw structural data to
processed patch--motif samples, model training, evaluation, downstream
screening, analysis figures, and grammar-guided design.

---

## Overview

PIGRAM is an interpretable patch--motif grammar model for protein--nucleic-acid recognition.

The pipeline has four core model stages:

```text
raw structural complexes
→ processed patch--motif samples
→ coarse grammar teacher
→ coarse-to-fine grammar foundation
→ sparse residual-rescue PIGRAM
```

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

The core training and evaluation code lives in:

```text
src/pigram/
scripts/data/
scripts/train/
scripts/eval/
```

---

## Stage 0: Environment setup

From the repository root:

```powershell
cd PIGRAM
conda activate pigram
$env:PYTHONPATH = "$PWD\src"
```

Core package:

```text
src/pigram/
```

Command-line wrappers:

```text
scripts/
```

---

## Stage 1: Raw structural data

Input files:

```text
data/raw/prot_rna.pkl
data/raw/prot_dna.pkl
```

The inspected raw files have the following structure:

```text
prot_rna.pkl: 1328 samples, split lengths [1062, 133, 133]
prot_dna.pkl: 2900 samples, split lengths [2618, 141, 141]
```

Each raw sample is a dictionary containing structural and sequence fields such as:

```text
pdb
kind
protein_seq
protein_coords
partner_seq
partner_coords
contact_map
affinity
```

Coordinate arrays use the structural format:

```text
protein_coords: [num_residues, 14, 3]
partner_coords: [num_nucleotides, 14, 3]
contact_map: [num_residues, num_nucleotides]
```

---

## Stage 2: Pair-disjoint split construction

Command:

```powershell
python .\scripts\data\split.py `
  --config .\configs\data.yaml
```

Purpose:

```text
Build pair-disjoint train/validation/test splits from raw protein--nucleic-acid complexes.
```

Expected outputs:

```text
data/split/prot_rna.pkl
data/split/prot_rna_summary.json
data/split/prot_dna.pkl
data/split/prot_dna_summary.json
```

---

## Stage 3: Patch--motif processed dataset construction

Command:

```powershell
python .\scripts\data\build.py `
  --config .\configs\data.yaml
```

Purpose:

```text
Convert raw structural complexes into processed patch--motif samples.
```

This stage builds:

```text
protein patches
single-stranded nucleic-acid motifs
secondary-structure annotations
binary patch--motif labels
continuous structure-aware patch--motif labels
56D protein patch attributes
27D nucleic-acid motif attributes
```

Expected outputs:

```text
data/processed/train.pkl
data/processed/val.pkl
data/processed/test.pkl
data/processed/summary.json
```

The inspected processed split sizes are:

```text
train: 3680 samples
val:   289 samples
test:  259 samples
```

The main model-facing processed fields include:

```text
patch_attr: [num_patches, 56]
motif_attr: [num_motifs, 27]
patch_motif_structure_energy_v2: [num_patches, num_motifs]
```

---

## Stage 4: Optional explicit-attribute rebuild

Command:

```powershell
python .\scripts\data\attr.py `
  --config .\configs\data.yaml
```

Purpose:

```text
Rebuild explicit patch_attr and motif_attr fields for already processed data.
```

Expected output:

```text
data/processed/attr_summary.json
```

---

## Stage 5: Processed schema validation

Command:

```powershell
python .\scripts\data\check.py `
  --config .\configs\data.yaml
```

Purpose:

```text
Verify that processed train/val/test pkl files contain the fields required by the training pipeline.
```

Expected result:

```text
status = ok
patch_attr_dim = 56
motif_attr_dim = 27
```

The inspected loader checks passed:

```text
train loader_check: 3/3 ok
val loader_check:   3/3 ok
test loader_check:  3/3 ok
```

---

## Stage 6: Coarse grammar teacher training

Command:

```powershell
python .\scripts\train\coarse.py `
  --config .\configs\coarse.yaml
```

Purpose:

```text
Train the sign-separated coarse patch--motif grammar teacher.
```

The coarse teacher learns:

```text
suppressive coarse rules
favorable coarse rules
patch-importance head
stage-2 fallback correction branch
```

Expected outputs:

```text
runs/coarse/best.pt
runs/coarse/last.pt
runs/coarse/history.json
runs/coarse/summary.json
```

---

## Stage 7: Fine pseudo-label generation

Command:

```powershell
python .\scripts\train\label.py `
  --config .\configs\label.yaml
```

Purpose:

```text
Use the coarse teacher to assign dominant coarse rules and build fine pseudo-labels
inside selected coarse-rule families.
```

Expected outputs:

```text
results/grammar/distill_plan.json
results/grammar/train_fine_pseudolabels.pkl
results/grammar/val_fine_pseudolabels.pkl
results/grammar/test_fine_pseudolabels.pkl
results/grammar/summary.json
```

Each fine pseudo-label annotation contains:

```text
fine_label_matrix
distill_mask_matrix
coarse_rule_matrix
```

Important:

```text
Annotation files should align with the full processed pkl files.
Subsampling should be done during training, not during annotation generation.
```

---

## Stage 8: Coarse-to-fine grammar foundation training

Command:

```powershell
python .\scripts\train\grammar.py `
  --config .\configs\grammar.yaml
```

Purpose:

```text
Train the coarse-to-fine grammar foundation using fine pseudo-labels.
```

The grammar foundation learns:

```text
coarse signed rule families
fine subtypes inside selected coarse rules
grammar-centered patch--motif prediction
```

Expected outputs:

```text
runs/grammar/best.pt
runs/grammar/last.pt
runs/grammar/history.json
runs/grammar/summary.json
runs/grammar/fine_rule_map.json
```

---

## Stage 9: Final PIGRAM residual-rescue training

Command:

```powershell
python .\scripts\train\pigram.py `
  --config .\configs\pigram.yaml
```

Purpose:

```text
Freeze the grammar foundation and train only the sparse residual-rescue branch.
```

Final model:

```text
S_final = H_A + r * Delta_B
```

where:

```text
H_A      = frozen grammar prediction
Delta_B  = residual Transformer correction
r        = sparse rescue router
```

Expected outputs:

```text
runs/pigram/best_by_composite.pt
runs/pigram/best_by_pearson.pt
runs/pigram/last.pt
runs/pigram/history.json
runs/pigram/summary.json
```

---

## Stage 10: Baseline training

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

Expected outputs:

```text
runs/baselines/concat_mlp/best.pt
runs/baselines/bilinear/best.pt
runs/baselines/transformer/best.pt
runs/baselines/prototype/best.pt
runs/baselines/proto_res/best.pt
runs/baselines/suite_summary.json
```

---

## Stage 11: Main benchmark evaluation

Command:

```powershell
python .\scripts\eval\main.py `
  --config .\configs\eval.yaml
```

Purpose:

```text
Evaluate final PIGRAM on validation and test splits.
```

Expected output:

```text
results/main/metrics.json
```

Main metrics include:

```text
energy_pearson
energy_spearman
binary_auroc
binary_auprc
head_a_only_pearson
grammar_only_pearson
route statistics
hard/easy pair rescue metrics
```

---

## Stage 12: Attribute-shuffle ablation

Command:

```powershell
python .\scripts\eval\shuffle.py `
  --config .\configs\eval.yaml
```

Purpose:

```text
Measure sensitivity to explicit patch-side and motif-side attributes.
```

Shuffle modes:

```text
original
patch
motif
both
```

Expected output:

```text
results/shuffle/attribute_shuffle.json
```

Note:

```text
struct_auroc and struct_auprc use patch_motif_structure_energy_v2 thresholding.
They are not directly comparable to the main benchmark binary AUPRC.
```

---

## Stage 13: Downstream candidate preprocessing

Command:

```powershell
python .\scripts\data\downstream.py `
  --config .\configs\downstream.yaml
```

Purpose:

```text
Convert downstream FAFormer candidate pools into PIGRAM-compatible samples.
```

Important:

```text
The original downstream candidate pools and validation/test split are preserved.
No pair-disjoint resplitting is applied.
```

Expected outputs:

```text
data/downstream_processed/manifest.json
data/downstream_processed/index.csv
data/downstream_processed/summary.csv
data/downstream_processed/*_downstream_val_processed.pkl
data/downstream_processed/*_downstream_test_processed.pkl
```

---

## Stage 14: Downstream PIGRAM evaluation

Command:

```powershell
python .\scripts\eval\downstream.py `
  --config .\configs\downstream.yaml
```

Purpose:

```text
Evaluate PIGRAM readouts on downstream candidate pools.
```

Readouts include:

```text
final S_final
grammar H_A
coarse grammar G
route-weighted final score
residual correction summaries
top-K aggregation scores
```

Expected outputs:

```text
results/downstream/pigram/scores.csv
results/downstream/pigram/metrics.csv
results/downstream/pigram/selected_readouts.csv
results/downstream/pigram/selected_test.csv
results/downstream/pigram/summary.json
```

---

## Stage 15: Downstream baseline evaluation

Command:

```powershell
python .\scripts\eval\baseline.py `
  --config .\configs\downstream.yaml
```

Purpose:

```text
Compare PIGRAM against internal baselines on the same downstream samples.
```

Expected outputs:

```text
results/downstream/baselines/scores.csv
results/downstream/baselines/metrics.csv
results/downstream/baselines/selected_readouts.csv
results/downstream/baselines/selected_test.csv
results/downstream/baselines/summary.json
```

---

## Stage 16: Analysis figures

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

Attribute-level route signature:

```powershell
python .\scripts\analysis\attr_route.py `
  --config .\configs\analysis.yaml
```

Rule--attribute alignment:

```powershell
python .\scripts\analysis\rule_attr.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/route/
results/figures/
results/structure/
```

---

## Stage 17: Grammar-guided design

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





