# Reproduction Guide

This document describes how to reproduce the PIGRAM pipeline, including data preprocessing, model training, baseline training, evaluation, downstream screening, analysis figures, and grammar-guided design.

All commands assume Windows PowerShell.

---

## 1. Environment setup

From the repository root:

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

## 2. Expected input data

Place raw data here:

```text
data/raw/prot_rna.pkl
data/raw/prot_dna.pkl
```

The inspected raw files contain:

```text
prot_rna.pkl: 1328 samples, split lengths [1062, 133, 133]
prot_dna.pkl: 2900 samples, split lengths [2618, 141, 141]
```

Raw samples contain fields such as:

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

---

## 3. Expected processed data

After preprocessing, the main training files are:

```text
data/processed/train.pkl
data/processed/val.pkl
data/processed/test.pkl
```

The inspected processed splits contain:

```text
train: 3680 samples
val:   289 samples
test:  259 samples
```

Each processed sample should include:

```text
patch_attr: [num_patches, 56]
motif_attr: [num_motifs, 27]
patch_motif_structure_energy_v2: [num_patches, num_motifs]
```

The processed loader check should pass:

```text
train loader_check: 3/3 ok
val loader_check:   3/3 ok
test loader_check:  3/3 ok
```

---

# Main pipeline

## 4. Build pair-disjoint raw splits

Command:

```powershell
python .\scripts\data\split.py `
  --config .\configs\data.yaml
```

Expected outputs:

```text
data/split/prot_rna.pkl
data/split/prot_rna_summary.json
data/split/prot_dna.pkl
data/split/prot_dna_summary.json
```

---

## 5. Build processed patch--motif datasets

Command:

```powershell
python .\scripts\data\build.py `
  --config .\configs\data.yaml
```

Expected outputs:

```text
data/processed/train.pkl
data/processed/val.pkl
data/processed/test.pkl
data/processed/summary.json
```

The processed files should contain:

```text
protein patches
nucleic-acid motifs
secondary-structure annotations
binary patch--motif labels
structure-aware patch--motif labels
patch_attr: [num_patches, 56]
motif_attr: [num_motifs, 27]
```

---

## 6. Optional: rebuild explicit attributes

Command:

```powershell
python .\scripts\data\attr.py `
  --config .\configs\data.yaml
```

Expected output:

```text
data/processed/attr_summary.json
```

---

## 7. Check processed data schema

Command:

```powershell
python .\scripts\data\check.py `
  --config .\configs\data.yaml
```

Expected output:

```text
data/processed/schema_check.json
```

Expected result:

```text
status = ok
patch_attr_dim = 56
motif_attr_dim = 27
```

---

# Model training

## 8. Train the coarse grammar teacher

Command:

```powershell
python .\scripts\train\coarse.py `
  --config .\configs\coarse.yaml
```

Expected outputs:

```text
runs/coarse/best.pt
runs/coarse/last.pt
runs/coarse/history.json
runs/coarse/summary.json
```

---

## 9. Build fine pseudo-labels

Command:

```powershell
python .\scripts\train\label.py `
  --config .\configs\label.yaml
```

Expected outputs:

```text
results/grammar/distill_plan.json
results/grammar/train_fine_pseudolabels.pkl
results/grammar/val_fine_pseudolabels.pkl
results/grammar/test_fine_pseudolabels.pkl
results/grammar/summary.json
```

Important:

```text
Annotation files should align with the full processed pkl files.
Subsampling should be done during training, not during annotation generation.
```

---

## 10. Train the coarse-to-fine grammar foundation

Command:

```powershell
python .\scripts\train\grammar.py `
  --config .\configs\grammar.yaml
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

## 11. Train final PIGRAM

Command:

```powershell
python .\scripts\train\pigram.py `
  --config .\configs\pigram.yaml
```

The final score is:

```text
S_final = H_A + r * Delta_B
```

where:

```text
H_A      = frozen coarse-to-fine grammar foundation
Delta_B  = residual Transformer correction
r        = budgeted sparse rescue router
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

# Baseline training

## 12. Train single-stage baselines

Command:

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

Expected outputs:

```text
runs/baselines/concat_mlp/best.pt
runs/baselines/bilinear/best.pt
runs/baselines/transformer/best.pt
runs/baselines/prototype/best.pt
runs/baselines/suite_summary.json
```

---

## 13. Train Prototype + residual baseline

Command:

```powershell
python .\scripts\train\proto_res.py `
  --config .\configs\baseline.yaml
```

Expected outputs:

```text
runs/baselines/proto_res/best.pt
runs/baselines/proto_res/last.pt
runs/baselines/proto_res/history.json
runs/baselines/proto_res/summary.json
```

---

# Evaluation

## 14. Evaluate final PIGRAM on validation/test splits

Command:

```powershell
python .\scripts\eval\main.py `
  --config .\configs\eval.yaml
```

Expected output:

```text
results/main/metrics.json
```

---

## 15. Attribute-shuffle ablation

Command:

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
struct_auroc and struct_auprc are computed by thresholding
patch_motif_structure_energy_v2 >= binary_threshold.
They are not directly comparable to the main benchmark binary AUPRC.
```

---

# Downstream screening

## 16. Preprocess downstream FAFormer candidate pools

Command:

```powershell
python .\scripts\data\downstream.py `
  --config .\configs\downstream.yaml
```

Important:

```text
The original FAFormer validation/test split and candidate pool are preserved.
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

## 17. Evaluate PIGRAM downstream readouts

Command:

```powershell
python .\scripts\eval\downstream.py `
  --config .\configs\downstream.yaml
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

## 18. Evaluate downstream baselines

Command:

```powershell
python .\scripts\eval\baseline.py `
  --config .\configs\downstream.yaml
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

# Analysis figures

## 19. Route-atlas analysis

Command:

```powershell
python .\scripts\analysis\route.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/route/rescue_route_evidence_summary.json
results/route/test/full_rescue_route_atlas.json
results/route/test/route_signature_table.csv
results/route/test/route_by_coarse_rule.csv
results/route/test/route_by_fine_subtype.csv
results/route/test/top_high_route_pairs.csv
results/route/test/top_rescued_pairs.csv
```

---

## 20. Route-atlas figures

Command:

```powershell
python .\scripts\analysis\route_fig.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/figures/route/main_route_atlas_dualy_with_error.png
results/figures/route/main_route_atlas_dualy_with_error.pdf
results/figures/route/appendix_fine_dualy_support_route_grammar.png
results/figures/route/appendix_fine_dualy_support_route_grammar.pdf
```

---

## 21. Fine-refinement composition figure

Command:

```powershell
python .\scripts\analysis\fine_fig.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/figures/fine/single_fine_refinement_composition.png
results/figures/fine/single_fine_refinement_composition.pdf
```

---

## 22. Dense matrix case study

Command:

```powershell
python .\scripts\analysis\case.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/figures/case/
```

---

## 23. Structure projection

Export PyMOL scripts:

```powershell
python .\scripts\analysis\pymol.py `
  --config .\configs\analysis.yaml
```

Then render generated PML scripts using PyMOL. Each case folder contains a generated render script.

Assemble the final structure projection panel:

```powershell
python .\scripts\analysis\structure_fig.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/structure/pymol/
results/figures/structure/structure_projection.png
results/figures/structure/structure_projection.pdf
```

---

## 24. Attribute-level route signature

Command:

```powershell
python .\scripts\analysis\attr_route.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/figures/attr_route/
```

---

## 25. Rule--attribute alignment

Command:

```powershell
python .\scripts\analysis\rule_attr.py `
  --config .\configs\analysis.yaml
```

Expected outputs:

```text
results/figures/rule_attr/
```

---

# Grammar-guided design

## 26. All-target design scan summary

Command:

```powershell
python .\scripts\design\scan.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/all_targets/summary/
```

---

## 27. GFP suppressive-context mutation design

Command:

```powershell
python .\scripts\design\gfp.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/gfp/mutation_designs/
```

---

## 28. GFP RNAfold validation

Command:

```powershell
python .\scripts\design\gfp_fold.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/gfp/mutation_designs/rnafold_validation/
```

---

## 29. GFP final oligo ordering table

Command:

```powershell
python .\scripts\design\gfp_order.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/gfp/mutation_designs/final_oligo_ordering/
```

---

## 30. NELF favorable Rule-4 design

Command:

```powershell
python .\scripts\design\nelf.py `
  --config .\configs\design.yaml
```

Expected outputs:

```text
results/design/nelf/mutation_designs/
```

---

