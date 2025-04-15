
# Neuron 373 Interpretability Experiments (GPT-2 Small)

This repository documents a series of interpretability experiments centered on **Neuron 373 in Layer 11** of GPT-2 Small, across both the 768-dimensional residual stream (`resid_post`) and the 3072-dimensional MLP output space (`hook_post`).

---

## 🧠 Primary Goal

To examine how clamping and projection affect:

- Neuron alignment and co-activation
- SRM (Spotlight Resonance Mapping) patterns
- Directional drift and latent field rotation
- Interpretability across dimensional boundaries

---

## 🧪 Experiment Categories

### 1. **MLP-Space SRM Spotlight Analysis**

These experiments use cosine projections in the 373+2202 plane of the 3072D MLP space, comparing clamped vs unclamped conditions.

- 👉 [`docu/srm_multi_threshold_sweep_v3_v4_FINAL.html`](https://apocryphaleditor.github.io/holding/docu/srm_multi_threshold_sweep_v3_v4_FINAL.html)  
  Final multi-threshold SRM sweep results and interpretation (with embedded plots).

- 👉 [`docu/srm_results_baseline_comparison_373_2202.html`](https://apocryphaleditor.github.io/holding/docu/srm_results_baseline_comparison_373_2202.html)  
  Earlier summary showing baseline vs intervention comparison in simpler form.

- 👉 [`docu/baseline_capture_spec_zero_intervention.html`](https://apocryphaleditor.github.io/holding/docu/baseline_capture_spec_zero_intervention.html)  
  Experimental spec for how the baseline MLP post-activation vectors were collected.

---

### 2. **Dimensionality & Projection Distortion**

These documents trace the difference in interpretability between residual-stream and native MLP representations.

- 👉 [`docu/averaging_justification_3072d.html`](https://apocryphaleditor.github.io/holding/docu/averaging_justification_3072d.html)  
  Justifies using averaged `[tokens, 3072] → [3072]` mean vectors for SRM comparisons.

- 👉 [`docu/neuron_373_768d_pairings_summary.html`](https://apocryphaleditor.github.io/holding/docu/neuron_373_768d_pairings_summary.html)  
  Residual stream-based pairing experiment, identifies top co-activators and antagonists.

- 👉 [`docu/neuron_373_correlation_comparison_768d_vs_3072d.html`](https://apocryphaleditor.github.io/holding/docu/neuron_373_correlation_comparison_768d_vs_3072d.html)  
  Shows projection distortion when comparing neuron similarity rankings in 768D vs 3072D.

---

## 🗃️ Code & Data

### Code
All scripts used in SRM analysis and vector capture are in `/code/`:

- `capture_baseline_mlp_post.py` — captures [tokens, 3072] activations with no intervention  
- `run_srm_sweep.py` — original SRM sweeper (single threshold)  
- `run_srm_sweep_multi_threshold.py` — multi-threshold sweep analyzer with plotting

### Prompts
The prompts used for generation and activation capture:

- `promptsv3.txt` — general reasoning and syntax tasks  
- `promptsv4.txt` — phrases from OpenAI Neuron Viewer for Neuron 373

---

## 🌀 Reproducibility Notes

All results were generated with:

- Model: `gpt2` via `transformer_lens`
- Hook point: `blocks.11.mlp.hook_post`
- Dimension: 3072D (MLP space)
- Generation length: 50 tokens
- Output: mean activation vectors per prompt
- SRM plane: Neuron 373 + Neuron 2202
- Angular sweep: 360 degrees, 1° steps
- Thresholds: 0.7, 0.5, 0.3, 0.1

---

## 📬 Contact

Maintained by [@ApocryphalEditor](https://github.com/ApocryphalEditor)

Feel free to fork, test, or contribute ideas and interpretations.
