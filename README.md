
# Neuron 373 Interpretability Experiments (GPT-2 Small)

This repository contains documentation from a series of experiments focused on Neuron 373 in Layer 11 of GPT-2 Small. The work explores how this neuron's behavior manifests across different internal representation spaces — specifically the 768-dimensional residual stream (`resid_post`) and the 3072-dimensional MLP post-activation space (`hook_post`).

## 🧪 Ongoing Goal

To understand how architectural decisions (e.g. projection from MLP to residual stream) affect interpretability outcomes like:
- Neuron alignment
- SRM pairing
- Conceptual drift
- Co-activation structure

---

## 📁 File Overview

### [`averaging_justification_3072d.html`](./averaging_justification_3072d.html)
Explains why we average `[tokens, 3072]` activations into `[3072]` mean vectors per run. Clarifies that this does not distort dimensionality, and is valid for correlation comparisons against previous 768D work.

### [`neuron_373_768d_pairings_summary.html`](./neuron_373_768d_pairings_summary.html)
Summarizes the original 768D residual stream experiment, which identified co-activators and antagonists of Neuron 373 using projected vectors. Used as the baseline for comparing native-space results.

### [`neuron_373_correlation_comparison_768d_vs_3072d.html`](./neuron_373_correlation_comparison_768d_vs_3072d.html)
Documents a key intermediate finding: comparing 373’s top aligned neurons in 768D vs 3072D reveals discrepancies caused by projection distortion. Establishes the need for MLP-native SRM analysis.

---

## 🔄 Status

This folder is a **holding space** for interpretability results in-progress — including writeups, analysis artifacts, and correlation comparisons. It may evolve or be folded into a larger repo later.

Suggestions, replications, and forks welcome.

---

## 📬 Contact

Maintained by [@ApocryphalEditor](https://github.com/ApocryphalEditor)
