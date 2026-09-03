# Contextual Representation Extraction via Middle-Layer Logit Lens

Package: `logit_lens_aim1_AND` (sibling of `output_logit_aim1_AND`).

### Objective
Same Aim-1 composition analysis as `output_logit_aim1_AND`, but probability
distributions are read from a **middle transformer layer** with the classic
logit lens (optional final RMSNorm → `lm_head` → softmax → Top-K), instead of
the model's final output logits.

### Protocol
For each corpus pair `(word_a, word_b)`:
1. Build the same typicality prompts `P_A`, `P_B`, `P_AB`
2. Forward pass with `output_hidden_states=True`
3. Take the residual stream after transformer block `layer_index` at the final token
4. Apply logit lens → Top-K (`K=4096` by default)
5. Compare pointwise **min / mean / max** of `(p_A, p_B)` to `p_AB`
6. Plot rank-aligned absolute difference curves per category

### Layer selection
- Default: middle layer = `num_hidden_layers // 2` (resolved after model load)
- Override in the notebook with `LAYER_INDEX = <int>`, via
  `run_experiment(layer_index=...)`, or env `LOGIT_LENS_LAYER`

Outputs are written under `output/layer_{L}/` so different layers do not collide.
