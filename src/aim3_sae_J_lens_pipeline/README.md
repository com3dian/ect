# Phase 4: Aim 3 - Evaluating Tropical Metric Geometry in J-Space via SAE and J-Lens

## 1. Objective and Theoretical Grounding
The objective of this phase is to evaluate the metric geometry of the global workspace (J-space) in Large Language Models[cite: 1]. Enriched Category Theory (ECT) predicts that the semantic metric space of language is not a smooth Euclidean space, but a generalized metric space governed by tropical (min-+) algebra[cite: 3, 11]. 

Instead of using traditional cosine similarity, we will measure semantic distances using the negative log probability: $d_M(x,y) = -\ln \pi(y|x)$[cite: 1, 11]. By tracking the trajectories of monosemantic features during contextual shifts, we aim to verify if the LLM's latent space strictly obeys the Tropical Triangle Inequality[cite: 1]. If the trajectories form branching structures known as "Tree Metrics" or "Tropical Polytopes", it physically proves that the neural network organizes concepts using tropical geometry[cite: 1, 3].

## 2. Infrastructure & Tooling
*   **Model:** `google/gemma-2-2b`[cite: 8].
*   **Monosemantic Feature Extraction:** `SAELens` combined with DeepMind's official `Gemma Scope` (e.g., Layer 14 SAE)[cite: 8]. This provides high-quality, pre-trained sparse dictionaries without the need to train them from scratch[cite: 1, 8].
*   **Causal Mapping (The J-Lens):** The Jacobian Lens ($J_\ell$) will be used to extract the average linear causal effect, acting as an inter-layer translator that maps intermediate features to the final output vocabulary[cite: 5, 6].

## 3. The Decoupled Engineering Pipeline
Because computing the Jacobian matrix is highly VRAM-intensive, Aim 3 perfectly fits the "Offline Extraction and Interactive Analysis" (Decoupled) paradigm[cite: 9]. We will separate the GPU-heavy extraction from the CPU-friendly topological math[cite: 9].

### Phase 3.1: Online Feature & Jacobian Extraction (GPU)
1.  **Forward Pass:** Feed the generated Aim 3 corpus (Type A, B, and C structures) into Gemma 2B[cite: 1, 9].
2.  **SAE Hooking:** Use SAELens to extract the highly sparse, monosemantic feature activations for the target concepts in the residual stream[cite: 1, 8].
3.  **Jacobian Computation:** Compute the Jacobian matrix $J_\ell = \mathbb{E} \left[ \frac{\partial h_L}{\partial h_\ell} \right]$ to map the causal effect of layer $\ell$ on the final layer $L$[cite: 1, 5]. 
4.  **Save and Clear:** Save the extracted SAE vectors and the Jacobian matrices to disk using the `.safetensors` format, then completely clear the GPU VRAM[cite: 7, 9].

### Phase 3.2: Offline Topological Measurement (CPU/RAM)
1.  **Abandon Euclidean Metrics:** Do not use Euclidean distance or Cosine Similarity to measure the distances between the extracted J-space features[cite: 1].
2.  **Tropical Distance Calculation:** Use the saved tensors to compute the categorical theoretical distance $d_M(x,y) = -\ln \pi(y|x)$[cite: 1, 11].
3.  **Phylogenetic Tree Construction:** Run phylogenetic tree-building algorithms, such as Neighbor-Joining, on the tropical distance matrix[cite: 1]. The goal is to see if the semantic concepts perfectly reconstruct a "Phylogenetic Tree" of human logic[cite: 1, 3].
4.  **Triangle Inequality Verification:** Observe the feature trajectories during the Type B (Sudden Contextual Shifts) datasets[cite: 1]. Verify if the distance geometry branches into a Tropical Polytope, maintaining the rigidity of the tropical triangle inequality even under contextual disruption[cite: 1].

## 4. Instructions for Cursor (Implementation Steps)
*   [ ] Initialize `google/gemma-2-2b` and load the Layer 14 SAE via `SAELens`.
*   [ ] Write a script to iterate through the `tropical_geometry_corpus.json` dataset.
*   [ ] Implement the Jacobian extraction function to compute $J_\ell$[cite: 1]. To prevent OOM, implement a Top-K truncation filter to isolate the core semantic manifold and eliminate Softmax long-tail noise[cite: 10].
*   [ ] Save the SAE feature activations and the truncated Jacobian matrices to a local `.safetensors` file[cite: 9].
*   [ ] Create a separate offline analysis script (e.g., `analyze_tropical_topology.py`) that loads the tensors from disk[cite: 9].
*   [ ] Implement the $d_M(x,y) = -\ln \pi(y|x)$ distance metric and integrate a standard Neighbor-Joining algorithm (via `SciPy` or `scikit-bio`)[cite: 1].
*   [ ] Output the resulting tree metrics and log any violations of the Tropical Triangle Inequality during contextual shifts[cite: 1].