# Automated Corpus Generation for Tropical Semantic Geometry (Aim 3)

### 1. Objective and Theoretical Grounding
The objective of this phase is to construct a specialized, highly controlled synthetic dataset to evaluate semantic geometry through the triangle inequality. In the enriched category theory framework, linguistic distance is defined as the negative logarithm of the conditional probability: $d(x,y) = -\ln \pi(y \mid x)$. By applying this negative logarithm, the probability-enriched syntax category transforms into a generalized metric space (tropical/min-+ algebra). This dataset is designed to test whether the internal geometry of feedforward neural networks (which compute tropical rational maps via ReLU activations) strictly obeys the categorical triangle inequality, or if its semantic space fundamentally warps under shifting contexts.  

### 2. The Topological Taxonomy (Data Structures)
To rigorously test the tree-like manifolds predicted by tropical geometry, the data generation pipeline must classify and generate text sequences across three distinct structural types:Type A: Strict Phylogenetic Chains (Tree Metrics):Concept: Pure hierarchical semantic chains (e.g., Entity $\rightarrow$ Category $\rightarrow$ Specific $\rightarrow$ Instance).Purpose: To establish a baseline for additive distances in a tropical metric space.Example: "Animal" $\rightarrow$ "Mammal" $\rightarrow$ "Dog" $\rightarrow$ "Poodle".Type B: Sudden Contextual Shifts (Surprisal):Concept: Text sequences that contain sudden contextual or topical shifts.  Purpose: To map theoretical distances against the actual cosine similarities of LLM hidden state vectors and test if the triangle inequality holds when the semantic trajectory is violently disrupted.  Example: "The surgeon carefully picked up the scalpel, and then suddenly started juggling glowing chainsaws."Type C: Adversarial Functors (Contextual Distortions):Concept: Standard logical entailments placed inside extreme, reality-distorting framing.Purpose: To test the topological resilience and rigidity of the semantic space against extreme contextual variables.Example: "In a surreal dream sequence on Mars, a poodle is barking." (Testing if the distance between Poodle and Dog remains rigid).

### 3. Infrastructure and Tooling

Generator API: SURF AI Hub Pilot (for LLM API calls utilizing state-of-the-art instruction-tuned models for synthetic data curation).

Output Format: Standardized JSONL or JSON Array.

Environment: Python 3.10+, utilizing asynchronous API calls and a batch-looping script to prevent output degradation and token cut-offs.

### 4. Implementation Specifications (Meta-Prompting)

The generation script (generate_aim3_corpus.py) must implement a loop over the 3 structural types to ensure diverse topological trajectories.

Meta-Prompt for the SURF API:

    "You are an expert computational linguist building a dataset for testing the tropical geometry and topological properties of Large Language Models. Your task is to generate 50 unique text structures that strictly exhibit the following topological relationship: [INSERT_TYPE_NAME].
    Definition: [INSERT_TYPE_DESCRIPTION].
    Ensure the examples are highly diverse and do not repeat. Output NOTHING ELSE but a valid JSON array of objects."

5. Expected JSON Schema

The script must parse the API response and append it to the local dataset (data/tropical_geometry_corpus.json) in the following exact format:

```json
[
  {
    "structure_type": "Type A: Strict Phylogenetic Chains",
    "node_1": "Vehicle",
    "node_2": "Car",
    "node_3": "Sports Car",
    "node_4": "Porsche 911",
    "context_sequence": "A vehicle is moving. It is a car. Specifically, a sports car. It is a Porsche 911."
  },
  {
    "structure_type": "Type B: Sudden Contextual Shifts",
    "base_context": "The accountant was reviewing the quarterly financial spreadsheet.",
    "shift_context": "Suddenly, the spreadsheet turned into a portal to a dimension of molten lava.",
    "combined_sequence": "The accountant was reviewing the quarterly financial spreadsheet. Suddenly, the spreadsheet turned into a portal to a dimension of molten lava."
  }
]
```
(Note: The script must include a deduplication filter to ensure absolute uniqueness across the dataset.)