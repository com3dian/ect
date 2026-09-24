# ECT + LLMs: Discussion Summary

Background paper: Bradley, Terilla, Vlassopoulos, "An Enriched Category Theory of Language: From Syntax to Semantics" (arXiv:2106.07890). The paper defines a syntax category L enriched over [0,1] (hom objects = conditional probabilities π(y|x)), passes via the Yoneda embedding to a semantic category of [0,1]-valued copresheaves, and shows that logical AND, OR, and implication correspond to categorical products, coproducts, and internal homs. Section 5 recasts this as a generalized (asymmetric) metric space and gestures at tropical geometry and phylogenetics.

---

## Part 1: Review of the original three-aim plan

The project (Aim 1: AND as pointwise limit, Aim 2: entailment via Jacobian-derived steering vectors, Aim 3: tropical triangle inequality / phylogenetic tree structure via SAEs on Gemma-2B) is a real empirical program, but as framed it has some problems a tough reviewer would flag:

- **The math is being used as inspiration, not as a falsifiable hypothesis.** Nothing in the plan specifies what result would count as *disconfirming* ECT specifically, as opposed to disconfirming a particular operationalization of it.
- **Aim 1**: finding "semantic drift" when combining concepts is consistent with ECT, but it's also just what anyone expecting nonlinear feature interactions in neural nets would predict. Not a discriminating test.
- **Aim 2**: the strongest of the three, but the win over a diff-of-means baseline could just mean "Jacobian-based steering beats mean-difference steering," which has nothing to do with category theory. Needs comparison against real baselines (ActAdd, DAS, representation engineering), not a strawman.
- **Aim 3**: interesting instinct to use `-ln π(y|x)` as a distance (this is fine as a *Lawvere metric*, i.e. no symmetry required — consistent with the enriched-category setup), but the jump from "triangle inequality holds" to "tree-like tropical polytope structure" needs an explicit bridge; the classical four-point-condition machinery that links metrics to trees assumes symmetric, additive metrics, and that needs justifying here. "Adversarial Functors" also needs to cash out as an actual functor (map on objects and morphisms) or it's just relabeled OOD prompts.
- **General critique**: language like "targeted drug effect" and "purely logical morphism" reads as narrative dressing on what should be plain quantitative claims. Missing throughout: baselines, sample sizes, effect sizes, generalization across models/scales.

**Bottom line from that pass**: there's a real empirical program buried in here, but it needs to earn the category theory's keep by showing it predicts something a non-categorical account wouldn't, not just redescribe known phenomena in fancier language.

---

## Part 2: The framing question — "how well does ECT work on real LLMs?"

This framing was flagged as a problem in itself, independent of the experiments:

- **Not falsifiable.** No matter what the results show, you can always narrate it as "works reasonably well, with some deviations." A good research question needs an answer that could come back "no."
- **It conflates three different claims**: (1) is the literal mathematical structure *present* in the network, (2) is the framework *useful* for generating good interventions, (3) is it *better* than competing formalisms (DisCoCat, plain vector geometry, causal abstraction). These can have different answers — e.g. ECT might not be literally instantiated in the network but still yield a good approximate steering vector.
- **"Real LLMs" isn't a variable.** It doesn't specify model, scale, or task, so it can't be pinned down as evidence for or against anything general.
- **Invites motte-and-bailey.** The exciting claim ("LLMs literally compute via enriched category theory") and the defensible claim ("we derived a useful vector from a categorical construction") are very different, and a vague top-level question makes it easy to blur them.

**Suggested reframe (plain language version):**

> Which parts of this category-theory picture of language actually show up inside a language model, and which parts are just a nice story we're telling after the fact?

With the three aims translated into plain, falsifiable sub-questions:

- **Aim 1**: "When the model combines two concepts, does it just stack them, or does it create something new?"
- **Aim 2**: "Does our theory-based nudging trick beat the standard nudging tricks people already use?"
- **Aim 3**: "Does the model's sense of 'how far apart two ideas are' form a neat branching tree — and does that fall apart under sudden topic changes or trick inputs?"

Each of these has a real "no" available, which is what makes them actual experiments instead of a story that can't lose.

---

## Part 3: Five new experiment ideas, pulled directly from claims in the paper

These are not "same experiment, different model" — each targets a different specific claim in the paper that Aims 1–3 don't touch.

### 1. Stress-test the foundational composition axiom
The whole framework rests on Equation (8): `π(z|y)·π(y|x) = π(z|x)`. This is what makes L a genuine enriched category (with equalities, not just inequalities). Real LLMs likely violate this due to context truncation, ordering effects, and sampling noise.

**Do this**: for triples x ⊂ y ⊂ z, measure `π(z|x)` directly and compare to `π(y|x)·π(z|y)`, across many triples, varying distance between them and whether there's a topic shift in between.

**Payoff**: if violations spike specifically at discourse/topic boundaries, that's a principled, theory-derived discontinuity detector — not just "does ECT hold," but a new usable signal. Cheap: only needs log-probs, no internals.

### 2. Directly test the Yoneda embedding identity
Corollary 1 makes an exact checkable claim: `L(y,x) = L̂(hˣ, hʸ)`. This is the actual mathematical core of the paper (Yoneda: an object is determined by its relationships to everything else), and nobody in the current plan tests it directly.

**Do this**: compute `π(x|y)` directly, and separately estimate `L̂(hˣ,hʸ)` by sampling many continuations `c` and computing `inf_c min{h_y(c)/h_x(c), 1}`. Compare.

**Payoff**: pure black-box numerical check, no gradients or internals needed. If this identity fails badly, the whole semantic category construction is standing on sand regardless of how good any downstream steering vector looks.

### 3. Use the internal hom formula as a zero-shot entailment predictor
The paper gives an explicit observational formula for entailment (Theorem 4 / Definition 13), computed purely from conditional probabilities — no causal intervention needed.

**Do this**: compute (or approximate via sampling) `[h^x, h^y](c) = inf_d π(d|y) / min{π(d|c), π(d|x)}` for premise-hypothesis pairs from a standard NLI dataset (SNLI/RTE). Check correlation with human entailment labels, against baselines like embedding cosine similarity or plain `π(y|x)`.

**Payoff**: testable purely via API, on any model (GPT, Claude, Gemini), not just open-weight ones. A genuinely different question from Aim 2's causal-intervention framing.

### 4. Build the paper's own "gender-neutral pronoun" example
The authors speculate in their conclusion that `h_he ⊔ h_she` (the coproduct) should behave like a gender-neutral concept, but never test it.

**Do this**: compute the coproduct copresheaf (pointwise max over many contexts) for pairs like he/she, cat/dog, Monday/Tuesday. Search the vocabulary for the expression whose own copresheaf best matches it. Does he ⊔ she land near "they"? Does cat ⊔ dog land near "pet" or "animal"?

**Payoff**: high interpretability-to-effort ratio — success or failure is intuitively legible to any reader, not just a metric that moved slightly. Good as a hook example regardless of what the other experiments show.

### 5. A lighter, non-SAE version of the tree-metric test
Aim 3's SAE + phylogenetic-tree pipeline on Gemma-2B is heavy. A cheaper version is available straight from the paper's own math.

**Do this**: use the classical four-point condition (Buneman) for tree metrics, checked directly on `d(x,y) = -ln π(y|x)` for curated quadruples with known hierarchy (e.g. animal → mammal → dog → poodle) versus scrambled/unrelated quadruples. Check whether the condition holds for the real hierarchy and breaks for the scrambled one.

**Payoff**: a cheap stepping-stone version of "is meaning tree-shaped," runnable without touching internals or SAEs, before committing to the heavier Aim 3 pipeline.

---

## Rough prioritization

- **#2 (Yoneda identity check)** — sharpest foundational test, costs almost nothing to run.
- **#3 (internal hom as entailment predictor)** — most publishable as a standalone result, has real baselines to compare against.
- **#1 (composition-law stress test)** — most conceptually novel; a systematic violation pattern would be a finding in its own right, not just a pass/fail on the theory.
- **#4 and #5** — good complementary pieces: #4 is a strong illustrative example, #5 is a lighter first pass at the Aim 3 question before going to the full SAE pipeline.
