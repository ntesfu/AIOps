# Assembly101 ego30 ablation wave 8

Wave 8 tests a structural correction to mistake detection. The legacy head makes
59 independent onset decisions at every feature row. In the expanded training
set, each of the 140 mistake timestamps identifies exactly one component, so a
shared temporal decision followed by conditional component localization matches
the observed label structure and removes 58 same-time negatives from the timing
task.

The opt-in factorized head computes
`P(component mistake at t) = P(any mistake at t) * P(component | mistake, t)`.
The first term receives every training mistake. The selector is trained only on
positive rows. Timing and localization losses are reported separately, while
the joint probability retains the existing per-component inference interface.
The added head has only `hidden_dim + 1` parameters and negligible VRAM cost.

## Attribution

- H0 versus E0 isolates factorization with the sampler and procedural context
  held at the baseline settings.
- H1 versus G1 isolates factorization when two event-centered crops are used.
- H1 versus H0 measures the crop effect when factorization is fixed on.

Both runs use seed 7 for direct paired comparisons and run for 32 epochs on the
expanded one-view cache. Model selection is validation-only. The operational
selector rejects checkpoints with zero strict mistake recall, more than two
false alerts per minute, or normality AP below the validation mistake prevalence.
It then uses a harmonic balance of action and mistake quality so strength on one
task cannot fully hide failure on the other. The held-out test split remains
untouched until the final architecture is selected.

This wave does not combine procedural context, a learned fusion head, hard
negative mining, or a larger backbone. Those factors previously changed several
signals at once or regressed validation performance and would prevent clean
attribution of the new head.
