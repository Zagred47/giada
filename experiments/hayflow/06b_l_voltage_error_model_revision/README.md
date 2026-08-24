# 06b-l — atomic voltage error-model revision

This is the terminal diagnostic before an architecture revision. It tests an
explicit mixture between persistence (zero voltage delta) and the frozen
06b-j dynamic update.

The aligned causal arms are hard and soft gates using either instantaneous or
temporal causal features. They are compared with a teacher-regime oracle and
the per-step optimal convex-blend oracle; neither oracle is selectable. All
causal gates are region-specific closed-form ridge models fit on recursive
static-lookup exposures.

Every possible outcome maps directly to an architecture change: a causal
hurdle gate, a regime-state encoder, a continuous mixture state or a revised
voltage expert family. This experiment cannot open another generic diagnostic
loop.
