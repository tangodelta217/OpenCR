# Reporting rules (Annex A)

## Non-negotiables
- Reports must be "non-misleading":
  - If ROC is not applicable (regression), explicitly state it and generate regression plots.
  - If probabilities are missing, do not compute AUC; report alternatives.

## ROC rules
- Binary: use y_score (probabilities or decision function).
- Multiclass: use aligned y_proba with a global class order. Never use undefined variables.
- Never silently skip ROC due to NameError/undefined class order; fail loudly or downgrade with an explicit message.

## Output contract
- Annex A must generate:
  - at least 1 table (csv + md)
  - at least 2 figures
- Keep outputs deterministic with the provided seed/splits.
