# MalEval2026 direct Stage-II route-budget diagnostic

This experiment uses one frozen 2017+2018 calibrated P31-SVM, one frozen
SVM-anchored MeanOpposition ranking, and one shared U2 direct C43 Stage-II
model.  It evaluates nested Top25%, Top30%, Top40%, and Top50% budgets.

The direct Stage-II logit replaces the Stage-I output for routed APKs.  The
architecture contains no bounded residual, tanh correction, max-delta
integration, or budget-specific model retraining.  The union Top50% cohort is
extracted and predicted once, after which all budgets are evaluated from the
same probabilities.

MalEval target labels are sealed until `evaluate.py`.  They are used only for
diagnostic metrics and cannot select a formal CARE-Droid route budget.
