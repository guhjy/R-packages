* Assumptions: correct model specification, normality, homoscedasticity (affect variance, but not bias), linearity, iid; selection bias, confounding, endogeneity (reverse causation; omitted var, measurement error, autoregression, simultaneity), collinearity (affect variance, but not bias), missing var, outliers, range restriction, range enhancement, fixed Xs, 10 events per var.
* Assumption tests (for explanation):
 * plot(density(x)),	car, fit.models, lmtest, nullabor (graphical inferences, e.g. qq plots), mvinfluence, LogisticDx
	gof (Cumres for lm and glm), hydroGOF, intRegGOF (lm. glm, nls), gvlma (lm)
* Bayesian: arm (bayesglm), BEST, rstan
