* Assumptions: correct model specification, normality, homoscedasticity (affect variance, but not bias), linearity, iid; selection bias, confounding, endogeneity (reverse causation; omitted var, measurement error, autoregression, simultaneity), collinearity (affect variance, but not bias), missing var, outliers, range restriction, range enhancement, fixed Xs, 10 events per var.
* Assumption tests (for explanation):
 * plot(density(x)),	car, fit.models, lmtest, nullabor (graphical inferences, e.g. qq plots), mvinfluence, LogisticDx
	gof (Cumres for lm and glm), hydroGOF, intRegGOF (lm. glm, nls), gvlma (lm)
* Bayesian: arm (bayesglm), BEST, rstan
* Causation:
 * Assumptions: counterfactural, d-separation (conditional independence), identifiability, positivity, exchangeability, consistency
 * bnlearn, causaleffect, cit, CompareCausalNetworks (backShift, CAM, InvariantCausalPrediction, pcalg), daggity (SEM), generalCorr (gmcmtx0), iWeigReg (ate.clik, missing data), multiPIM (interaction, tmle), SID, bnlearn, wfe (longitudinal data), conformal, conformalinference
 * Binary Rx: ATE (no need to specify outcome or Rx models), causalFX, causalGAM, causalsens, causalTree (subgroup treatment effects), CovSel, CovSelHigh, EffectLiteR (categorical Rx, manifest or latent vars, SEM), tetrad
 * Categorical Rx: EffectLiteR
 * Continuous or discrete vars: bdgraph, pcalg, ParallelPC, PCovR (optimze both prediction and pca), stableSpec, tmle.npvi (continuous exposure), MatchLinReg, tmle, tmlecte, treatSens
 * Continuous Rx: causalR, CBPS (dr, gen. ps), causaldrf (gen. ps)
 * Double-robust: drgee, tmle
 * High-dimensional: crossEstimation
* changepioint: bcp, ebayesthresh, quantreg (rq), strucchange, segmented, wavethresh
* Collinearity (VIF>10, only for explanation):
 * CorReg, ridge, bestglm, yhat(dominance analysis), rf, pcareg, penalized (hd1), perturb, pls (pcr), sem, centering
* Compare:
 * compareGroups, contrast, lsmeans, perfect-t-test
* Crossvalidation:
 * cvAUC
* Description:
 * compareGroups, DescTools, pastecs (stat.desc)
* Effect size:
 * bootES, effsize, effects (graphing glm effects and interactions)
* Ensemble or model averaging (only for prediction):
 * AICcmodavg, BMA, BMS, caretEnsemble, glmulti, h2oEnsemble, MuMIn, bagging (bootstrap aggregating), boosting, caretEnsemble, medley, subsemble, superlearner
* Explanation:
 * Theory-driven, data-generating process, inference, correlation, effect size, R2, rmse, p-value, minimize bias
 * CovSel (for lm)
* Feature selection (fs) or penalization:
 * AIC: AIC=AIC(y) + 2*sum(log(y)) (Jacobian of the transformation) for lognormal models (log(y))
 * mplot (lm & glm, bs & stability), mht, c060 (Cox and glm), bestglm, fealect, glmnet (lasso), glmulti (for interactions), meifly, pls, pcr, Boruta, caret(earth, gcvEarth, glmnet, glmStepAIC, lmstepAIC), fscaret, Fselector (rf), FWDselect (binary, lm, Poisson), loo (bayes), penalized, regsel, mlr, polywog, NonpModelCheck, RRegrs, varbvs, VSURF, apricom, shrink
 * binary Y: VariableSelection
 * classification: 
 * lm:	BayesVarSel, CovSel (Model-Free Covariate or confounder Selection for explanation or causation), knockoff, leaps, glmnet, relaxo
 * survival: c060, rsig, glmnet, glmulti (interaction), ipred, mRMRe, penalized, ranger (rf), SurvRank
* Fuzzy:
 * frbs, RoughSets (fs)
* Graph:
 * ggpubr, ggcorplot, easyGgplot2, ggfortify, ggplot2, ggvis, sjPlot, survminer, heplots, plotmo, plotluck, sjPlot
* Hierachical:
 * AutoModel, bhglm (bayes), dhglm (lmm), glmbb, hglm, HGLMMM (lmm), hiertest, hisemi, hit (hd1), mdhglm (lmm, mv), structSSI (multiplicity, fdr)
* High-Dimensional:
 * hdglm, hdi, hdm, stabs, tsne
* Importanace:
 * AICcmodavg, Boruta (rf), conformalinference, MuMIn, multiPIM (interaction, tmle), relaimpo, rms [plot(anova(f), what='proportion chisq')], tmle.npvi, yhat (Interpreting Regression Effects), vita
 * lm: CovSel, lm, lmtest, relaxo
* Instrumental var: I causes x but not y, x does not cause I
* Interaction (moderation): residual centering: pequod, rockchalk, semTools (residualCovariate)
* Machine learning (for prediction): https://cran.r-project.org/web/views/MachineLearning.html
 * bartMachine, BayesTree, mlr, rgp (genetic programming, automatic generation of equations), RRegrs, SuperLearner, xgboost, interpretation: ICEbox, interpretR
* Matching:
 * CBPS, MatchingFrontier, cem, designmatch, hdps, MatchIt, PSAboot, TriMatch, twang, Matching, cobalt
* Measurement errors: findFn("measurement errors")
 * decon, glsme, lpme, sem, semtools, simex
* Mediation: findFn"mediation"
 * cit (Causal Inference Test), DirectEffects, mediation, medflex (for uc), RMediation (CI for a*b)
* Missing:
 * Amelia, BaBooN, cmpute, Hmisc, imputeLCMD, jomo, mi, mice, miceadds, MissingDataGUI, mitml, mitools, mix, mvnmle, norm, rrcovNA, softImpute, VIM
* Multiple comparison: findFn("multiple comparison")
 * adjust, coin, ETC, gMCP, lsmeans, MCPAN, mcprofile, mratios, MultEq, multcomp, multtest, MUTOSS, nparcomp, PMCMR SimComp, someMTP, fdrtool, qvalue, selectiveCI
* Omics:
 * ZeroSum
* Outliers:
 * FastPCs, mvoutlier
* Penalized regression (shrinkage, no SE or CI): bayesPen, biglars, covTest, elasticnet, extlasso, frailtypack, glmnet, EBglmnet, glmmlasso, glmpath, grplasso, grpreg, lars, lasso2, lqa, monomvn, ncvreg, parcor, penalized, penalizedSVM, plus, quantreg, polywog, relaxo, ridge, rqPen, RXshrink, SIS
* PLS-SEM (variance-based SEM; reflective indicators: PLSc, high reliability and validity; formative indicators: ):
 * matrixpls (with wrappers for plsm, sempls), plsRbeta, plsRcox, plspm, sempls
* Power:
 * poweR, pwr
* Prediction:
 * data-driven, c-index, overfitting, out-of-sample prediction error, cross-validation or bootstrapping for internal validation, to minimize bias plus variance, rmse)
 * caret (train, calibrate, validate, bs), machine learning
 * apricom (lm & lr, shrinkage), conformalinference, hdps, ModelGood, RRegrs
 * Prediction error: A3, perry (Resampling-based prediction error estimation for regression models), pec (survival)
* Regression:
 * visreg, use prior knowledge, do NOT let the computers do all the works for you, do NOT drop pre-specified insignificant variables, imputation (Hmisc: aregimpute), interaction and rcs for Xs (k knots, k=3-5 chosen by AIC, k=3 if n<30, k=5 if n>100) followed by penalization, check additivity by a global test and either keep all or delete all interactions, fastbw if parsimony is more important than prediction), MBESS
 * Hmisc: areg (Additive Regression with Optimal Transformations on Both Sides using Canonical Variates), aregImpute,redun (redundancy analysis), transcan
 * rms:	calibrate, fastbw, plsmo, Rq (quantreg), rrcov, validate
 * car:	avPlot (influential data), boxCox, crPlot (linearity), ncvTest (heteroscedasticity test for lm), anova (for nested models), dataEllipse, influencePlot (to detect outliers, better than Cooks distance in plot), linearHypothesis, outlierTest, leveragePlots, qqPlot (normality), vif
 * Binary Y: arm (bayesglm)
 * GAM (gamlss, mgcv, VGAM), Hierachical regression (AutoModel), MARS (earth), quantile regression (quantreg), alr4, MBESS, regclass, rms (calibrate, validate)
* Reproducible research:
 * markdown: knitr, eidtR, notebook, archivist, archivist.github, ProjectTemplate
* Robust:
 * drgee, robust (glmRob), mblm, rgam, MASS (lqs, rlm), robcor, robustlmm, robustreg, sandwich (vcovHC), wrs2, lqmm (lmm quantreg) rqpd
* Sensitivity analysis
 * causalsens (uc), episensr, mediation, multisensi, obsSens, rbounds, SBSA (fs for glm and survival analysis), sensitivity, sensitivitymv, sensitivityPstrat (principal stratification), SobolSensitivity, treatSens (binary uc)
Structural equation model (only for reflective indicators)
 * lavaan, sem
* Survival:
 * c060 (lasso, complexity.glmnet, Plot.peperr.curves, plot.stabpath), glmbfp (Bayesian model selection), greyzoneSurv, rms
 * joint model: JM, JMbayes
