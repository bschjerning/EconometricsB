## Specification issues for censored regression and corner solution outcomes and estimation under weaker assumptions

The notebook [clad.ipynb](https://github.com/bschjerning/metrics2021/blob/main/17_censored_regrssion_clad/clad.ipynb) discuss the consequences of misspecifications of the Tobit model such as neglected heterogeneity, non-normality and heteroscedasticity. We further discuss how to test for these misspecifications (using conditional moments tests) and how to estimate censored regression models under weaker assumptions. 

Particular emphasis is on Powell's *censored least absolute deviation estimator (CLAD)* and the challenges for implementation and inference that arise because the objective function is *non-differentiable*. For implementation we consider several ways out such as the *Iterative Linear Programming ALgorithm (ILPA)* presented in [Buchinsky and Han (Econometrica 1998)](https://doi.org/10.2307/2998578). We also discuss how to add *logit-smoothing* as a devise to obtain a well-behaved objective function that still allow us to approximate the model of interest. Due to the non-differentiability of the CLAD objective, inference is non-standard. We therefore illustrate the power of *Bootstrapping* as a method to obtain (for example) standard errors. 
 

Our treatment of censored regression and corner solution outcomes is based on chapter 18 in Wooldridge's (2010). 

The material presented in 5 videotaped lectures available on YouTube

1. [Neglected heterogeneity, heteroscedasticity and non-normality (33:41 min)](https://youtu.be/OlmqEQEZAiU)
2. [Specification testing (15:42 min)](https://youtu.be/eRkozJlu2xA)
3. [Powell's censored least absolute deviation estimator (CLAD)(13:20 min)](https://youtu.be/SmiBmW7oSfU)
4. [Computational issues - smoothing and ILPA (44:45 min)](https://youtu.be/HE6_WR0XrTs)
5. [Bootstrapping standard errors (34:22 min)](https://youtu.be/3dAO5pMEOA8)
