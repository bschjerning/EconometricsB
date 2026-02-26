
## Binary response models for panel data

**Agenda:** We will now move on to binary response models for panel data.  We start with binary response models without unobserved which we will estimate by partial maximum likelihood (PML). PML is described in Wooldridge section 13.8. We then move on to unobserved effects models under strict exogeneity, where we wish to explicitly model unobserved individual specific and time constant heterogeneity. Here we consider a battery of different econometric models such random effects probit models, correlated random effects models and fixed effect logit. To estimate random effects binary response models we often need to numerically solve integral to "integrate out"  the unobserved effects, We therefore spend some time on a small recap on numerical quadrature and show how it is used in the context of an econometric model. 

We end our treatment of chapter 15 with dynamic unobserved effects models where particular interest is in coefficient on lagged dependent variable (state dependence). We illustrate the importance of handling the so-called initial condition problem in dynamic and appropriately control for unobserved heterogeneity. Failing to control for these effects results in spurious state dependence. 

In order to illustrate the properties and practical implementation of various binary response models, we run a number of simulations and work with the example from the textbook on female labor force participation (see table 15.3 and example 15.6). 

**Readings:** Wooldridge Chapter 15, Sections 15.8, and Section 13.8.

I do not cover 15.7 in detail the video. But it contains important stuff about specification issues in binary response models: such as endogeneous explanatory variables. In the simulations, we do analyze the effects of neglected heterogeneity. 

**Slides/Notebook**: [binary_choice_panel.ipynb](/lectures/12_binary_response_panel/binary_choice_panel.ipynb) 

Earlier version of the material is presented in 5 videotaped lectures available on YouTube: 

The material presented in 5 videotaped lectures available on YouTube

1. [Binary response for panel data without unobserved effects (51 min)](https://youtu.be/K-VZ6gi7bz8)
2. [Unobserved effects models under strict exogeneity (1h 40 min)](https://youtu.be/3vWxSIQjvhU)
3. [Dynamic unobserved effects models (21 min)](https://youtu.be/XTxAtPBo_ZA)
4. [Dynamic unobserved effects models - simulation example (21 min)](https://youtu.be/TommACLjz08)
5. [Dynamic unobserved effects models  - empirical example (20 min)](https://youtu.be/hZcHFyhmbhE)
