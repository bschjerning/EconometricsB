## Introduction to binary response models. 

**Agenda:** We give an introduction to binary response models. The most common application of binary response models is when we are interested in “explaining” a binary outcome in terms of some explanatory variables. So we are interested in a conditional probability $p(x)=P(y=1|x)$.

We start with Linear Probability Model (LPM) which is the simplest possible model for binary outcomes where our model for $p(x)$ is simply a linear index $x\beta$. We the move on to linear index models such as Probit and Logit and discuss the importance of scaled normalizations as a requirement for identification and in turn how this affects the interpretation of  of results from LPM, Probit and Logit. Having established identification though a scaled normalization, we move on to estimation of parameter estimates and marginal effects and compare and interpret the results from different models. 

In order to illustrate the properties and practical implementation of various binary response models, we work with the example from the textbook on female labor force participation. 

**Slides/Notebook**: [binary_choice.ipynb](/lectures/10_binary_response/binary_choice.ipynb)

**Readings:** Wooldridge Chapter 15, Sections 15.1-15.4.


Earlier version of the material is presented in 5 videotaped lectures available on YouTube:

1. [Introduction](https://youtu.be/muER_OevcIs)
2. [Linear probability model](https://youtu.be/0QVbFbtNqW4)
3. [Index models for binary response: Identification issues.](https://youtu.be/WfzM5v9IVJM)
4. [Maximum likelihood estimation of binary response index models.](https://youtu.be/6Z4_DtY5Bng)
5. [Partial Effects](https://youtu.be/-T3VeZVb5P8)



