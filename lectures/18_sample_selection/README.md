## Sample selection in regression models
The notebook [sample_selectionn.ipynb](https://github.com/bschjerning/EconometricsB/blob/main/lectures/18_sample_selection/sample_selectionn.ipynb) gives an introduction to sample selection in regression models based on chapter 19 in Wooldridge (2010). 

We first consider sample selection in regression models such as

<img src="https://render.githubusercontent.com/render/math?math=y_{1}=x_{1}\beta _{1}+u_{1},\quad \quad E( u_{1}|x_{1}) =0">


where y_1 or x_1 or both are unobserved when some selection indicator s=0

We consider 5 cases: 
1. s is a function of x_1 only
1. s is independent of x_1, and u_1
1. s=1(a_1 < y_1 < a_2) (truncation)
1. s=1(x delta_2+v_2>0) (discrete response selection with dependence between u_1 and v_2)
1. y_2=max(0,x delta_2+v_2) and s=1(y_2>0) (Tobit selection with dependence between u_1 and v_2 - implies more structure)

Case 1-3 and introcution is presented in video 1 and case 4-5 is presneted in video 2 (links to videos are given below) 

We then move on and discuss

1. The impotance of exclusion restrictions (video 3)
1. Likelihood models (video 4)
1. Nonparametric bounds: Set vs point identification in the sample selection model (video 5)

The material presented in 5 videotaped lectures available on YouTube

1. [Introduction to sample selection in regression models - case 1-3 (38:28 min)](https://youtu.be/fXK7y9PzpOY)
2. [Sample selection in regression models with incidental truncation - case 4-5 (41:08 min)](https://youtu.be/wKG6FuMyuH0)
3. [Exclusion restrictions (16:06 min)](https://youtu.be/yX_Hc75LFN8)
4. [Likelihood models (39:34 min)](https://youtu.be/9970FJKTlAU)
5. [Nonparametric identification (21:04 min)](https://youtu.be/6iMGLztA_2s)
