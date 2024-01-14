# CS329 Homework #2

*Course: Machine Learning(H)(CS329) - Instructor: Qi Hao*

Name: Jianan Xie(谢嘉楠)

SID: 12110714

## Question 1

*(a)* **[True or False]** If two sets of variables are jointly Gaussian, then the conditional distribution of one set conditioned on the other is again Gaussian. Similarly, the marginal distribution of either set is also Gaussian



##### Ans：True

*(b)* Consider a partitioning of the components of $x$ into three groups $x_a$, $x_b$, and $x_c$, with a corresponding partitioning of the mean vector $\mu$ and of the covariance matrix $\Sigma$ in the form

$$\mu = \left( \begin{array}{c} \mu_a\\\mu_b\\\mu_c \end{array} \right) , \quad \Sigma=\left( \begin{array}{ccc} \Sigma_{aa} & \Sigma_{ab} & \Sigma_{ac}\\  \Sigma_{ba} & \Sigma_{bb} & \Sigma_{bc}\\  \Sigma_{ca} & \Sigma_{cb} & \Sigma_{cc} \end{array} \right).$$

Find an expression for the conditional distribution $p(x_a|x_b)$ in which $x_c$ has been marginalized out.



##### Ans:

For a joint Gaussian distribution $p(\mathbf{x})=N(\mathbf{x}|\mathbf{\mu,\Sigma})$, where $\mathbf{x}=\left(\begin{array}{c} \mathbf{x}_a \\ \mathbf{x}_b \end{array}\right)$, $\mu= \left( \begin{array}{c} \mu_a \\ \mu_b \end{array} \right)$, $\Sigma= \left( \begin{array}{c} \Sigma_{aa} \ \Sigma_{ab} \\ \Sigma_{ba} \ \Sigma_{bb}\end{array} \right)$, $\Lambda= \left( \begin{array}{c} \Lambda_{aa} \ \Lambda_{ab} \\ \Lambda_{ba} \ \Lambda_{bb}\end{array} \right)=\Sigma^{-1}$ ,we have known that the conditional distribution $p(\mathbf{x_a|x_b})=N(\mathbf{x}|\mathbf{\mu_{a|b},\Lambda_{aa}^{-1}})$, where $\mu_{a|b}=\mu_a -\Lambda_{aa}^{-1}\Lambda_{ab}(x_b-\mu_b)$. And the marginal distribution $p(\mathbf{x_a}) =N(\mathbf{x}|\mathbf{\mu_a, \Sigma_{aa}})$.

So, first we consider $\mathbf{x}=\left(\begin{array}{c} \mathbf{x}_{a,b} \\ \mathbf{x}_c \end{array}   \right)$, where $\mathbf{x_{a,b}}=\left(\begin{array}{c} \mathbf{x}_a \\ \mathbf{x}_b   \end{array}   \right)$. From above, we get the marginal distribution of $\mathbf{x_{a,b}}$ to make $\mathbf{x_c}$ marginalized out, then we get  $p(\mathbf{x_{a,b}})=N(\mathbf{x}|\mathbf{\mu_{a,b}}, \Sigma_{a,b})$ ,$\mu_{a,b} = \left( \begin{array}{c} \mu_a \\ \mu_b \end{array} \right)$,  $\Sigma_{a,b}= \left( \begin{array}{c} \Sigma_{aa} \ \Sigma_{ab} \\ \Sigma_{ba} \ \Sigma_{bb}\end{array} \right)$. Next we find the conditional distribution  $p(\mathbf{x_a|x_b})=N(\mathbf{x}|\mathbf{\mu_{a|b},\Lambda_{aa}^{-1}})$, where $\mu_{a|b}=\mu_a -\Lambda_{aa}^{-1}\Lambda_{ab}(x_b-\mu_b)$, using what we learned above.

## Question 2

Consider a joint distribution over the variable

$$\mathbf{z}=\left(  \begin{array}{c}   \mathbf{x}\\\mathbf{y} \end{array} \right)$$

whose mean and covariance are given by

$$    \mathbb{E}[\mathbf{z}]=\left(   \begin{array}{c}    \mu \\ \mathbf{A}\mu \mathbf{+b}    \end{array} \right),    \quad   \mathrm{cov}[\mathbf{z}]=\left( \begin{array}{cc}   \mathbf{\Lambda^{-1}}\quad\quad \mathbf{\Lambda^{-1}A^\mathrm{T}}\\ \mathbf{A\Lambda^{-1}}\quad \mathbf{L^{-1}+A\Lambda^{-1}A^\mathrm{T}}   \end{array} \right).$$

*(a)* Show that the marginal distribution $p(\mathbf{x})$ is given by $p(\mathbf{x})=\mathcal{N}(\mathbf{x|}\mu\mathbf{, \Lambda^{-1}})$.



##### Ans:

From what we have learned, when given a joint Gaussian distribution $p(\mathbf{x})=N(\mathbf{x}|\mathbf{\mu,\Sigma})$, where $\mathbf{x}=\left(\begin{array}{c} \mathbf{x}_a \\ \mathbf{x}_b \end{array}\right)$, $\mu= \left( \begin{array}{c} \mu_a \\ \mu_b \end{array} \right)$, $\Sigma= \left( \begin{array}{c} \Sigma_{aa} \ \Sigma_{ab} \\ \Sigma_{ba} \ \Sigma_{bb}\end{array} \right)$, then the marginal distribution $p(\mathbf{x_a}) =N(\mathbf{x}|\mathbf{\mu_a, \Sigma_{aa}})$. Here, $\mu_x = \mu$ and $\Sigma_{xx}=\Lambda^{-1}$, Therefore, $p(\mathbf{x})=\mathcal{N}(\mathbf{x|}\mu\mathbf{, \Lambda^{-1}})$.

*(b)* Show that the conditional distribution $p(\mathbf{y|x})$ is given by $p(\mathbf{y|x})=\mathcal{N}(\mathbf{y|Ax+b, L^{-1}})$.



##### Ans:

From what we have learned, when given a joint Gaussian distribution $p(\mathbf{x})=N(\mathbf{x}|\mathbf{\mu,\Sigma})$, where $\mathbf{x}=\left(\begin{array}{c} \mathbf{x}_a \\ \mathbf{x}_b \end{array}\right)$, $\mu= \left( \begin{array}{c} \mu_a \\ \mu_b \end{array} \right)$, $\Sigma= \left( \begin{array}{c} \Sigma_{aa} \ \Sigma_{ab} \\ \Sigma_{ba} \ \Sigma_{bb}\end{array} \right)$, then the conditional distribution $p(\mathbf{x_a|x_b})=N(\mathbf{x}|\mathbf{\mu_{a|b},\Lambda_{aa}^{-1}})$, where $\mu_{a|b}=\mu_a -\Lambda_{aa}^{-1}\Lambda_{ab}(x_b-\mu_b)$. Therefore, $\Sigma_{y|x}=\Lambda_{yy}^{-1}=\Sigma_{yy}-\Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}=(\mathbf{L^{-1}+A\Lambda^{-1}A^\mathrm{T}})-(\mathbf{A\Lambda^{-1}})(\mathbf{\Lambda^{-1}})^{-1}(\mathbf{\Lambda^{-1}A^\mathrm{T}})=\mathbf{L^{-1}}$ and $\mu_{y|x}=\mu_y+\Sigma_{yx}\Sigma_{xx}^{-1}(x-\mu)=(\mathbf{A}\mu \mathbf{+b})+(\mathbf{A\Lambda^{-1}})(\mathbf{\Lambda^{-1}})^{-1}(x-u)=\mathbf{Ax} \mathbf{+b}$​. So,  $p(\mathbf{y|x})=\mathcal{N}(\mathbf{y|Ax+b, L^{-1}})$.

## Question 3

Show that the covariance matrix $\Sigma$ that maximizes the log likelihood function is given by the sample covariance

$$\mathrm{ln}p(\mathbf{X}|\mu, \Sigma)=-\frac{ND}{2}\mathrm{ln}(2\pi)-\frac{N}{2}\mathrm{ln}|\Sigma|-\frac{1}{2}\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu).$$

Is the final result symmetric and positive definite (provided the sample covariance is nonsingular)?

> #### Hints
>
> *(a)* To find the maximum likelihood solution for the covariance matrix of a multivariate Gaussian, we need to maximize the log likelihood function with respect to $\Sigma$. The log likelihood function is given by
>
> $$\mathrm{ln}p(\mathbf{X}|\mu, \Sigma)=-\frac{ND}{2}\mathrm{ln}(2\pi)-\frac{N}{2}\mathrm{ln}|\Sigma|-\frac{1}{2}\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu).$$
>
> *(b)* The derivative of the inverse of a matrix can be expressed as
>
> $$\frac{\partial}{\partial x}(\mathbf{A}^{-1})=-\mathbf{A}^{-1} \frac{\partial\mathbf{A}}{\partial x} \mathbf{A}^{-1}$$
>
> We have the following properties
>
> $$\frac{\partial}{\partial \mathbf{A}} \mathrm{Tr}(\mathbf{A}) = \mathbf{I}, \quad \frac{\partial}{\partial \mathbf{A}} \mathrm{ln}|\mathbf{A}| = (\mathbf{A^{-1}})^\mathrm{T}.$$



##### Ans:

From the hint(a), we only need to maximize the log likelihood function with respect to $\Sigma$. Here, we will use two useful identities for computing gradients. First is $\frac{\part{\mathbf{a^\mathrm{T} X^{-1} b}}}{\part{\mathbf{X}}}=-(\mathbf{X}^{-1})^\mathrm{T}\mathbf{ab}^\mathrm{T}(\mathbf{X}^{-1})^\mathrm{T}$, and the second is $\frac{\partial}{\partial \mathbf{A}} \mathrm{ln}|\mathbf{A}| = (\mathbf{A^{-1}})^\mathrm{T}$ in hint(b).

So, we derive that
$$
\begin{align}
\frac{\part{\mathrm{ln}p(\mathbf{X}|\mu, \Sigma)}}{\part\Sigma}&=-\frac{N}{2}\frac{\part}{\part \Sigma}\mathrm{ln}|\Sigma|-\frac{1}{2}\frac{\part}{\part \Sigma}\sum^N_{n=1}(\mathbf{x}_n-\mu)^\mathrm{T}\Sigma^{-1}(\mathbf{x}_n-\mu)
\\&=-\frac{N}{2}(\Sigma^{-1})^\mathrm{T}+\frac{1}{2}\sum^N_{n=1}(\Sigma^{-1})^\mathrm{T}(\mathbf{x}_n-\mu)(\mathbf{x}_n-\mu)^\mathrm{T}(\Sigma^{-1})^\mathrm{T}
\\&=-\frac{N}{2}(\Sigma^{-1})+\frac{1}{2}\sum^N_{n=1}(\Sigma^{-1})(\mathbf{x}_n-\mu)(\mathbf{x}_n-\mu)^\mathrm{T}(\Sigma^{-1}) \tag{Since $\Sigma$ is symmetric}
\\&=0
\end{align}
$$
Then we get $-N\Sigma+\sum^N_{n=1}(\mathbf{x}_n-\mu)(\mathbf{x}_n-\mu)^\mathrm{T}=0$ by multiplying $\Sigma$ on left side and on right side in sequence. Hence, we derive the $\Sigma_{MLE}=\frac{1}{N}\sum^N_{n=1}(\mathbf{x}_n-\mu)(\mathbf{x}_n-\mu)^\mathrm{T}$ which is the sample covariance matirx. Since the sample covariance is nonsingular, the final result is symmetric and positive definite.

## Question 4

*(a)* Derive an expression for the sequential estimation of the variance of a univariate Gaussian distribution, by starting with the maximum likelihood expression

$$\sigma^2_{\mathrm{ML}} =\frac{1}{N}\sum^N_{n=1}(x_n-\mu)^2.$$

Verify that substituting the expression for a Gaussian distribution into the Robbins-Monro sequential estimation formula gives a result of the same form, and hence obtain an expression for the corresponding coefficients $a_N$. 



##### Ans: 

we will denote $\sigma^{(N)}_{ML}$ as maximum likelihood estimator of $\sigma$ when it is based on N observations
$$
\begin{align}
\sigma^{2(N)}_{\mathrm{ML}} &=\frac{1}{N}\sum^N_{n=1}(x_n-\mu)^2.
\\&=\frac{1}{N}(x_N-\mu)^2+\frac{1}{N}\sum^{N-1}_{n=1}(x_n-\mu)^2
\\&=\frac{1}{N}(x_N-\mu)^2+\frac{N-1}{N}\sigma^{2(N-1)}_{\mathrm{ML}}
\\&=\sigma^{2(N-1)}_{\mathrm{ML}}+\frac{1}{N}((x_N-\mu)^2-\sigma^{2(N-1)}_{\mathrm{ML}})
\end{align}
$$
by Robbins-Monro sequential estimation formula,
$$
\theta^{(N)}=\theta^{(N-1)}-\alpha_{N-1}\frac{\part}{\part{\theta^{(N-1)}}}[-\mathrm{ln}p(x_N|\theta^{(N-1)})]
$$
substituting the expression for a Gaussian distribution into the Robbins-Monro sequential estimation formula gives,
$$
\begin{align}
\sigma^{2(N)}_{ML}&= \sigma^{2(N-1)}_{ML}-\alpha_{N-1}\frac{\part}{\part{\sigma^{2(N-1)}_{ML}}}[-\mathrm{ln}p(x_N|\sigma^{2(N-1)}_{ML})]
\\&=\sigma^{2(N-1)}_{ML}-\alpha_{N-1}\frac{\part}{\part{\sigma^{2(N-1)}_{ML}}}[\frac{1}{2}\mathrm{ln}(2\pi)+\frac{1}{2}\mathrm{ln}\sigma^{2(N-1)}_{ML}+\frac{(x_N-\mu)^2}{2\sigma^{2(N-1)}_{ML}}]
\\&=\sigma^{2(N-1)}_{ML}-\alpha_{N-1}(\frac{1}{2\sigma^{2(N-1)}_{ML}}-\frac{(x_N-\mu)^2}{2\sigma^{4(N-1)}_{ML}})
\\&=\sigma^{2(N-1)}_{ML}+\frac{\alpha_{N-1}}{2\sigma^{4(N-1)}_{ML}}((x_N-\mu)^2-\sigma^{2(N-1)}_{ML})
\end{align}
$$
Thus, we take $\alpha_N=\frac{2\sigma^{4(N)}_{ML}}{N+1}$ to make the result of Robbins-Monro same with $\sigma^{2(N-1)}_{\mathrm{ML}}+\frac{1}{N}((x_N-\mu)^2-\sigma^{2(N-1)}_{\mathrm{ML}})$.

*(b)* Derive an expression for the sequential estimation of the covariance of a multivariate Gaussian distribution, by starting with the maximum likelihood expression

$$\Sigma_{\mathrm{ML}}=\frac{1}{N}\sum^N_{n=1}(\mathbf{x}_n-\mu_{\mathrm{ML}})(\mathbf{x}_n-\mu_{\mathrm{ML}})^\mathrm{T} .$$

Verify that substituting the expression for a Gaussian distribution into the Robbins-Monro sequential estimation formula gives a result of the same form, and hence obtain an expression for the corresponding coefficients $a_N$.

> #### Hints
>
> *(a)* Consider the result $\mu_\mathrm{ML}=\frac{1}{N}\sum^N_{n=1}\mathbf{x}_n$ for the maximum likelihood estimator of the mean $\mu_\mathrm{ML}$, which we will denote by $\mu^{(N)}_{\mathrm{ML}}$ when it is based on $N$ observations. If we dissect out the contribution from the final data point $\mathbf{x}_N$, we obtain
>
> $$ \mu^{(N)}_{\mathrm{ML}} =\frac{1}{N}\sum^N_{n=1}\mathbf{x}_n    = \frac{1}{N}\mathbf{x}_N+\frac{1}{N}\sum^{N-1}_{n=1}\mathbf{x}_n   = \frac{1}{N}\mathbf{x}_N+\frac{N-1}{N}\mu^{(N-1)}_{\mathrm{ML}} $$
>
> *(b)* Robbins-Monro for maximum likelihood
>
> $$\theta^{(N)}=\theta^{(N-1)}+a_{(N-1)}\frac{\partial}{\partial\theta^{(N-1)}}\mathrm{ln}p(x_N|\theta^{(N-1)}).$$



##### Ans:

we will denote $\Sigma^{(N)}_{ML}$ as maximum likelihood estimator of $\Sigma$ when it is based on N observations
$$
\begin{align}
\Sigma^{(N)}_{\mathrm{ML}}&=\frac{1}{N}\sum^N_{n=1}(\mathbf{x}_n-\mu_{\mathrm{ML}})(\mathbf{x}_n-\mu_{\mathrm{ML}})^\mathrm{T}
\\&=\frac{1}{N}(\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}+\frac{1}{N}\sum^{N-1}_{n=1}(\mathbf{x}_n-\mu_{\mathrm{ML}})(\mathbf{x}_n-\mu_{\mathrm{ML}})^\mathrm{T}
\\&=\frac{1}{N}(\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}+\frac{N-1}{N}\Sigma^{(N-1)}_{ML}
\\&=\Sigma^{(N-1)}_{ML}+\frac{1}{N}((\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}-\Sigma^{(N-1)}_{ML})
\end{align}
$$
by Robbins-Monro sequential estimation formula,
$$
\theta^{(N)}=\theta^{(N-1)}-\alpha_{N-1}\frac{\part}{\part{\theta^{(N-1)}}}[-\mathrm{ln}p(x_N|\theta^{(N-1)})]
$$
substituting the expression for a Gaussian distribution into the Robbins-Monro sequential estimation formula gives,
$$
\begin{align}
\Sigma^{(N)}_{\mathrm{ML}}&= \Sigma^{(N-1)}_{\mathrm{ML}}-\alpha_{N-1}\frac{\part}{\part{\Sigma^{(N-1)}_{\mathrm{ML}}}}[-\mathrm{ln}p(\mathbf{x}_N|\Sigma^{(N-1)}_{\mathrm{ML}})]
\\&=\Sigma^{(N-1)}_{\mathrm{ML}}-\alpha_{N-1}\frac{\part}{\part{\Sigma^{(N-1)}_{\mathrm{ML}}}}[\frac{ND}{2}\mathrm{ln}(2\pi)+\frac{N}{2}\mathrm{ln}|\Sigma^{(N-1)}_{\mathrm{ML}}|+\frac{1}{2}(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}(\mathbf{x}_N-\mu_{\mathrm{ML}})]
\\&=\Sigma^{(N-1)}_{\mathrm{ML}}-\alpha_{N-1}[\frac{N}{2}((\Sigma^{(N-1)}_{\mathrm{ML}})^{-1})^\mathrm{T}-\frac{1}{2}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}(\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}]
\\&=\Sigma^{(N-1)}_{\mathrm{ML}}-\alpha_{N-1}[\frac{N}{2}((\Sigma^{(N-1)}_{\mathrm{ML}})^{-1})-\frac{1}{2}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}(\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}]

\\&=\Sigma^{(N-1)}_{\mathrm{ML}}+\alpha_{N-1}\frac{N}{2}(\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}[\frac{1}{N}((\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}-\Sigma^{(N-1)}_{ML})](\Sigma^{(N-1)}_{\mathrm{ML}})^{-1}
\end{align}
$$

Thus, we take $\alpha_N=\frac{2\Sigma^{2(N)}_{ML}}{N+1}$ to make the result of Robbins-Monro same with $\Sigma^{(N-1)}_{ML}+\frac{1}{N}((\mathbf{x}_N-\mu_{\mathrm{ML}})(\mathbf{x}_N-\mu_{\mathrm{ML}})^\mathrm{T}-\Sigma^{(N-1)}_{ML})$.

## Question 5

Consider a $D$-dimensional Gaussian random variable $\mathbf{x}$ with distribution $N(x|\mu, \Sigma)$ in which the covariance $\Sigma$ is known and for which we wish to infer the mean $\mu$ from a set of observations $\mathbf{X}=\{x_1, x_2, ......, x_N\}$. Given a prior distribution $p(\mu)=N(\mu|\mu_0, \Sigma_0)$, find the corresponding posterior distribution $p(\mu|\mathbf{X})$.



##### Ans:

The likelihood function is $p(\mathbf{X|\mu})=\prod^N_{n=1}p(\mathbf{x_n}|\mu)=\frac{1}{(2\pi)^{DN/2}|\Sigma|^{N/2}}exp \{ \sum^{N}_{n=1} -\frac{1}{2}(\mathbf{x_n-\mu})^\mathrm{T}\Sigma^{-1}(\mathbf{x_n-\mu}) \}$ , and the prior distribution is $p(\mu)=N(\mu|\mu_0, \Sigma_0)$.  Here, the prior *p*(*µ*) is given by a Gaussian, it will be a conjugate distribution for this likelihood function because the corresponding posterior will be a product of two exponentials of quadratic functions of *µ* and hence will also be Gaussian. Thus, we suppose  $p(\mu|\mathbf{X})=N(\mu|\mu_N,\Sigma_N)$.  According to Bayesian Inference $p(\mu|\mathbf{X}) \propto p(\mathbf{X|\mu})p(\mu)$, we focus on the exponential term and rearrange it according to $\mu$. 

We have the exponential term of $p(\mathbf{X|\mu})p(\mu)$ like this:
$$
\begin{align}
&\sum^{N}_{n=1} -\frac{1}{2}(\mathbf{x_n-\mu})^\mathrm{T}\Sigma^{-1}(\mathbf{x_n-\mu}) -\frac{1}{2}(\mathbf{\mu-\mu_0})^\mathrm{T}\Sigma^{-1}_0(\mathbf{\mu-\mu_0})
\\&=-\frac{1}{2}\mu^\mathrm{T}(N\Sigma^{-1}+\Sigma_0^{-1})\mu+\frac{1}{2}\mu^\mathrm{T}(\Sigma^{-1}\sum_{n=1}^N\mathbf{x_n}+\Sigma_0^{-1}\mu_0)+\frac{1}{2}(\sum_{n=1}^N\mathbf{x_n}^\mathrm{T}\Sigma^{-1}+\mu_0^\mathrm{T}\Sigma^{-1})\mu-\frac{1}{2}(\sum_{n=1}^N\mathbf{x_n}^\mathrm{T}\mathbf{x_n}+\mu_0^\mathrm{T}\mu_0)
\end{align}
$$
and exponential term of $p(\mu|\mathbf{X})$ like this:
$$
\begin{align}
&-\frac{1}{2}(\mu-\mu_N)^\mathrm{T}\Sigma_N^{-1}(\mu-\mu_N)
\\&=-\frac{1}{2}\mu^\mathrm{T}\Sigma_N^{-1}\mu+\frac{1}{2}\mu^\mathrm{T}\Sigma_N^{-1}\mu_N
+\frac{1}{2}\mu_N^\mathrm{T}\Sigma_N^{-1}\mu-\frac{1}{2}\mu_N^\mathrm{T}\mu_N
\end{align}
$$
So, we have 
$$
\begin{align}
\Sigma_N^{-1}&=N\Sigma^{-1}+\Sigma_0^{-1}
\\\mu_N&=\Sigma_N(\Sigma^{-1}\sum_{n=1}^N\mathbf{x_n}+\Sigma_0^{-1}\mu_0)=(N\Sigma^{-1}+\Sigma_0^{-1})^{-1}(\Sigma^{-1}\sum_{n=1}^N\mathbf{x_n}+\Sigma_0^{-1}\mu_0)
\end{align}
$$
