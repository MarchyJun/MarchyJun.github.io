---
layout: article
title: Matrix Decomposition - (1) Decomposition Methods
mathjax: true
aside:
  toc: true
---


# 1. Eigenvalue & Eigenvector

All matrix has some information and the information must not be changed according to how we see the matrix : in rows, or in columns. Eigenvalue and Eigenvector are related to this essential information of its matrix.          
For any square matrix A, we define $\lambda$ and $\vec{v}$($\ne$ 0) satisfying $A\vec{v} = \lambda \vec{v}$ as eigenvalue $\lambda$ and eigenvector $\vec{v}$ of the matrix A. Now let's find eigenvalue and eigenvector of A(p by p).
$$ \,\\
\qquad 
A\vec{v} = \lambda \vec{v} \\ \, \leftrightarrow
(\lambda I_{p} - A)\vec{v} = 0 \\ \,\leftrightarrow
B\vec{v} = 0 \:\: (\,let \:\: \lambda I_{p} - A = B \,) $$
        
Now note that null space of B is $$N(B) = \{\vec{x} \in R^{p} \vert B\vec{x} = 0 \}$$. So $\vec{v} \in N(B)$. It means that $N(B) = N(\lambda I_{p} - A)$ is nontrivial.              
Also note that for matrix M, M's columns are linearly independent iff $N(B) = {0}$. Since $N(B) \ne {0}$, it means B's columns are not linearly independent. So $B = \lambda I_{p} - A$ is not invertible and $det(\lambda I_{p} - A) = 0$             
That is, 

$$ There\:\: is\:\: nonzero\:\: \vec{v}\:\: s.t\:\: A\vec{v} = \lambda \vec{v}\; \leftrightarrow\; det(\lambda I_{p} - A) = 0 $$   

There are up to p eigenvalues $\lambda_{1} \geqslant \lambda_{2} \geqslant \dots \geqslant \lambda_{p}$ of A, and for each eigenvalue $\lambda_{i}$, a correspond eigenvector $\vec{v}_{i}$ exists

# 2. Matrix Decomposition 

## 2.1. Eigenvalue Decomposition

Let A be n by n square matrix. Let's denote eigenvalue and eigenvector of A as $$\lambda_{i}, \vec{x}_{i}$$, and let $$S = \begin{bmatrix} \vec{x}_{1} & \vec{x}_{2} & \dots & \vec{x}_{n} \end{bmatrix}$$, 
$$\Lambda = \begin{bmatrix} \lambda_{1} & 0       & 0  \\
                           0           & \ddots  & 0  \\
                           0           & 0       & \lambda_{n} \end{bmatrix} $$.             
Then             
$$
AS 
= A\begin{bmatrix}           \vec{x}_{1} &            \vec{x}_{2} & \dots &            \vec{x}_{n} \end{bmatrix} \\ \quad\:\,
= \begin{bmatrix}           A\vec{x}_{1} &           A\vec{x}_{2} & \dots &           A\vec{x}_{n} \end{bmatrix} \\ \quad\:\,
= \begin{bmatrix} \lambda_{1}\vec{x}_{1} & \lambda_{2}\vec{x}_{2} & \dots & \lambda_{n}\vec{x}_{n} \end{bmatrix} \\ \quad\:\,
= \begin{bmatrix} \vec{x}_{1} & \vec{x}_{2} & \dots & \vec{x}_{n} \end{bmatrix} 
  \begin{bmatrix} \lambda_{1} & 0       & 0  \\
                           0           & \ddots  & 0  \\
                           0           & 0       & \lambda_{n} \end{bmatrix} \\ \quad\:\,
= S\Lambda $$
          
If $$S = \begin{bmatrix} \vec{x}_{1} & \vec{x}_{2} & \dots & \vec{x}_{n} \end{bmatrix}$$ are linearly independent, then $$det(S) \ne 0$$ and $$S^{-1}$$ exists, so $$A = S\Lambda S^{-1}$$   
               
That is, if A is n by n square matrix and eigenvectors are linearly independent, A can be decomposed as $$A = S\Lambda S^{-1}$$. 

## 2.2. Jordan Form

However, if eigenvectors of square matrix A are not linearly independent, we can not decompose A by above procedure. In this case, we have to change our goal from finding $$\Lambda$$ to finding most simiarl matrix with $$\Lambda$$. We call it as jordan form.

Let A be n by n square matrix and rank(A) = s. If A can not be decomposed by eigenvalue decomposition because of linearly dependence of eigenvectors of A, we can decompose A as similar jordan form : $$A = MJM^{-1}\: 
where\:\: J = \begin{bmatrix} J_{1} & 0      & 0 \\
                              0     & \ddots & 0 \\
                              0     & 0      & J_{s} \end{bmatrix}  
         = \begin{bmatrix} \lambda_{1} & 1           &             &             &             &           \\
                           0           & \lambda_{1} &             &             &             &           \\
                                       &             & \lambda_{2} & 1           & 0           &           \\ 
                                       &             & 0           & \lambda_{2} & 1           &           \\
                                       &             & 0           & 0           & \lambda_{2} &           \\
                                       &             &             &             &             & \ddots &  \\
                                       &             &             &             &             &        & \lambda_{s}
                              \end{bmatrix}$$
                               

For example, let A be 3 by 3 matrix with $\lambda$ multiplicity 3. That is, $$\lambda_{1} = \lambda_{2} = \lambda_{3} \overset{let}{=} \lambda$$. Let $$rank(A - \lambda I_{3}) = 2$$. Then, since $$rank(A - \lambda I_{3}) + nullity(A - \lambda I_{3}) = 3, \: nullity(A - \lambda I_{3}) = 1$$. It means $$(A - \lambda I_{3})\vec{y} = \vec{0}$$ has 1 solution and this $$\vec{y}$$ is eigenvector of A.           
Let $$\vec{v}_{1} = \vec{y}$$, then we know $$A\vec{v}_{1} = \lambda\vec{v}_{1}$$. Since there is only 1 eigenvector of A, for any other $$\vec{v}$$, $$(A - \lambda I_{3})\vec{v} \ne \vec{0}$$. But there is $$\vec{v}_{2}$$ s.t $$(A - \lambda I_{3})^{2}\vec{v}_{2} \ne \vec{0}$$            

Let's say genelize eigen vector of grade 2 as $$\vec{v}_{2}$$. Then 
$$\,(A - \lambda I)\vec{v}_{2} \ne \vec{0} \\ \,
   (A - \lambda I)^{2}\vec{v}_{2} = \vec{0}$$ 
      
Also, let's define generalized eigenvector of grade 3 as $$\vec{v}_{3}$$. Then 
$$\,(A - \lambda I)^{2}\vec{v}_{3} \ne \vec{0} \\ \, 
   (A - \lambda I)^{3}\vec{v}_{3} = \vec{0}$$.
      
So, $$ \begin{cases} A\vec{v}_{1} = \lambda\vec{v}_{1} \\
                     A\vec{v}_{2} = \vec{v}_{1} + \lambda\vec{v}_{2} \\
                     A\vec{v}_{3} = \vec{v}_{2} + \lambda\vec{v}_{2} \end{cases} \quad \rightarrow \vec{v}_{1}, \; \vec{v}_{2}, \; \vec{v}_{3} \:\: are \:\: linearly \:\: independent\\ $$
     
Let $$V = \begin{bmatrix}\vec{v}_{1} & \vec{v}_{2} & \vec{v}_{3} \end{bmatrix}$$, then $$V$$ is invertible.       
So, $$A = VJV^{-1} \:\: where\:\: J = \begin{bmatrix} \lambda & 1       & 0 \\
                                                     0       & \lambda & 1 \\ 
                                                     0       & 0       & \lambda \end{bmatrix}$$ 

## 2.3. Spectral Decomposition

If A is square and also symmetric matrix, then we can use spectral decomposition.          
Let A be n by n square and symmetrix matrix. Let's denote eigenvalue and eigenvector of A as $$\lambda_{i}, \vec{\gamma_{i}}$$, and let $$\Gamma = \begin{bmatrix} \vec{\gamma_{1}} & \vec{\gamma_{2}} & \dots & \vec{\gamma_{n}} \end{bmatrix}$$, 
$$\Lambda = \begin{bmatrix} \lambda_{1} & 0       & 0  \\
                           0           & \ddots  & 0  \\
                           0           & 0       & \lambda_{n} \end{bmatrix} $$. Then $$\: \Gamma \Gamma^{T} = \Gamma^{T} \Gamma  = I$$  

- If A is full rank, $$A = \Gamma \Lambda \Gamma^{T} \\ \;\:\,
                         = \sum_{i = 1}^{n}{\lambda_{i}\gamma_{i}\gamma_{i}^{T}} \\ $$
                               
- If $rank(A) = r < n$, $$A = \Gamma \Lambda \Gamma^{T} \\ \;\:\,
                           = \begin{bmatrix} \Gamma_{1} & \Gamma_{0} \end{bmatrix} 
                             \begin{bmatrix} \Lambda_{1} & 0 \\
                                              0          & 0 \end{bmatrix}
                             \begin{bmatrix} \Gamma_{1}^{T} \\ \Gamma_{0}^{T}\end{bmatrix} \\ \;\:\,
                           = \Gamma_{1}\Lambda_{1}\Gamma_{1}^{T} \\ \;\:\,
                           = \sum_{i = 1}^{r}{\lambda_{i}\gamma_{i}\gamma_{i}^{T}}\:\:
where\:\: \Gamma_{1} : (\,n, r\,),\:\: \Gamma_{0} : (\,n, n-r\,)\:\: \Lambda_{1} : (\,r, r\,) $$                 

## 2.4. Singular Value Decomposition

Eigenvalue decomposition, Jordan form, Spectral decomposition can be used only when matrix is square matrix. But singular value decomposition can be used regardless of a shape of matrix.     
           
Let A (n, p) has rank $$r \leqslant min(n, p)$$          
Then 
$$AA^{T}$$ : n by n square, symmetric matrix $$\rightarrow \:\: AA^{T} = \Gamma\Lambda\Gamma^{T} = \sum_{i=1}^{r}{\lambda_{i}\gamma_{i}\gamma_{i}^{T}} $$

$$\quad\;\;\;\; A^{T}A$$ : p by p square, symmetric matrix $$\rightarrow \:\: A^{T}A = \Delta\Lambda\Delta^{T} = \sum_{i=1}^{r}{\lambda_{i}\delta_{i}\delta{i}^{T}}$$      
      
And $$A = \Gamma\Sigma\Delta^{T} \\ \:\,\;
       = \Gamma_{1}\Sigma_{1}\Delta_{1}^{T} \\ \:\,\;
       = \sum_{i=1}^{r}{\lambda_{i}^{\frac{1}{2}}\gamma_{i}\delta_{i}} \qquad where\:\: 
\Gamma=\begin{bmatrix} \Gamma_{1}\,(n, r) \, \vert \, \Gamma_{0}\,(n,n-r) \end{bmatrix} 
      =\begin{bmatrix} \gamma_{1} & \dots & \gamma_{r} \,\vert\, \gamma_{r+1} & \dots & \gamma_{n} \end{bmatrix}\\ \qquad\qquad\qquad\qquad\qquad\quad\;\;\; 
\Delta = \begin{bmatrix} \Delta_{1}\,(p,r) \, \vert \, \Delta_{0}\,(p,p-r) \end{bmatrix} 
       = \begin{bmatrix} \delta_{1} & \dots & \delta_{r} \,\vert\, \delta_{r+1} & \dots & \delta_{p} \end{bmatrix}\\
\qquad\qquad\qquad\qquad\qquad\quad\;\;\;
\Sigma = \begin{bmatrix} \Sigma_{1} & 0         \\
                         0          & \Sigma_{0}\end{bmatrix}, \:\: \Sigma_{1} = diag\{\lambda_{1}^{\frac{1}{2}}, \dots \lambda_{r}^{\frac{1}{2}}\} 
                $$

Singular value decomposition is very useful in many ways: one example is image compression. Let's compress image by singular value decomposition.
