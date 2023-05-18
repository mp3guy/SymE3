# SymE3
SymE3 is a small library built on top of [SymPy](https://www.sympy.org/en/index.html) for computing symbolic derivatives of functions that involve leveraging the exponential map from elements $\delta$ of the Lie algebra $se(3)$ to elements of the Lie group $SE(3)$. Specifically, it is designed to be used in formulations where an update to the Lie algebra parameter $\delta$ is linearised around the previous estimate with the intention of exploiting the fact that the derivative of the exponential map w.r.t. its parameters at zero is the set of trivially computable _generator_ matrices. 

## Example
A lot of problems in 3D machine perception use this "trick" to simplify calculation of jacobian matrices. Automatic computation of such derivatives is not supported in SymPy natively, which is where this library comes in. It defines a suite of specialized types which can be assembled together to form full function expressions and subsequently take derivatives. It is best described with a concrete example, point-to-plane ICP:

```python
from SymE3.core import PointH, NormalH, LieAlgebra, LieGroup, TotalFunction, exp

n_ri = NormalH("{n_{r_i}}")
r_i = PointH("{r_i}")
l_i = PointH("{l_i}")
rhat_i = PointH("{\\hat{r}_i}")
That_rl = LieGroup("{\\hat{T}_{rl}}")
d = LieAlgebra("{\\delta}")

e = n_ri.transpose() * ((exp(d) * That_rl * l_i) - r_i)
print("Function:")
display(e)
e = e.subs(That_rl * l_i, rhat_i)

f = TotalFunction(e)

print("Substituted:")
display(f)

print("Expanded:")
fe = f.as_explicit()
display(fe)

print("Derivative w.r.t. delta:")
df_dd = f.diff(d)
display(df_dd)
```
Function:
```math
\displaystyle {n_{r_i}}^{T} \left(\mathrm{exp}\left({\delta}\right) {\hat{T}_{rl}} {l_i} - {r_i}\right)
```
Substituted:
```math
\displaystyle {n_{r_i}}^{T} \left(\mathrm{exp}\left({\delta}\right) {\hat{r}_i} - {r_i}\right)
```
Expanded:
```math
\displaystyle \left[\begin{matrix}{{n_{r_i}}_{0}} \left({{\hat{r}_i}_{0}} - {{r_i}_{0}}\right) + {{n_{r_i}}_{1}} \left({{\hat{r}_i}_{1}} - {{r_i}_{1}}\right) + {{n_{r_i}}_{2}} \left({{\hat{r}_i}_{2}} - {{r_i}_{2}}\right)\end{matrix}\right]
```
Derivative w.r.t. delta:
```math
\displaystyle \left[\begin{matrix}{{n_{r_i}}_{0}} & {{n_{r_i}}_{1}} & {{n_{r_i}}_{2}} & {{\hat{r}_i}_{1}} {{n_{r_i}}_{2}} - {{\hat{r}_i}_{2}} {{n_{r_i}}_{1}} & - {{\hat{r}_i}_{0}} {{n_{r_i}}_{2}} + {{\hat{r}_i}_{2}} {{n_{r_i}}_{0}} & {{\hat{r}_i}_{0}} {{n_{r_i}}_{1}} - {{\hat{r}_i}_{1}} {{n_{r_i}}_{0}}\end{matrix}\right]
```
As shown above, the function you want to take derivatives with respect to can be expressed in a natural symbolic form. From here, SymE3 takes care of all the hardwork involved in applying the $\frac{\partial exp(\delta)}{\partial\delta} \biggr\rvert_{\delta = 0}$ approximation. There are a number of example functions provided in unit tests and ipython notebooks. The tests can be run from the repo root with `python -m pytest test/*`.

## How does it work?
The application of the generator matrices can seem sort of like an ad-hoc unnatural "hack" when done manually, almost as if it "wills" extra columns in the jacobian into existence. However, it can also be expressed as a tensor product operation followed by a tensor contraction, which is exactly what SymE3 does. Most of the code is concerned with intercepting / redefining calls to the `diff` function to alter the expression in a way that applies the rules our special case needs before allowing the rest of the SymPy machinery to do its job. 

Internally the code also *verifies* the derivative it provides by also computing the numerical derivative (using [Sophus](https://github.com/strasdat/Sophus)) and comparing the result using realistic variable values. This helps eliminate uncertainty in the result that may have otherwise been introduced by the hacking required to make this all work within SymPy. 

## Limitations
Although there is a $log$ function defined, this is merely for identity operations with the $exp$ function. In general, derivatives of the exponential map evaluated away from 0 and of the logarithmic map at any point are **not** supported. If you have for example, an error defined over some $e \in se(3)$ such as [this](https://fzheng.me/2020/06/19/jacobian-g2o-edgese3expmap/), SymE3 will not work for you. In those cases I'd recommend manually leveraging the adjoint identity ([Equation 87](https://ethaneade.com/lie.pdf)), using numerical methods, or auto diff with something like [Ceres](https://github.com/ceres-solver/ceres-solver) and [Sophus](https://github.com/strasdat/Sophus). Checkout [Hauke Strasdat's thesis](http://hauke.strasdat.net/files/strasdat_thesis_2012.pdf) for more insights. 

Although extensively tested, this library is by no means rigorous. There are undoubtedly expressions which will break the logic; if you find one please send it on. 

## Further Reading
Here's a bunch of awesome material I found useful while working on this. 
 - [Hauke Strasdat's PhD Thesis](http://hauke.strasdat.net/files/strasdat_thesis_2012.pdf)
 - [Joan Solà et al.'s A micro Lie theory for state estimation in robotics](https://arxiv.org/abs/1812.01537)
 - [Tim Barfoot's SE(3) Identities](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17_identities.pdf)
 - [Fan Zheng's Derivation of Jacobians in g2o::EdgeSE3Expmap](https://fzheng.me/2020/06/19/jacobian-g2o-edgese3expmap/)
 - [Ethan Eade's Derivative of the Exponential Map](https://ethaneade.com/exp_diff.pdf)
 - [Ethan Eade's Lie Groups for 2D and 3D Transformations](https://ethaneade.com/lie.pdf)
 - [Ethan Eade's Lie Groups for Computer Vision](https://ethaneade.com/lie_groups.pdf)
 - [Tom Drummond's Lie groups, Lie algebras, projective geometry and optimization for 3D Geometry, Engineering and Computer Vision](https://www.dropbox.com/s/5y3tvypzps59s29/3DGeometry.pdf?dl=0)
 - [Ankur Handa's Simplified Jacobians in 6-DoF Camera Tracking](https://www.doc.ic.ac.uk/~ahanda/simjacob.pdf)
