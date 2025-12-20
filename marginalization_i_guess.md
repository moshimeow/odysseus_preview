
## Moshi References
https://copark86.github.io/post/2018-04-04-marginalization
https://en.wikipedia.org/wiki/Marginal_likelihood

stergios roumeliotis square root inverse filter
    https://www.roboticsproceedings.org/rss11/p08.pdf
    https://mars.cs.umn.edu/research/sriswf.php (maybe same work?)

https://arxiv.org/pdf/2109.02182 mateo!
    The problem is thorougly explained here: https://arxiv.org/abs/2109.02182
    That "square root marginalization" was probably the last big improvement from the original Basalt authors.
    I can't say I understand it fully yet because I've not sit down and do the derivations myself, but the broad idea of the issue is that for keeping a fixed size window of keyframes to perform the optimization, you need to remove (marginalize) old information. For this you introduce a marginalization term in the error function that tries to summarize the removed information (3.1.2). When forming this term and trying to solve it, the hessian matrix appears (3.2.1), and furthermore the schur complement appears in the process.
    In section 4 they explain that the hessian is problematic because it has a condition number that is squared compared to the jacobian and so it's easy for small deltas to produce very large changes in the estimated optimal state, and thus nonlinear optimization being an iterative method this means that even small increments (floating-point-precision small) in the state might perform a big jump over the solution we were looking, producing divergence.
    The paper then introduces a way to formulate everything from a square root perspective that helps alleviate this issues significantly, trying to always use the Jacobian instead of the Hessian to have a better condition number.
    Also, I'm not sure this particular "elegant" way of expressing the marginalization that the paper improves on is something that everyone uses, or just something that a particular line of systems is using. The paper cites a couple of systems like DSO that suffer from this problem, but it's far from a thorough list. For example I can't say for sure whether ORB-SLAM3 suffers from this or not (although I do remember it being super prone to diverge and crash).


"BAL dataset" - maybe good benchmark for slam backends??


Information Sparsification in Visual-Inertial Odometry
https://shsiung.github.io/assets/pdf/Hsiung18iros.pdf


https://vnav.mit.edu/material/24-SLAM2-FactorGraphsAndMarginalization-slides.pdf
https://robotics.stackexchange.com/questions/15464/why-do-we-need-a-marginalization-in-bundle-adjustment
https://en.wikipedia.org/wiki/Marginal_likelihood
from https://www.youtube.com/watch?v=5JCwMSZr4IM :
    YES, optimization uses "active residuals + a marginalization prior"
    OK, so what would the marginalization prior actually mean??
https://mars.cs.umn.edu/tr/attitude_gyroless.pdf

https://robotics.stackexchange.com/questions/8900/slam-why-is-marginalization-the-same-as-schurs-complement
https://gbhqed.wordpress.com/2010/02/21/conditional-and-marginal-distributions-of-a-multivariate-gaussian/

factor graphs for robot perception
https://www.cs.cmu.edu/%7Ekaess/pub/Dellaert17fnt.pdf

https://gitlab.freedesktop.org/mateosss/basalt

vocab:
"Normal equations" -- normal way to solve Ax=b? (why does this matterrr)
    Vs QR decomposition


## Moshi Intuition:

* Ok so we have this thing called the "linearization point" - my belief is that this is "last frame's solve." This one should be N_PARAMETERS_NEW long.
* Then, we have some matrix that is either like, hessian or jacobian or something that tells us how important it is to be close to the linearization point.

From Claude (PLEASE rewrite this, holy shit):
That's actually a pretty good way of thinking about it! Let me clarify and expand on your intuition:
Your Intuition (mostly correct!)
Pseudo-residuals: Yes! They absolutely change based on what the current solve estimates. They're essentially saying "here's how far you are from where the marginalized information thinks you should be." Information matrix: Exactly right - it tells you how strong/confident each constraint should be. Higher values = tighter constraint.
The Full Picture
Let me connect this to what's happening in the code:
Method 2: Matrix (Information Matrix)
// After marginalizing, we get:
MarginalizedPrior {
    hessian: H_marg,           // Information matrix
    rhs: b_marg,               // "Gradient" term
    linearization_point: x_lin // Where we computed the Schur complement
}

// In the solver:
// (J^T J + H_marg) δ = -(J^T r + b_marg)
What's happening: The marginalized Hessian H_marg gets added to J^T J, making the system "stiffer" in directions where old observations had information. The b_marg term pulls the solution toward the linearization point.
Method 3: Square Root (Pseudo-Residuals)
// After marginalizing, we get:
MarginalizedPrior {
    sqrt_information: L,       // Where H_marg = L^T L
    linearization_point: x_lin // Where we computed the Schur complement
}

// In the cost function:
r_prior = L * (x - x_lin)  // Pseudo-residual
J_prior = L                 // Pseudo-Jacobian
What's happening:
The pseudo-residual r_prior = L * (x - x_lin) measures how far you've deviated from the linearization point
When the solver computes J^T J, it does: J_prior^T * J_prior = L^T * L = H_marg
And J^T r becomes: L^T * L * (x - x_lin)
So the pseudo-residuals are literally synthesizing fake observations that encode the marginalized information!
Your Question: "Do they help constrain us to a certain region?"
YES! Think of it this way:
Without prior: The solver only cares about fitting the new observations
With prior: The solver also has to satisfy these "fake measurements" from the past
The pseudo-residuals are saying: "Based on everything we saw before, state x₁ should be around x_lin, with uncertainty described by L^-1"
The Key Insight
Both methods are doing the same thing mathematically, just represented differently:
Matrix method: Directly adds H_marg to the Hessian approximation → changes the curvature of the optimization landscape
Sqrt method: Adds pseudo-residuals whose J^T J reconstructs H_marg → adds fake measurements that pull you toward the right place
The sqrt method is clever because:
You don't need to change the solver API
The Cholesky factor L has better numerical properties (condition number is √κ instead of κ)
It's conceptually cleaner: "marginalization = adding synthesized measurements from the past"
Analogy
Imagine you're trying to find your position using GPS:
New observations: GPS satellites you can currently see
Prior from marginalization: "Ghost satellites" from the past that you can't see anymore, but you remember their constraints
The information matrix tells you how much to trust each ghost satellite. The pseudo-residuals tell you how far your current guess is from satisfying those ghost constraints. Does that help clarify the intuition?


* Ok, so then "how the fuck do we connect this to sparsity" HAHA I HAVE NO FUCKIN CLUE HEHEHEHEHE. It probably is something like this:
    * When we try to sparsify, linearization point doesn't change
    * Simply.... discard off-diagonal elements, maybe? Do something where the diagonal has to do with the norm of the row and column it's on ??? not sure.


Ok also. So a very simple marginalization thing is :

approx_hessian = J^T * J (remember that for a linear least squares problem, hessian is just jacobian squared)

Partitioning into blocks: We get h_oo, h_on, h_no, h_nn. Basically think about dividing it into quadrants where (eg) top left is all old state and bottom right is all new state. So we get two quadrants that are all old or new, and two that are "half old" and "half new"

Then, cholesky! Cholesky == LDLT == L * L^T (actually no, but maybe?) (https://en.wikipedia.org/wiki/Cholesky_decomposition)
This... turns the matrix into its lower triangular TIMES its lowar triangular squared?
So, our information matrix is all zeroes on the upper triangular? why?

So that is, technically, it. 


## Linalg things we need to know:

What is positive definite?
    !!!! This mainly has to do with the matrix being symmetric (ie. if you flip it over its diagonal it is the same).
    I think that J^T * J (approximate Hessian) is kinda super duper required to be positive definite. Whether or not the regular jacobian is positive definite is eluding me now - i think the answer is no since it doesn't have to be square?

What is Cholesky decomposition / L * L^T? Why does it matter for square root marginalization?
    Ok, so first of all it only works on positive definite matrices, eg. "covariance" or "information" or Hessian.
    Then... I guess 


What is inner product?

What does it mean to take the square root of a matrix? is it per cell?

Hessian matrix: why is it square? That feels wrong. 

