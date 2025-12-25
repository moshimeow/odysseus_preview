


MOSHI IS WRITING THIS ON FIRST NIGHT OF UNDERSTANDING:

Ok so how does this work?

Imagine we are an IMU going through space. We don't know where we came from, we don't know where we're doing (cotton eye joe). We don't know where we are or where we aren't.

What can we know?

* Relative position!!!
    *Let's say we know our orientation kinda-well (hint: well enough that we can do small-angle approximations and be OK.). What do we know?
    * We can say that "if our position and velocity were 0, we should now be at position X and velocity Y at the end of our preintegration period."
    * Ok, how is that helpful?
    * Well, if our solver gives us an initial position, we can just add that as a constant to the final position estimate.
    * Similarly, if our solver gives us an initial velocity, we can add our initial velocity as a constant.
    * Ok, but what about initial velocity contribution to position?
    * Well actually that's also not bad. It's just initial_velocity * integration_period_time - just add that to the final estimate as well.
* Relative velocity!!!
    * This is actually exactly the same, except for it's easier. Assuming we are integrating actual angular velocity and we aren't wrong about biases, we do really get a relative orientation out of this that we can apply to any initial poses.

Ok, but what about IMU biases? Don't they affect this a bunch?

YES! Totally!!!

Things to think about:
* Gyro biases affecting orientation estimate, affecting accelerometer measurements getting rotated into world space
    * A thing I didn't quite say before is that we need to integrate accelerometer measurements in world space to get a useful answer. It's basically like, measurements at the IMU are affected by a nonlinearity: the device's orientation!
    * Ok sure. So, is this okay? yes, as long as we aren't so wrong about the *gyro* biases that our orientation is super off
* Accel biases being incorrect, and since they're rotated by the gyro it's hard to know where we should have been given the correct biases
    * Aaaa hard to think about; just not gonna explain this one. There might be something better you can do, but agh.


Alright, forget that example. That is 1000% a weird thought experiment. Let's go to something else, then I'll call back to that example.


Ok, so imagine we do this preintegration thing. Given some IMU biases and some samples to integrate (as in fundamental theorem of calculus integrate), we can get a relative pose, and a covariance on that pose. How is this useful in our optimizer if we want to optimize IMU biases?

Ready for this bullshit?

We have ANOTHER layer of linearization. Yes, ANOTHER. In our code, before the core optimization loop happens, we do this.
* ok we start with this initial guess of IMU biases
* compute the relative pose and relative covariance
* then compute the FUCKING JACOBIAN FOR THAT. Function inputs are biases, function outputs are relative pose. That jacobian.
* Then, in the optimizer, we let the IMU biases float, and then our "pseudo observation" here is WE START AT THE LINEARIZATION POINT OF THE PREINTEGRATION, MOVE ALONG THE BIAS AXES, AND USE THE JACOBIAN TO COME UP WITH A SMALL DELTA FROM THE LINEARIZATION POINT. THEN WE JUST USE ( CURRENT CAMERA POSE - (LAST CAMERA POSE * PREINTEGRATION RESULT) ) as an observation
    * Note I said nothing about how the variance on this observation is computed.