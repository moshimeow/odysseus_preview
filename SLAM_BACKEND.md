THIS IS STUFF SHENNA AND I TALKED ABOUT:
Frontend: We aren't doing that
It is a magic box that gives us 2d points that we turn into 3d points
Backend: Talked about a lot
We're doing the reverse of the 'typical' torch style autodiff, where instead of building the tree beginning at the parameters and terminating at the loss in forward mode and then doing a backwards pass doing the chain rule along the tree to get the derivative of each parameter w.r.t the loss, we're using 'jets' which store the value of a given parameter and the residuals associated with each operation, doing the chain rule to update the residuals to the correct value for each operation, until we calculate the loss, upon which we will have calculated the derivatives of each... parameter? with respect to the input variables? I think?
    - Moshi: Yes pretty much. I forgot to say that "forwards" autodiff - what we're doing - is great because allocating a graph is hard. "Reverse" autodiff (what PyTorch does) is less operations and makes sense on GPU since the GPU operations are a lot slower than creating the graph.
    - Moshi: No, it's the derivatives of each residual wrt. each {parameter, input variable} (those are synonyms) . (I hope it's not the other way around, but the point is what you said was equivalent to saying "the derivatives of each parameter with respect to each parameter" which is wrong)
Once we have the derivatives of each parameter w.r.t the loss, we can do basically gradient descent, but actually levenberg marquardt, which wikipedia claims interpolates between gradient descent and gauss newton, in order to optimize the parameters to solve the non-linear least squares error, minimizing the sum of the squared distance of each 3d point's 2d projected location given a estimated camera pose from its observed 2d location. Probably
I'm gonna be honest I don't really understand how jets are implemented, either version, although the static version seems like it's going to be the preferred version.
eventually fusing the SLAM estimate and the IMU data will be part of the backend
New points need to be initialized with triangulation and 'just left to float for a while'
we definetly want all of this to either be on the stack, or allocated onto the heap at startup and never again. That's actually very important.
We need a buttload of metrics and tests to figure out if our code is actually working like it's supposed to. There are some errors visible in the demo right now even though there probably shouldn't be, and tracking that down is important.
Future things:
* Inverse depth
* Read basalt slam paper (https://arxiv.org/abs/1904.06504 right?), understand it all
* Generally have a good approach for how to handle uncertainty / variance for all parameters






keyframing:
* every time half the features are old and half the features are new, add a keyframe
* if you have a new keyframe that's within 0.25m and 30deg of an old keyframe, delete the old one. (basically this helps us kill points that haven't been seen and don't seem to be coming back)
* if you have over 100 keyframes, delete the oldest one. (this is mainly for continuity purposes, best to forget the oldest stuff rather than have two disconnected maps)
    * moshi spitball: shouldn't this be specifically about the amount of memory we allocated for observations and keyframes? need to be more clear.
* points in the map not associated with a keyframe should be deleted
* why keyframing?
    * we need a way to optimize the whole map and close loops. we can a) have a prior that says "points shouldn't move much from where we originally saw them" but that sucks and doesn't work very well bc once we've been mapping for a long time we can get off by a lot, OR b) make "keyframes" that go into the optimization (maybe not every frame but often) that constrain us to being on the rest of the map.
    * moshi spitball: ok so optimizing every keyframe at once is still hard and sucks (and I think is what's called global bundle adjustment and isn't done often). I think we do need a way to basically say "we are only using the keyframes that share observations with us and only optimizing the points we can currently observe in this frame" so that the keypoints keep us grounded within the points they can see?
Shenna interpretation: Keyframes store a position, orientation, and a list of visible keypoints, and are used to manage which previous keypoints we care about. Sometimes we go back and use our new information to optimize our previous understanding of point positions, using the keyframes and the map. If we need to constrain which points we optimize we should use the points that we can see.


inverse depth parametrization:
* yup we just parametrize it as origin vector, bearing vector and inverse of distance _FROM THE CAMERA THAT FIRST OBSERVED IT_. I think the way to think about this is "if both cameras saw it at once, camera0. otherwise whichever one saw it first" 
Shenna: oh so we're storing depth as it's inverse (1/depth)? I think I understand part of why you would do that.


* memory:
    * what sucks is that "growing" might be correct as long as it's not in hot loops. having the normal code path allocate 2GB of memory before it can run is probably unacceptable.
    * we need a more nuanced move


