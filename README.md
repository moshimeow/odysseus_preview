# Odysseus CV

Made by nobody, provided for everybody.

# Don't be complicit in the tragedy of the commons!

I, Moshi Turner, am putting this first because I think that information should be free to all.

In the corporate software world, I see this pattern a lot.
* A useful MIT licensed project comes out
* It is copied into a corporate codebase and modified inside that codebase
* Those modifications are never even submitted to the original project
* Every time engineers are asked to update the project to latest, they now have a bunch of work each time to port their internal modifications forward, AND nobody else in the world can benefit from what they did
* This is usually not for a particularly critical component, and it's usually because bad management doesn't understand where the value of their codebase lies and wants to maintain ownership over all the *code* instead of maintaining ownership ver their *relationships* with their employees, processes, customers, stakeholders, etc. and being able to deliver value quickly in the real world. Companies only have value in what they can do in the real world!

My overarching point is that I really want you to use this, but I also really want you to understand that this project is the blood, sweat, tears and late nights of a bunch of random people who really want to make the world better. We published this for you, it's up to you to do the right thing for yourself and for the world, and push your management to let you contribute changes back to this project.

So. Yes, we (will probably make) this project MIT. Yes, you (will be) allowed to copy it and do whatever you want with it, and we fundamentally believe in your *right* to do so. However, we also fundamentally believe that sharing knowledge is everyone's *responsibility* to build a better world, and for this project we are putting a lot of good faith on the line hoping that everybody involved will take their responsibility here seriously.


# What is this?

This is an attempt at reimplementing important building blocks of C++ computer vision systems in Rust. The overall goals are something like
* Something similar to Eigen3: reasonably powerful optimized math library.
    * Bits of this already exist in eg. nalgebra and sprs, and we are just going to use those where possible
    * One thing we are doing in this repo (for now?) is our own 3D math library. What we need is currently nebulous enough that it's worth having our own til it's better defined
* Something similar to Ceres. This is mostly done in odysseus-solver.
* Something similar to Basalt. We have a decent start at a SLAM backend in odysseus-slam.
* Learning resources to show people how forwards autodiff, nonlinear optimization, marginalization, sparse matrix operations, and various other CV tools work.


Some random other disclosures: We are heavily using Claude Code to write and test our code. Neither of us are like super locked in SLAM PhDs *or* rust developers and a lot of this represents our first attempts at these things. Please do not trust this code with anything important.


# Contributing, Code of Conduct

## CLA
For now, until we are more confident about what we are doing, we require that all contributors sign a CLA that allows us to relicense *new* releases of this software at any license under our discretion (Moshi Turner and Shenna Summerfelt). (What legalese do we need about electronic signatures?) To sign a CLA, please sign ContributorLicenseAgreement.pdf with eg. Preview on MacOS, and (@shenna can you recommend a good PDF sign / editor for Linux?) and submit a PR that adds yours to SIGNED_CLAS.

## Code of conduct
For now, we adopt https://www.contributor-covenant.org/version/3/0/code_of_conduct/. Do not do anything that would make us write more in this section.

## Project contacts
Moshi Turner
@moshi.53 on signal
First name, last name with no caps or spaces at protonmail dot com

Shenna Summerfelt
quantum one thousand and two at gee mail dot com

