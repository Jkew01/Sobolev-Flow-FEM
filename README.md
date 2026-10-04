# Sobolev-Flow-FEM

This program applies the finite element method with C1 elements vector elements to solve the Sobolev Gradient Flow numerically. The program comes with a few example curves to run with and has some parameters to set at the start of the program corresponding to model parameters and time step/mesh size.

n -> nodes in mesh
a,lda (lambda) -> model parameters
dt -> time step

Note that you should also configure the viewport appropriately to your given initial curve. Since this is a shrinking model, it suffices to ensure the whole initial curve is in frame. There is a class for this to edit the x and y intervals.

A direct weak formulation method has now been cooked up. I recommend using that one over the original. It is a lot faster and I suspect a lot more accurate. For example it now produces a believeable flow of the spiral example included in the code. Whats curious is comparison with curve shortening flow highlights the absence of a smoothing effect in this H^1(ds) flow. Indeed, for the spiral which is piecewise smooth, there exist four unsmooth points which are preserved by this flow whereas they are smoothed by csf. Note that this new better script is only for the standard H^1(ds) Sobolev gradient flow and not the homogeneous one.
