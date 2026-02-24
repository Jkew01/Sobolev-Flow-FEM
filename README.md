# Sobolev-Flow-FEM

This program applies the finite element method with C1 elements vector elements to solve the Sobolev Gradient Flow numerically. The program comes with a few example curves to run with and has some parameters to set at the start of the program corresponding to model parameters and time step/mesh size.

n -> nodes in mesh
a,lda (lambda) -> model parameters
dt -> time step

Note that you should also configure the viewport appropriately to your given initial curve. Since this is a shrinking model, it suffices to ensure the whole initial curve is in frame. There is a class for this to edit the x and y intervals.
