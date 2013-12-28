GPUEularSolver2D
================

A 2D Eular Solver on GPU platform.

We implement a unstructured CFD solver on GPU. 
In considered to the features of the Fermi architecture, we re-optimized the CFD solver on GPU and get more speed up. 
The main contribution of this paper is: 

1. We summarize the workflow of a FVM (Finite Volume Method) solver based on the unstructured grid, and introduce the development of parallel process unit, especially the changes of GPU from a graphic unit to a parallel processor. The new features of the Fermi architecture, as also detailed as a parallel processor. 
2. Some optimization methods are used on the unstructured grid CFD solver on GPU. A two-dimension unstructured grid CFD solver has been implemented on GPU, which uses Euler function and FVM. Then the program is optimized using many methods, such as changing array-of-structure to structure-of-array, decreasing data transfer, unrolling loops, optimizing instructions.
3. We have made a special effort to analyze the performance bottleneck of our un unstructured grid solver. In consideration to the features of the Fermi architecture, we propose a new method fit to the unstructured grid. The storage sequence of unstructuredgrid data is adjusted, and the "occupancy" of warps is reduced to explore more potentialities of GPU and gain much more speedup. Experimental results show that the GPU program finally gains about 40X speedup after the optimization, in contract to the initial 10X speedup in the beginning.
