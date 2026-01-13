# SO3 Equivariant Graph Transformer

Experimental, independent re-implementation of the Equiformer neural network.

## Status
This repository contains a runnable research prototype of an SO(3)-equivariant graph neural network.
The code is intended for testing theoretical properties on synthetic data and has not been validated
on real-world applications. Results should not be interpreted as performance benchmarks.

## Attribution
This repository contains an independent re-implementation of the method introduced in:

Liao, Y. and Smidt, T., *Equiformer: Equivariant Graph Attention Transformer for 3D Atomistic Graphs*  
arXiv:2206.11990

The code was written from scratch based on the published paper and does not reuse the original
implementation. Parts of the attention mechanism are inspired by the e3nn documentation:
https://docs.e3nn.org/en/stable/guide/transformer.html

## Description
The neural network takes as input a point cloud represented as a graph, where each node corresponds
to a point and is associated with a set of features organized according to the irreducible
representations of the rotation group SO(3). Graph edges are constructed based on a distance
threshold using point coordinates. The network performs equivariant message passing between node
features using a combination of spherical harmonics, an attention mechanism, and nonlinear gating
for higher-order features.

The code consists of a single runnable file that tests the equivariance of the neural network on
randomly generated graphs.


## Run
python3 standalone_Equiformer_w_batch.py
