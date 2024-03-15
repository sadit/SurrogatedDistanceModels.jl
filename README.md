# SurrogatedDistanceModels.jl

This package contains methods that can be used to replace big metric databases and its related metric function by a more light weight encoding and distance function. This package is designed to be used with [`SimilaritySearch.jl`](https://github.com/sadit/SimilaritySearch.jl) but can be used without it.

In particular, it contains the following methods/types:


- `BinPerms`: Binary encoding based on Brief permutations (with shift-based encoding)
- `BinPermsDiffEnc`: Binary encoding based on Brief permutations with differential encoding
- `HyperplaneEncoding`: Hyperplane-based binary encoding
- `HighEntropyHyperplanes`: Binary encoding based on high entropy hyperplanes
- `NearestReference`: String-based encoding using references
- `Perms`: Vector-based encoding of Permutations 
- `PCAProjection`: Vector-based encoding just using `PCA` from `MultivariateStats` package.
- `RandomProjection`: Vector-based encoding using Gaussian random projections.

All methods use `fit` and `predict` methods with reasonable default parameters.

TODO: Put citations for each method

