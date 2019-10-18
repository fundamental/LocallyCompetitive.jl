using LocallyCompetitive
using Test
using Random
using Statistics

# Testing params
M = 100
N = 100

k = 10

Random.seed!(0)

# Dictionary
basis = randn(M,N)

# Make k sparse signal
vals  = rand(M)
vals[sortperm(rand(M))[k+1:end]] .= 0

# Input signal with k sparse structure
sig = basis*vals

# Learn 'vals' based upon sig and the basis
out = LocallyCompetitive.lca(sig, basis)

recon = basis*out

sparse_orig = vals .!= 0
sparse_new  = out  .!= 0

sse_orig  = mean(recon.^2)
sse_new   = mean(sig.^2)
sse_diff  = mean((sig.-recon).^2)

# Same sparsity found in output
@test sparse_orig == sparse_new

# Reconstruction models data
@test sse_orig*0.01 > sse_diff
@test sse_new *0.01 > sse_diff
