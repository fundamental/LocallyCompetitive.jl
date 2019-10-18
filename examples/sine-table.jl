# Sine Dictionary based example
using PyPlot
using Statistics
using Random

include("../src/LocallyCompetitive.jl")
Random.seed!(0xcafebeef)

# First we create 'dim' basis functions, each consisting of 'n_elm' samples.
# Using the dictionary 'D' a set of output samples can be created.
# We will create a 'k' sparse signal 'y' and attempt to reconstruct it.
# For a basis function we will use a set of sine functions with the norm of each
# function set to 1.
# The sparse signal 'y' will have 'k' randomly chosen signals.
#
# LCA can effectively reconstruct 'y' original representation even with changes
# to the dictionary used to create it and if LCA attempts to reconstruct
# 'y'+noise it will attempt to impose sparsity on the solution

dim     = 64   #number of basis functions
n_elms  = 1024 #samples per function
D       = zeros(n_elems,dim);
k       = 3    #Sparsity factor
s       = zeros(dim)

for i=1:dim,
    j=1:n_elems
    D[j,i] = sin(2pi*j*i/n_elems)
end
LCA.unitary_columns!(D)

# The original sparse representation
s = rand(dim)*4
s[sortperm(randn(dim))[k+1:end]] .= 0

# The signal that is observed
y = D*s

noise   = randn(n_elems)

sparse_clean  = LCA.lca(y+0.0*noise, D, T_soft=0.20)
sparse_noise  = LCA.lca(y+0.2*noise, D, T_soft=0.25)

figure(1);PyPlot.clf()
subplot(3,1,1)
title("The original sparse elements")
ylabel("Magnitude")
stem(s)
subplot(3,1,2)
title("The reconstructed sparse elements")
ylabel("Magnitude")
stem(sparse_clean)
subplot(3,1,3)
title("The reconstructed elements under noisy observation")
xlabel("Dictionary Element")
ylabel("Magnitude")
stem(sparse_noise)

figure(2);PyPlot.clf()
title("The original sparse elements")
ylabel("Magnitude")
plot(D*s)
plot(D*sparse_clean)
figure(43);PyPlot.clf()
subplot(1,3,1)
plot(y+0.2*noise)
axis([0,n_elms, -0.6, 0.6])
subplot(1,3,2)
plot(y)
axis([0,n_elms, -0.6, 0.6])
subplot(1,3,3)
plot(D*sparse_noise)
axis([0,n_elms, -0.6, 0.6])

println("If LCA is working as expected, then the sparsity of the original")
println("signal input should be identical. For processing the signal with noise")
println("the sparsity should be similar though higher (sparsity will go down")
println("with higher soft_threshold values)")

println()

println("Sparsity of original signal = ", sum(s.!=0))
println("Sparsity of LCA on the clean observation = ", sum(sparse_clean.!=0))
println("Sparsity of LCA on the dirty observation = ", sum(sparse_noise.!=0))

println()
println("Although the sparsity of LCA on the dirty observation is higher,")
println("let's see what happens to the reconstruction error")
println()
println("Original     = ", norm(D*s-y), " (Defined to be zero)")
println("LCA clean    = ", norm(D*sparse_clean-y), " (Should be small)")
println("noisy signal = ", norm(y+noise-y),
        " (Defined to be the norm of the noise")
println("LCA dirty    = ", norm(D*sparse_noise-y),
        " (Much lower than the noisy observation)")
