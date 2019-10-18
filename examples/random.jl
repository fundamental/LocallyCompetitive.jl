# Random Dictionary based example
using PyPlot
using Statistics

include("../src/LocallyCompetitive.jl")

dim     = 64
n_elems = 1024;
D       = randn(dim,n_elems);
s       = randn(dim)
sparse  = LCA.lca(s, D, T_soft=0.4)

println("Original  Sparsity = ", sum(s.!=0))
println("Original  RMS      = ", sqrt(mean(s.^2)))
println("Resulting Sparsity = ", sum(sparse.!=0))
println("Recovery  RMS      = ", sqrt(mean((s.-D*sparse).^2)))


figure(1)
PyPlot.clf()
title("Sparse Realization of input")
xlabel("Dictionary Element")
ylabel("Magnitude")
stem(sparse)
savefig("lca-example.png")

figure(2)
PyPlot.clf()
title("Reconstruction of random input")
plot(s)
plot(D*sparse)
legend(["Original", "Recovered"])
