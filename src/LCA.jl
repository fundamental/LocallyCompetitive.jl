module LCA
export lca

#Make all columns of the input matrix have a norm of 1
# D - input matrix (modified in place)
function unitary_columns!(D::Matrix{Float64})
    for i = 1:size(D,2)
        D[:,i] = D[:,i]/norm(D[:,i]);
    end
    nothing
end

#Perform the positive/negative soft threshold function on each element
# a - result vector (modified in place)
# u - input vector
# T - threshold
function soft_threshold!(a::Vector{Float64}, u::Vector{Float64}, T::Float64)
    for i = 1:length(u)
        if abs(u[i]) > T
            a[i,1] = u[i]-sign(u[i])*T;
        else
            a[i,1] = 0;
        end
    end
    nothing
end

#Perform Locally Competitive Algorithm for sparse recovery
# s      - Input to be approximated
# D      - Dictionary
# T_soft - Soft threshold value (higher promotes more sparseness)
# iter   - maximum number of iterations
# τ      - 1/learning rate
function lca(s::Vector{Float64}, D::Matrix{Float64}; T_soft::Float64=0.1,
             iter::Int=10000, τ::Float64=100.0)
    #Impose Unitary norm
    unitary_columns!(D)

    # Inhibition Matrix
    G = D'*D;
    for i=1:size(G,1)
        G[i,i] = 0; 
    end

    # Initialize LCA Parameters
    b = D'*s; # initial projection / excitatory input

    u = zeros(size(b)); # initial state of nodes = 0
    a = zeros(size(b)); # initial sparse rep = 0          

    for i=1:iter
        # soft thresholding function
        soft_threshold!(a, u, T_soft)

        # node dynamics
        Δu = (b  - u - G*a)/τ; 
        u += Δu;

        if(norm(Δu) < 1e-5)
            break
        end
    end

    return a
end

if(false)
    # Random Dictionary based example
    using PyPlot

    dim     = 64
    n_elems = 1024;
    D       = randn(dim,n_elems);
    s       = randn(dim)
    sparse  = lca(s , D, T_soft=0.4)

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

    figure(2)
    PyPlot.clf()
    title("Reconstruction of random input")
    plot(s)
    plot(D*sparse)
    legend(["Original", "Recovered"])
end

end # module
