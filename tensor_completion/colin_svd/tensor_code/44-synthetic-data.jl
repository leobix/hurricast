using Distributions
using Random

function normalizeMatrix(B)
    (B .- mean(B, 1)) ./ std(B, 1)
end

function addNoise(X, σ; seed = 1)
    n, p = size(X)
    if σ > 0
        srand(seed + 1000)
        noise = rand(Normal(0, σ), n, p)
    else
        noise = zeros(n,p)
    end
    return(X + noise)
end

function generateSyntheticData(n,m,ℓ,r0,p,q; seed = 1)
    srand(seed)
    # U = (rand(n, r0))
    # S = [(rand(r0, r0)) for _ in 1:ℓ]
    # V = (rand(m, r0))
    U = rand(Normal(0, 1), n, r0)
    S = [rand(Normal(0, 1), r0, r0) for _ in 1:ℓ]
    V = rand(Normal(0, 1), m, r0)
    Z = [U * S[dose] * V' for dose in 1:ℓ]
    X = U[:,1:p]
    Y = V[:,1:q]

    return(Z, X, Y)
end

function generateSyntheticData(seed)
    n = 200
    m = 200
    ℓ = 10
    r0 = 20
    p = 20
    q = 20

    generateSyntheticData(n,m,ℓ,r0,p,q; seed = seed);
end
