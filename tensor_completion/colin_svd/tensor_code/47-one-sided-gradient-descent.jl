function getKnownPairs(matrix1)
    Ω = .!isnan.(matrix1)
    #Ωmatrix = hcat(ind2sub(Ω, find(Ω))...)    
    Ωmat = hcat(Tuple(CartesianIndices(Ω)[(LinearIndices(Ω))[findall(Ω)]])...)
    Ωmatrix = transpose(vcat([i[1] for i in Ωmat], [i[2] for i in Ωmat]))


    numPairs = size(Ωmatrix, 1)
    #knownPairs = Vector{Vector{Int}}(numPairs)
    knownPairs = Array{Array{Int, 1}}(undef, numPairs)

    for i in 1:numPairs
        knownPairs[i] = [Ωmatrix[i,1], Ωmatrix[i,2]]
    end

    return knownPairs
end

function getFull∇w(γ, w, X, z, knownPairs)
    numPairs = length(knownPairs)
    result = (2 / γ) * w

    for (i, j) in knownPairs
        result[:,j] += 2 * X[i,:] * (X[i,:]' * w[:,j] - z[i,j])
    end
    
    return(result)
end

function acceleratedGradDescent(γ, X::Matrix{Float64},
    z::Matrix{Float64},
    Ω::BitArray{2},
    knownPairs; maxGradSteps::Int = 1000,
    ν0 = 0.001,
    verbose = true)
    n, p = size(X);
    n, m = size(z)
    w = zeros(p, m);
    wprev = zeros(p, m);
    numPairs = length(knownPairs);

    loss(γ, w, X, z, Ω) = sum((z - X * w)[Ω] .^ 2) + (numPairs / γ) * dot(w, w)

    t = 1
    for t in 1:maxGradSteps
        wavg = w + ((t - 1) / (t + 2)) * (w - wprev)
        ∇wavg = getFull∇w(γ, wavg, X, z, knownPairs)
        ∇wavg2 = dot(∇wavg, ∇wavg)
        ν = ν0
        while loss(γ, wavg - ν * ∇wavg, X, z, Ω) > loss(γ, w, X, z, Ω) - (ν / 2) * ∇wavg2
            ν *= 0.5
            if ν < 1e-20
                break
            end
        end
        
        if ν < 1e-20
            break
        end

        wprev = w
        w = wavg - ν * ∇wavg
        # println("ν = $ν")
        if verbose
            println("Loss = ", loss(γ, w, X, z, Ω))
        end
    end
    if verbose
        println("Loss = ", loss(γ, w, X, z, Ω))
    end
    
    return(w)
end

function genomicAlgorithm(tensorTrain,
    X, y; verbose = true,
    W = nothing)
    
    ℓ = length(tensorTrain)
    n, p = size(X);
    n, m = size(tensorTrain[1]);
    if isa(W, Nothing)
        W = [zeros(p, m) for _ in 1:ℓ];
    end

    mp = getMP(tensorTrain)

    if verbose
        println("Genomic-Drug algorithm with γ = $y")
    end

    dose = 1
    for dose in 1:ℓ
        if verbose
            println("Dose = ", dose)
        end
        z = tensorTrain[dose] #- tensorBias[dose]
        knownPairs = getKnownPairs(tensorTrain[dose])
        Ω = .!isnan.(tensorTrain[dose])
        W[dose] = acceleratedGradDescent(y, X, z, Ω,
            knownPairs; verbose = verbose)
    end

    return(W)
end

function genomicLearning(tensorTrain, X;
    γList = [0.1, 0.01, 0.001, 0.0001],
    percentValid = 0.2,
    boundTensor = true,
    seed = 123,
    verbose = false,
    tuneBias = false)
    
    ℓ = length(tensorTrain)
    mpTrain = getMP(tensorTrain)
    mpValid = generateValidMP(mpTrain, percentValid, seed)
    tensorTrainMini = setMP(tensorTrain, mpValid)
    genomic = deepcopy(tensorTrainMini)
    baselineTensorMini = getBaselineTensor(tensorTrainMini)
    baselineSqErrorMini = getSqError(baselineTensorMini, tensorTrain, mpValid)
    tensorDrugMeans = getTensorDrugMeans(tensorTrain)
    tensorDrugMeansMini = getTensorDrugMeans(tensorTrainMini)
    genomic = deepcopy(tensorTrain)
    if tuneBias
        tensorBias = deepcopy(tensorDrugMeans)
        tensorBiasMini = deepcopy(tensorDrugMeansMini)
    else
        tensorBias = [zeros(n,m) for _ in 1:ℓ]
        tensorBiasMini = [zeros(n,m) for _ in 1:ℓ]
    end

    tensorTrainCentered = deepcopy(tensorTrain)
    tensorTrainCenteredMini = deepcopy(tensorTrainMini)
    for dose in 1:ℓ
        tensorTrainCentered[dose] = tensorTrain[dose] - tensorBias[dose]
        tensorTrainCenteredMini[dose] = tensorTrainMini[dose] - tensorBiasMini[dose]
    end

    # tensorTrain = deepcopy(tensorTrain)
    # tensorTrainMini = deepcopy(tensorTrainMini)
    

    validR2 = fill(-Inf, length(γList))

    if (length(validR2) > 1)
        for (i, γ) in enumerate(γList)
            W = genomicAlgorithm(tensorTrainCenteredMini,
                X, γ, verbose = verbose)
            for dose in 1:ℓ
                genomic[dose][mpValid[dose]] = (X * W[dose] + tensorBiasMini[dose])[mpValid[dose]]
            end
            validSqError = getSqError(genomic, tensorTrain, mpValid)
            validR2[i] = 1 - mean(validSqError ./ baselineSqErrorMini)
            println("γ = $γ, Valid R2 = ", round(validR2[i], 5))
        end
        γ = γList[indmax(validR2)]
    else
        γ = γList[1]
    end

    # tensorTrainWarmStart = linearInterp(tensorTrain)
    W = genomicAlgorithm(tensorTrainCentered,
        X, γ; verbose = verbose)
    for dose in 1:ℓ
        genomic[dose][mpTrain[dose]] = (X * W[dose] + tensorBias[dose])[mpTrain[dose]]
    end

    if boundTensor
        minValue, maxValue = getExtremeValues(tensorTrain)
        tensorImputed = boundExtremeValues(genomic,
            minValue, maxValue)
    end

    return(tensorImputed, γ)
end

