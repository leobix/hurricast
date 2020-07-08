function getBias(tensor, λ1, λ2)
    mp = getMP(tensor)
    ℓ = length(tensor)
    n,m = size(tensor[1])
    μ = [mean(tensor[dose][.!mp[dose]]) for dose in 1:ℓ]

    numCellLines = [sum(.!mp[dose], dims=1) for dose in 1:ℓ]
    numDrugs = [sum(.!mp[dose], dims=2) for dose in 1:ℓ]

    drugBias = [zeros(m) for _ in 1:ℓ]
    cellLineBias = [zeros(n) for _ in 1:ℓ]

    for dose in 1:ℓ
        for j in 1:m
            drugBias[dose][j] = sum(tensor[dose][.!mp[dose][:,j],j] .- μ[dose]) /
                (numCellLines[dose][j] + λ1)
        end
    end

    for dose in 1:ℓ
        for i in 1:n
            cellLineBias[dose][i] = sum((tensor[dose][i,:] .- drugBias[dose])[.!mp[dose][i,:]] .- μ[dose]) /
                (numDrugs[dose][i] + λ2)
        end
    end

    return(μ, drugBias, cellLineBias)
end

function getTensorBias(μ, drugBias, cellLineBias)
    ℓ = length(μ)
    n = length(cellLineBias[1])
    m = length(drugBias[1])
    tensorBias = [zeros(n,m) for _ in 1:ℓ]
    tensorDrugBias = [repeat(c', n, 1) for c in drugBias]
    tensorCellLineBias = [repeat(r, 1, m) for r in cellLineBias]
    for dose in 1:ℓ
        tensorBias[dose] = μ[dose] .+ tensorDrugBias[dose] .+ tensorCellLineBias[dose]
    end

    return(tensorBias)
end

function tuneTensorBias(tensor;
    λ1List = [0; [2.0 ^ i for i in 0:10]],
    λ2List = [0; [2.0 ^ i for i in 0:10]])
    mp = getMP(tensor)
    mpTest = generateMP(mp, 0.2, NaN, 123)
    tensorTrain = setMP(tensor, mpTest)
    baselineTensor = getBaselineTensor(tensorTrain)
    baselineSqError = getSqError(baselineTensor, tensor, mpTest)
    
    params = collect(Iterators.product(λ1List, λ2List))[:]
    R2 = fill(-Inf, length(params))
    maxR2 = -Inf

    for (i, (λ1, λ2)) in enumerate(params)
        μ, drugBias, cellLineBias = getBias(tensorTrain, λ1, λ2)
        tensorBias = getTensorBias(μ, drugBias, cellLineBias)
        sqError = getSqError(tensorBias, tensor, mpTest)
        R2[i] = mean(1 .- sqError ./ baselineSqError)
        if R2[i] > maxR2
            println("λ1 = $λ1, λ2 = $λ2, R2 = ", round(R2[i], digits=5))
            maxR2 = R2[i]
        end
    end

    #change NaN values to -Inf
    for i in 1:length(R2)
        if isnan(R2[i])
            R2[i] = -Inf
        end
    end

    λ1, λ2 = params[argmax(R2)]
    μ, drugBias, cellLineBias = getBias(tensor, λ1, λ2)
    tensorBias = getTensorBias(μ, drugBias, cellLineBias)

    return(tensorBias)
end
