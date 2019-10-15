function fitTensorOneSidedAlgorithm(tensorWarmStart,
    mpTrain,
    tensorBias,
    X, y, r;
    TH = 0.001, maxIter = 50, verbose = true,
    tensor = nothing, mpTest = nothing,
    tensorTrain = nothing)

    ℓ = length(tensorWarmStart)
    n,m = size(tensorWarmStart[1])
    n,p = size(X)

    println("Tensor One-Sided algorithm: rank = $r, γ = $y")

    if !isa(tensor, Nothing)
        baselineTensor = getBaselineTensor(tensorTrain)
        baselineSqError = getSqError(baselineTensor, tensor, mpTest)
    end

    W = [zeros(p,m) for _ in 1:ℓ];
    Z = deepcopy(tensorWarmStart);
    S = [zeros(r,r) for _ in 1:ℓ];
    U = zeros(n,r);
    V = zeros(m,r);

    R1 = deepcopy(tensorWarmStart);
    R2 = deepcopy(tensorWarmStart);

    t = 1
    for t in 1:maxIter
        dose = 1

        for dose in 1:ℓ
            R1[dose] = Z[dose] - (X * W[dose] .+ tensorBias[dose])
        end
        U = svd(hcat(R1...)).U[:,1:r];
        V = svd(vcat(R1...)).V[:,1:r];
        for dose in 1:ℓ
            S[dose] = U' * R1[dose] * V
        end

        for dose in 1:ℓ
            R2[dose] = Z[dose] - (U * S[dose] * V' .+ tensorBias[dose])
        end
        R2 = setMP(R2, mpTrain)

        X_n = Array{Float64,2}(X)

        W = genomicAlgorithm(R2,
                    X_n, y; verbose = false, W = W)

        Zprev = deepcopy(Z)
        for dose in 1:ℓ
            Z[dose][mpTrain[dose]] = (X * W[dose] + tensorBias[dose] + U * S[dose] * V')[mpTrain[dose]]
        end

        ΔMAE = mean(getMAE(Z, Zprev, mpTrain))
        if verbose
            print("Iteration $t: ΔMAE = ", round(ΔMAE, digits=5))
            if !isa(tensor, Nothing)
                sqError = getSqError(Z, tensor, mpTest)
                testR2 = 1 - mean(sqError ./ baselineSqError)
                print(", Test R2 = ", round(testR2, digits=5))
            end
            println()
        end

        if ΔMAE < TH
            println("Algorithm converged.")
            break
        elseif t == maxIter
            println("Max iteration limit reached.")
        end
    end

    return(Z)
end

function fitTensorOneSided(tensorTrain, X;
    rankList = [10],
    γList = [10.0^-i for i in 1:10],
    warmstart = "PiecewiseLinear",
    boundTensor = true, maxIter = 50,
    percentValid = 0.2, seed = 123,
    tensor = nothing, mpTest = nothing,
    tuneBias = false)

    ℓ = length(tensorTrain)
    n, m = size(tensorTrain[1])

    tensorTrain0 = fillZeroTensor(tensorTrain)
    mpTrain = getMP(tensorTrain)

    println("Tensor One-Sided Learning")
    println("Warmstart: ", warmstart)
    println("Gamma: ", γList)
    println("Rank: ", rankList)

    mpTrain = getMP(tensorTrain)
    mpValid = generateValidMP(mpTrain, percentValid, seed)
    tensorTrainMini = setMP(tensorTrain, mpValid)
    mpTrainMini = getMP(tensorTrainMini)
    tensorTrainMini0 = fillZeroTensor(tensorTrainMini)
    baselineTensorMini = getBaselineTensor(tensorTrainMini)
    baselineSqErrorMini = getSqError(baselineTensorMini, tensorTrain, mpValid)

    if tuneBias
        tensorBias = tuneTensorBias(tensorTrain)
    else
        tensorBias = [zeros(n,m) for _ in 1:ℓ]
    end

    if warmstart == "PiecewiseLinear"
        tensorWarmStart = linearInterp(tensorTrain)
        tensorTrainMiniWarmStart = linearInterp(tensorTrainMini)
    elseif warmstart == "DrugMeans"
        tensorWarmStart = getTensorDrugMeans(tensorTrain)
        tensorTrainMiniWarmStart = getTensorDrugMeans(tensorTrainMini)
    else
        tensorWarmStart = deepcopy(tensorTrain0)
        tensorTrainMiniWarmStart = deepcopy(tensorTrainMini0)
    end

    params = collect(Iterators.product(γList, rankList))[:]

    if (length(params) > 1)
        validR2 = fill(-Inf, length(params))

        for (i, (y, r)) in enumerate(params)
            Z = fitTensorOneSidedAlgorithm(
                tensorTrainMiniWarmStart,
                mpTrainMini,
                tensorBias,
                X, y, r;
                tensor = tensor,
                mpTest = mpTest,
                tensorTrain = tensorTrain)
            validSqError = getSqError(Z, tensorTrain, mpValid)
            validR2[i] = 1 - mean(validSqError ./ baselineSqErrorMini)
            println("Valid R2 = ", round(validR2[i], 5))
        end
        (y, r) = params[indmax(validR2)]
    else
        (y, r) = params[1]
    end

    Z = fitTensorOneSidedAlgorithm(tensorWarmStart,
        mpTrain,
        tensorBias,
        X, y, r;
        maxIter = maxIter,
        tensor = tensor,
        mpTest = mpTest,
        tensorTrain = tensorTrain)

    tensorImputed = deepcopy(tensorWarmStart)
    for dose in 1:ℓ
        tensorImputed[dose][mpTrain[dose]] = Z[dose][mpTrain[dose]]
    end

    if boundTensor
        minValue, maxValue = getExtremeValues(tensorTrain)
        tensorImputed = boundExtremeValues(tensorImputed,
            minValue, maxValue)
    end

    return(tensorImputed, y, r)
end


