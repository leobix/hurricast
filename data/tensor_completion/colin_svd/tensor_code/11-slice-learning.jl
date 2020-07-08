using LinearAlgebra

function sliceIteration(tensor, r::Int)
    newTensor = deepcopy(tensor)
    ℓ = length(tensor)

    # matrixWide = hcat(newTensor...)
    U = svd(hcat(newTensor...)).U[:,1:r]

    # matrixTall = vcat(newTensor...)
    V = svd(vcat(newTensor...)).V[:,1:r]

    Pu = U * U'
    Pv = V * V'

    for dose in 1:ℓ
        newTensor[dose] = Pu * newTensor[dose] * Pv
    end

    return(newTensor)
end

function sliceLearningAlgorithm(tensorTrain, tensorWarmStart, r;
    T = 10, verbose = true, TH = 0.001,
    tensor = nothing, baselineSqError = 0,
    mpTest = nothing,
    imputeMiddle = false)

    newTensor = deepcopy(tensorWarmStart)
    if imputeMiddle
        mp = getMP(tensorTrain)
    else
        mp = getToughMP(tensorTrain)
    end
    ℓ = length(tensorTrain)

    if verbose
        println("Slice learning algorithm with rank = $r")
    end

    iterations = 1
    for t in 1:T
        sliceTensor = sliceIteration(newTensor, r)
        fullMAE = getMAE(newTensor, sliceTensor, mp)
        ΔMAE = mean(fullMAE[.!isnan.(fullMAE)])

        for dose in 1:ℓ
            newTensor[dose][mp[dose]] = sliceTensor[dose][mp[dose]]
        end

        if verbose
            print("Iteration: $t, ΔMAE = $(round(ΔMAE, digits=3))")
        end

        if !isa(tensor, Nothing)
            sqError = getSqError(newTensor, tensor, mpTest)
            R2 = mean(1 .- sqError ./ baselineSqError)
            println(", Test R2 = $(round(R2, digits=4))")
        else
            println()
        end

        if ΔMAE < TH
            println("Algorithm Converged.")
            break
        end

        iterations += 1
    end

    if iterations == T
        println("Maximum iteration limit reached.")
    end

    return(newTensor)
end

function sliceLearning(tensorTrain;
    tensor = nothing, mpTest = nothing,
    rankList = 14:14, percentValid = 0.2, seed = 123,
    verbose = true, boundTensor = true, T = 50,
    warmstart = "PiecewiseLinear",
    imputeMiddle = false)

    mpTrain = getMP(tensorTrain)
    mpValid = generateValidMP(mpTrain, percentValid, seed)
    tensorTrain0 = setMP(tensorTrain, mpValid)

    if warmstart == "PiecewiseLinear"
        tensorTrainWarmStart0 = linearInterp(tensorTrain0)
        tensorWarmStart = linearInterp(tensorTrain)
    else
        tensorTrainWarmStart0 = getTensorDrugMeans(tensorTrain0)
        tensorWarmStart = getTensorDrugMeans(tensorTrain)
    end

    baseline = getMean(tensorTrain)

    if !isa(tensor, Nothing)
        baselineSqError = getBaselineSqError(baseline, tensor, mpTest)
    else
        baselineSqError = fill(Inf, length(tensorTrain))
    end

    if verbose
        println("Slice Learning, Rank ($rankList)")
    end

    if length(rankList) > 1
        MAE = fill(Inf, length(rankList))

        for (i, r) in enumerate(rankList)
            tensorImputed = 
                sliceLearningAlgorithm(tensorTrain0, tensorTrainWarmStart0, r;
                    tensor = tensor,
                    baselineSqError = baselineSqError,
                    mpTest = mpTest,
                    imputeMiddle = imputeMiddle)
            MAE[i] = mean(getMAE(tensorImputed, tensorTrain, mpValid))
            if verbose
                println("Rank = $r, MAE = $(round(MAE[i], digits=5))")
            end
        end

        r = rankList[argmin(MAE)]
    else
        r = rankList[1]
    end

    tensorImputed = sliceLearningAlgorithm(tensorTrain,
                        tensorWarmStart,
                        r; T = T,
                        tensor = tensor,
                        baselineSqError = baselineSqError,
                        mpTest = mpTest,
                        imputeMiddle = imputeMiddle)

    if boundTensor
        minValue, maxValue = getExtremeValues(tensorTrain)
        tensorImputed = boundExtremeValues(tensorImputed,
            minValue, maxValue)
    end

    return(tensorImputed, r)
end
