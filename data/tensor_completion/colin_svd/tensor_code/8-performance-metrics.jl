function getTotalMSE(tensor1, tensor2, mpValid)
    ℓ = length(tensor1)
    totalSum = 0
    totalNum = 0
    
    for dose in 1:ℓ
        x1 = tensor1[dose][mpValid[dose]]
        x2 = tensor2[dose][mpValid[dose]]
        totalSum += sum((x1 - x2) .^ 2)
        totalNum += sum(mpValid[dose])
    end

    return(totalSum / totalNum)
end

function getTotalMAE(tensor1, tensor2, mpValid)
    ℓ = length(tensor1)
    totalSum = 0
    totalNum = 0
    
    for dose in 1:ℓ
        x1 = tensor1[dose][mpValid[dose]]
        x2 = tensor2[dose][mpValid[dose]]
        totalSum += sum(abs.(x1 - x2))
        totalNum += sum(mpValid[dose])
    end

    return(totalSum / totalNum)
end

function getMAE(tensor1, tensor2)
    [mean(abs.(tensor1[dose] - tensor2[dose])) for dose in 1:length(tensor1)]
end

function getMAE(tensor1, tensor2, mpValid)
    ℓ = length(tensor1)
    result = Array{Float64, 1}(undef, ℓ)
    
    for dose in 1:ℓ
        x1 = tensor1[dose][mpValid[dose]]
        x2 = tensor2[dose][mpValid[dose]]
        result[dose] = mean(abs.(x1 - x2))
    end

    return(result)
end

function getMean(tensor)
    ℓ = length(tensor)
    result = Array{Float64, 1}(undef, ℓ)
    
    for dose in 1:ℓ
        observed = .!isnan.(tensor[dose])
        result[dose] = mean(tensor[dose][observed])
    end

    return(result)
end

function getSqError(tensor1, Y, mpValid)
    ℓ = length(tensor1)
    #sqError = Vector{Float64}(ℓ)
    sqError = Array{Float64, 1}(undef, ℓ)
    
    for dose in 1:ℓ
        sqError[dose] = sum((tensor1[dose][mpValid[dose]] - Y[dose][mpValid[dose]]) .^ 2)
    end

    return(sqError)
end

function getSqErrorTest(tensor1, Z, test)
    ℓ = length(tensor1)
    sqError = Vector{Float64}(ℓ)
    
    for dose in 1:ℓ
        Zactual = Z[dose][test,:]
        known = .!isnan.(Zactual)
        sqError[dose] = sum((tensor1[dose][test,:] - Zactual)[known] .^ 2)
    end

    return(sqError)
end

function getAbsError(tensor1, Y, mpValid)
    ℓ = length(tensor1)
    absError = Vector{Float64}(ℓ)
    
    for dose in 1:ℓ
        absError[dose] = sum(abs.(tensor1[dose][mpValid[dose]] - Y[dose][mpValid[dose]]))
    end

    return(absError)
end

function getBaselineAbsError(baseline, Y, mpValid)
    ℓ = length(baseline)
    absError = Vector{Float64}(ℓ)
    
    for dose in 1:ℓ
        absError[dose] = sum(abs.(baseline[dose] - Y[dose][mpValid[dose]]))
    end

    return(absError)
end

function getBaselineSqError(baseline, Y, mpValid)
    ℓ = length(baseline)
    sqError = Array{Float64, 1}(undef, ℓ)
    
    for dose in 1:ℓ
        sqError[dose] = sum((baseline[dose] .- Y[dose][mpValid[dose]]) .^ 2)
    end

    return(sqError)
end

function getNumMissing(mpValid)
    return([sum(x) for x in mpValid])
end

function getR2(tensor1, Y, baseline, mpValid)
    ℓ = length(tensor1)
    result = Vector{Float64}(ℓ)
    
    for dose in 1:ℓ
        SSE = sum((tensor1[dose][mpValid[dose]] - Y[dose][mpValid[dose]]) .^ 2)
        SST = sum((baseline[dose] - Y[dose][mpValid[dose]]) .^ 2)

        result[dose] = 1 - SSE / SST
    end

    return(result)
end

function getColumnMean(tensorTrain)
    ℓ = length(tensorTrain)
    n,m = size(tensorTrain[1])
    result = deepcopy(tensorTrain)

    dose = 1
    j = 1
    for dose in 1:ℓ
        x = tensorTrain[dose]
        doseMean = mean(x[.!isnan.(x)])
        for j in 1:m
            missing = isnan.(x[:,j])
            observedValues = x[.!missing,j]
            if length(observedValues) > 0
                result[dose][missing,j] = mean(observedValues)
            else
                result[dose][missing,j] = doseMean
            end
        end
    end

    return(result)
end

function getColumnDrugMeans(tensorTrain)
    ℓ = length(tensorTrain)
    n,m = size(tensorTrain[1])
    result = [zeros(m) for _ in 1:ℓ]

    dose = 1
    j = 1
    for dose in 1:ℓ
        x = tensorTrain[dose]
        doseMean = mean(x[.!isnan.(x)])
        for j in 1:m
            missing = isnan.(x[:,j])
            observedValues = x[.!missing,j]
            if length(observedValues) > 0
                result[dose][j] = mean(observedValues)
            else
                result[dose][j] = doseMean
            end
        end
    end

    return(result)
end

function getTensorDrugMeans(tensorTrain)
    columnDrugMeans = getColumnDrugMeans(tensorTrain)
    n, m = size(tensorTrain[1])

    tensorDrugMeans = [repeat(c', n, 1) for c in columnDrugMeans]

    return(tensorDrugMeans)
end

function getTensorCellLineMeans(tensorTrain, tensorDrugMeans)
    ℓ = length(tensorTrain)
    tensorCellLineResiduals = Vector(ℓ)
    
    for dose in 1:ℓ
        tensorCellLineResiduals[dose] = tensorTrain[dose] - tensorDrugMeans[dose]
    end

    tensorCellLineResidualsTranspose = getTensorTranspose(tensorCellLineResiduals)
    tensorCellLineMeansTranspose = getTensorDrugMeans(tensorCellLineResidualsTranspose)
    tensorCellLineMeans = getTensorTranspose(tensorCellLineMeansTranspose)
    return(tensorCellLineMeans)
end

function fillZeroTensor(tensorTrain)
    ℓ = length(tensorTrain)
    tensorTrain0 = deepcopy(tensorTrain)

    for dose in 1:ℓ
        tensorTrain0[dose][isnan.(tensorTrain[dose])] .= 0
    end

    return(tensorTrain0)
end

function getBaselineTensor(tensorTrain)
    baselineTensor = deepcopy(tensorTrain)
    ℓ = length(tensorTrain)

    for dose in 1:ℓ
        observed = .!isnan.(tensorTrain[dose])
        baselineTensor[dose][.!observed] .= mean(tensorTrain[dose][observed])
    end

    return(baselineTensor)
end

function setTrainMP(tensor, test)
    tensorTrain = deepcopy(tensor)
    ℓ = length(tensor)

    for dose in 1:ℓ
        tensorTrain[dose][test,:] = NaN
    end

    return(tensorTrain)
end
