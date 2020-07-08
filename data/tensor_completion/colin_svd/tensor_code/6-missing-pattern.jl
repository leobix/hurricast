using StatsBase
using Random

# Get missing pattern
function getMP(tensor)
    [isnan.(x) for x in tensor]
end

function getMissingEnds(v)
    observed = findall(.!isnan.(v))
    result = falses(length(v))

    if length(observed) != 0
        i = observed[1]
        result[1:(i-1)] .= true

        i = observed[end]
        result[(i+1):end] .= true
    end

    return(result)
end

function getToughMP(tensor)
    ℓ = length(tensor)
    n, p = size(tensor[1])

    toughMP = [falses(n,p) for _ in 1:ℓ]

    for i in 1:n
        for d in 1:p
            v = [tensor[dose][i,d] for dose in 1:ℓ]
            vectorMP = getMissingEnds(v)

            for dose in 1:ℓ
                toughMP[dose][i,d] = vectorMP[dose]
            end
        end
    end

    return(toughMP)
end

function generateMatrixMP(mpMatrix, percent; seed = 123)
    n, p = size(mpMatrix)
    mpNew = falses(n, p)

    srand(seed)
    for d in 1:p
        observed = find(.!mpMatrix[:,d])
        numMissing = Int(floor(percent * length(observed)))
        missing = sample(observed, numMissing, replace = false)

        mpNew[missing,d] = true
    end

    return(mpNew)
end

function generateMP(mp, pct, OPP, seed)
    if isnan(OPP)
        result = generateValidMP(mp, pct, seed)
    else
        result = generateClinicalMP(mp, pct, OPP, seed)
    end

    return(result)
end

function generateClinicalMP(mp, pct, OPP, seed)
    ℓ = length(mp)
    n, m = size(mp[1])

    mpClinical = [falses(n,m) for _ in 1:ℓ]
    
    srand(seed)
    test = sample(1:n, Int(floor((pct * n))), replace = false)
    
    for i in test
        for dose in 1:ℓ
            mpClinical[dose][i,:] = .!mp[dose][i,:]
        end
        observed = find(vcat([.!mp[dose][i,:] for dose in 1:ℓ]...))
        srand(seed + i)
        observedIndiv = sample(observed, OPP, replace = false)
        for jk in observedIndiv
            k = 1 + Int(floor((jk - 1) / m))
            j = jk - m * (k - 1)
            mpClinical[k][i,j] = false
        end
    end

    return(mpClinical)
end

function generateValidMP(mp, pct, seed)
    ℓ = length(mp)
    n, p = size(mp[1])

    mpValid = [falses(n,p) for _ in 1:ℓ]
    
    Random.seed!(seed)
    for dose in 1:ℓ
        for d in 1:p
            observed = findall(.!mp[dose][:,d])
            numMissing = Int(floor(pct * length(observed)))
            missing = sample(observed, numMissing, replace = false)

            mpValid[dose][missing,d] .= true
        end
    end

    return(mpValid)
end

function setMP(tensor, mpValid)
    tensorNew = deepcopy(tensor)

    for dose in 1:length(tensor)
        tensorNew[dose][mpValid[dose]] .= NaN
    end

    return(tensorNew)
end

function setZeroMP(tensor, mpValid)
    tensorNew = deepcopy(tensor)

    for dose in 1:length(tensor)
        tensorNew[dose][mpValid[dose]] = 0
    end

    return(tensorNew)
end

function getTrainTest(mpTest)
    ℓ = length(mpTest)
    n,m = size(mpTest[1])
    test = sort(unique(vcat([find(any(mpTest[dose],2)) for dose in 1:ℓ]...)))
    train = setdiff(1:n, test)

    return(train, test)
end

function getReducedTrain(tensorTrain, train)
    ℓ = length(tensorTrain)
    tensorReducedTrain = Vector(ℓ)

    for dose in 1:ℓ
        tensorReducedTrain[dose] = tensorTrain[dose][train,:]
    end

    return(tensorReducedTrain)
end

function getFullTrain(tensorReducedTrain, train, test)
    ℓ = length(tensorReducedTrain)
    n1, m = size(tensorReducedTrain[1])
    n = length(train) + length(test)
    tensorFullTrain = [fill(NaN, n, m) for _ in 1:ℓ]

    for dose in 1:ℓ
        tensorFullTrain[dose][train,:] = tensorReducedTrain[dose]
    end

    return(tensorFullTrain)
end

