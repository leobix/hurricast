function fillVectorEnds(v)
    result = deepcopy(v)
    observed = findall(.!isnan.(result))

    if length(observed) != 0
        i = observed[1]
        result[1:(i-1)] .= result[i]

        i = observed[end]
        result[(i+1):end] .= result[i]
    end

    return(result)
end

function fillVectorMiddle(v)
    observed = findall(.!isnan.(v))
    missing = findall(isnan.(v))
    result = deepcopy(v)

    if length(missing) != 0 && length(observed) != 0
        for i in missing
            x1 = findlast(.!isnan.(v[1:(i-1)]))
            x2 = i + findfirst(.!isnan.(v[(i+1):end]))
            result[i] = (i - x1) * (v[x2] - v[x1]) / (x2 - x1) + v[x1]
        end
    end

    return(result)
end

function fillTensor(tensor)
    tensorNew = deepcopy(tensor)

    ℓ = length(tensorNew)
    n, p = size(tensorNew[1])

    for i in 1:n
        for d in 1:p
            v = fillVectorEnds([tensorNew[dose][i,d] for dose in 1:ℓ])
            v = fillVectorMiddle(v)
            for dose in 1:ℓ
                tensorNew[dose][i,d] = v[dose]
            end
        end
    end

    return(tensorNew)
end

function fillMean(tensor)
    tensorNew = deepcopy(tensor)

    ℓ = length(tensorNew)
    n, p = size(tensorNew[1])

    for dose in 1:ℓ
        for d in 1:p
            v = tensorNew[dose][:,d]
            missing = findall(isnan.(v))
            observed = findall(.!isnan.(v))
            tensorNew[dose][missing,d] .= mean(v[observed])
        end

        if (sum(isnan.(tensorNew[dose])) > 0)
            for i in 1:n
                v = tensorNew[dose][i,:]
                missing = find(isnan.(v))
                observed = find(.!isnan.(v))
                tensorNew[dose][i,missing] = mean(v[observed])
            end
        end
    end

    return(tensorNew)
end

function linearInterp(tensor)
    tensorNew = deepcopy(tensor)
    tensorNew = fillTensor(tensorNew)
    tensorNew = fillMean(tensorNew)

    return(tensorNew)
end
