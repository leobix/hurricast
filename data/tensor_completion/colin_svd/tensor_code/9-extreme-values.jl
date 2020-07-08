function getExtremeValues(tensor)
    ℓ = length(tensor)
    minValue = Array{Float64, 1}(undef, ℓ)
    maxValue = Array{Float64, 1}(undef, ℓ)

    for dose in 1:ℓ
        observed = .!isnan.(tensor[dose])
        minValue[dose] = minimum(tensor[dose][observed])
        maxValue[dose] = maximum(tensor[dose][observed])
    end

    return(minValue, maxValue)
end

function boundExtremeValues(tensor, minValue, maxValue)
    ℓ = length(tensor)
    newTensor = deepcopy(tensor)

    for dose in 1:ℓ
        extremeHigh = tensor[dose] .> maxValue[dose]
        extremeLow = tensor[dose] .< minValue[dose]

        newTensor[dose][extremeHigh] .= maxValue[dose]
        newTensor[dose][extremeLow] .= minValue[dose]
    end

    return(newTensor)
end
