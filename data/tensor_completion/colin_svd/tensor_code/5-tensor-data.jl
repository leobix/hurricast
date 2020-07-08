using DataFrames
using CSV

function convertToMatrix(A::DataFrame)
    [A[nm]=convert(Array{Union{Float64, Missing}},A[nm]) for nm in names(A)]
    #[A[nm]=convert(Array{Union{Float64, String, Missing}},A[nm]) for nm in names(A)]
    #[A[A[nm].=="NA",nm]=NaN for nm in names(A)]
    [A[DataFrames.ismissing.(A[nm]),nm]=NaN for nm in names(A)]
    return(convert(Matrix{Float64},A))
end

function getDataGDSC()
    gdsc = Vector{Any}(undef, 12)
    #gdsc = Vector{Any}(12)

for dose in 1:12
        println(dose)
        #gdsc[dose] = readtable("tensor_data/14-gdscDrugResponseGenomicDose$dose.csv")
        gdsc[dose] = CSV.read("tensor_data/14-gdscDrugResponseGenomicDose$dose.csv"; missingstring="NA")
    end


    for dose in 1:12
        gdsc[dose] = convertToMatrix(gdsc[dose][:,2:end])
    end

    return(gdsc)
end



#A = gdsc[1]
#DataFrames.writetable("julia0.7_test_gdsc[1]", A)
#CSV.write("julia0.7_test_gdsc[1]", A)

