include("tensor_code/5-tensor-data.jl")
include("tensor_code/6-missing-pattern.jl")
include("tensor_code/7-warm-start.jl")
include("tensor_code/8-performance-metrics.jl")
include("tensor_code/9-extreme-values.jl")
include("tensor_code/11-slice-learning.jl")
include("tensor_code/43-biases.jl")
include("tensor_code/47-one-sided-gradient-descent.jl")
include("tensor_code/48-tensor-one-sided.jl")

#### Real Data Example ####
tensor = getDataGDSC()
X = Matrix(CSV.read("tensor_data/37-XnormalizedOncogenesGDSC.csv"));
seed = 1
pct = 0.8
mp = getMP(tensor)
mpTest = generateValidMP(mp, pct, seed)
tensorTrain = setMP(tensor, mpTest)
baseline = getBaselineTensor(tensorTrain)
baselineSqError = getSqError(baseline, tensor, mpTest)

# Slice Learning Algorithm
# Arguments:
#  - tensorTrain: tensor with some missing values
#  - tensor: ground-truth tensor (optional),
#            required to display test R2 each iteration
#  - mpTest: missing values in the test set (optional),
#            required to display test R2 each iteration
#  - rankList: vector of slice ranks to cross-validate over
#  - T: number of iterations
#  - imputeMiddle: If false, indicates that we use linear
#                  interpolation to impute values in the
#                  middle slices of the tensor whenever possible.
# Returns:
#  - tensorImputed: tensor with all values filled in
#  - r: slice rank selected via cross-validation
rankList = [20,40,60,80]
tensorImputed, r = sliceLearning(tensorTrain;
    tensor = tensor, mpTest = mpTest,
    rankList = rankList, T = 20,
    imputeMiddle = false)
sqError = getSqError(tensorImputed, tensor, mpTest)
mean(1 .- sqError ./ baselineSqError)

# Tensor One-Sided
# Arguments:
#  - tensorTrain: tensor with some missing values
#  - X: matrix of side information for the rows
#  - tensor: ground-truth tensor (optional),
#            required to display test R2 each iteration
#  - mpTest: missing values in the test set (optional),
#            required to display test R2 each iteration
#  - rankList: vector of slice ranks to cross-validate over
#              (Typically, we use r chosen by Slice Learning)
#  - γList: vector of γ values to cross-validate over
#  - tuneBias: If true, adds a constant term for the regression
# Returns:
#  - tensorOneSided: tensor with all values filled in
#  - γ: regularization parameter selected via cross-validation
#  - r: slice rank selected via cross-validation
rankList = [r]
γList = [10.0^(-i) for i in 1:5][1]
tensorOneSided, γ, r  = fitTensorOneSided(tensorTrain, X;
        rankList = rankList,
        γList = γList,
        tensor = tensor, mpTest = mpTest,
        tuneBias = true)
sqError = getSqError(tensorOneSided, tensor, mpTest)
mean(1 .- sqError ./ baselineSqError)





