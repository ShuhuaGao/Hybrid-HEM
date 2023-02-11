# Behavior cloning where the policy is approximated by a DNN

"""
    normalize(X::AbstractMatrix) -> (Matrix, UnitRangeTransform)

Min-max normalize the input matrix `X`. The normalized version and the associated `UnitRangeTransform`
object are returned.
"""
function normalize(X::AbstractMatrix)
    urt = StatsBase.fit(UnitRangeTransform, X; dims = 2)
    return StatsBase.transform(urt, X), urt
end

function shuffle_split_obs(X, y; at = 0.9)
    return splitobs(shuffleobs((X, y)); at)
end

struct Agent{T<:Flux.Chain}
    urt::UnitRangeTransform # min-max scaling 
    model::T
end

function (agent::Agent)(X)
    Xn = StatsBase.transform(agent.urt, X)
    return agent.model(Xn)
end


build_AD_net(tf = relu) = Chain(
    Dense(3, 200, tf),
    Dropout(0.1),
    Dense(200, 100, tf),
    Dropout(0.1),
    Dense(100, 50, tf),
    Dropout(0.1),
    Dense(50, 1)
)


"""
    train_AD_agent(X::AbstractMatrix, y::AbstractVector; tf=relu, kwargs...)->Tuple{Agent, Matrix}

Train a decision agent for an AD load using state-action data (`X`, `y`). The model in the returned
agent is a DNN regressor.

A tuple `(agent, loss_history)` is returned. The two columns of `loss_history` refer to the training
loss and validation loss curves respectively.
"""
function train_AD_agent(X::AbstractMatrix, y::AbstractVecOrMat; tf = relu, kwargs...)::Tuple{Agent,Matrix}
    Xn, urt = normalize(X)
    model = build_AD_net(tf)
    loss = Flux.mse

    hist = trainNN!(model, Xn, y, loss; kwargs...)
    return (Agent(urt, model), hist)
end

"""
    eval_loss(loader, model, l; device=cpu) -> Float64

Compute the total loss of data in `loader` using the DNN `model` assessed by the `loss` function.
The aggregator of `loss` should be `mean`. The input data in `loader` should be normalized already.
"""
function eval_loss(loader, model, loss; device = Flux.cpu)::Float64
    n = 0
    tl = 0.0
    model = model |> device
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        tl += loss(ŷ, reshape(y, size(ŷ)...)) * length(y)
        n += length(y)
    end
    return tl / n
end

"""
    eval_loss(loader, agent::Agent, loss; device = Flux.cpu) -> Float64

Compute the total loss of data in `loader` using the DNN `agent` assessed by the `loss` function.
The aggregator of `loss` should be `mean`. The input data in `loader` will get first normalized by
the transform in `agent`.
"""
function eval_loss(loader, agent::Agent, loss; device = Flux.cpu)::Float64
    urt = agent.urt
    norm_loader = [(StatsBase.transform(urt, X), y) for (X, y) in loader]
    return eval_loss(norm_loader, agent.model, loss; device)
end

"""

Train a neural network `model`. The weights of the `model` are updated in place. The loss history of
the training set and the validation set during training is returned. At last, the `model` has parameters
corresponding to the minimum validation loss.

Note that the input `X` should have already been normalized properly.
`loss(ŷ, y)`
"""
function trainNN!(model, X, y, loss; val_ratio = 0.1,
    η = 1e-2, batchsize = 256, epochs = 500, device = Flux.gpu,
    slow_learning_threshold = 10, early_stopping_threshold = 20,
    report_freq = 5, verbose::Bool=false)

    (X_train, Y_train), (X_val, Y_val) = shuffle_split_obs(X, y; at = 1 - val_ratio)
    loader_train = Flux.Data.DataLoader((X_train, Y_train); batchsize, shuffle = true)
    loader_val = Flux.Data.DataLoader((X_val, Y_val); batchsize, shuffle = false)

    m = model |> device
    ps = Flux.params(m)
    opt = Flux.Optimise.ADAM(η)

    best_loss_val = Inf
    best_ps = nothing
    last_improvement = 0
    loss_history = zeros(epochs, 2)

    loss_train = eval_loss(loader_train, m, loss; device)
    loss_val = eval_loss(loader_val, m, loss; device)
    # loss_history[e, :] .= (loss_train, loss_val)
    report_freq > 0 && @info 0 loss_train loss_val

    for e = 1:epochs
        for (x, y) in loader_train  # mini_batch
            x, y = device(x), device(y)
            gs = gradient(ps) do
                ŷ = m(x)
                loss(ŷ, reshape(y, size(ŷ)...))
            end
            Flux.update!(opt, ps, gs)
        end

        # calculate all at once since the dataset is small
        loss_train = eval_loss(loader_train, m, loss; device)
        loss_val = eval_loss(loader_val, m, loss; device)
        loss_history[e, :] .= (loss_train, loss_val)
        if report_freq > 0 && e % report_freq == 0
            @info e loss_train loss_val
        end

        if loss_val < best_loss_val
            last_improvement = e
            best_loss_val = loss_val
            # @save model_file model = cpu(m)
            best_ps = deepcopy(m |> cpu |> params)
        end
        # learning rate schedule
        if e - last_improvement >= slow_learning_threshold && η > 1e-5
            η /= 1.5
            verbose && @info "No improvement in $(slow_learning_threshold) epochs. Drop learning rate to" η
            last_improvement = e
        end
        if e - last_improvement >= early_stopping_threshold
            verbose && @info "No improvement in $(early_stopping_threshold) epochs. Early stopped." best_loss_val
            loss_history = loss_history[1:e, :]
            break
        end
    end

    # write the optimized parameters into `model`
    @info "The minimum validation loss is $best_loss_val"
    Flux.loadparams!(model, best_ps)
    return loss_history
end

"""
    build_SU_net(tf=relu) -> Chain

State: ``[t, ρ, P^{PV}]``. Action: ``z^{su} \\in {0, 1}``
"""
build_SU_net(in; tf = relu) = Chain(
    Dense(in, 200, tf),
    Dropout(0.1),
    Dense(200, 100, tf),
    Dropout(0.3),
    Dense(100, 50, tf),
    Dropout(0.5),
    Dense(50, 1, sigmoid)
)

build_SU_net2(in; tf = relu) = Chain(
    Dense(in, 100, tf),
    Dense(100, 50, tf),
    # Dense(100, 50, tf),
    Dense(50, 1, sigmoid)
)


"""
    weighted_binarycrossentropy(ŷ, y; w1=1.0)

Add weight to Flux `binarycrossentropy` loss. The weight of class 0 is fixed to 1.0.
"""
function weighted_binarycrossentropy(ŷ, y; w1 = 1.0)
    # based on the source of Flux.binarycrossentropy
    ϵ = Flux.epseltype(ŷ)
    mean(@.(-w1 * Flux.Losses.xlogy(y, ŷ + ϵ) - Flux.Losses.xlogy(1 - y, 1 - ŷ + ϵ)))
end

function train_SU_agent(X::AbstractMatrix, y::AbstractVector; tf = relu, w1 = NaN, kwargs...)::Tuple{Agent,Matrix}
    Xn, urt = normalize(X)

    if isnothing(w1)
        loss = (ŷ, y) -> Flux.binary_focal_loss(ŷ, y; γ = 3.5)
        @show "gamma"
    else
        if isnan(w1)
            # imbalanced -> reweight. Fix w0 = 1.
            cm = countmap(y)
            w1 = cm[0] / cm[1]
        end
        loss = (ŷ, y) -> weighted_binarycrossentropy(ŷ, y; w1)
    end
    model = build_SU_net2(size(X)[1]; tf)
    hist = trainNN!(model, Xn, y, loss; kwargs...)
    return (Agent(urt, model), hist)
end

function eval_accuracy(loader, agent::Agent; device = cpu)
    n_acc = n_all = 0
    model = agent.model |> device
    for (x, y) in loader
        x, y = StatsBase.transform(agent.urt, x) |> device, y |> device
        ŷ = model(x) .> 0.5
        n_acc += sum(y .== reshape(ŷ, size(y)...))
        n_all += length(y)
        @show n_acc n_all
    end
    return n_acc / n_all
end