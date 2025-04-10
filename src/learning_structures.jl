struct NoLearningRule <: AbstractLearningRule end

@kwdef struct HebbianPlasticity <:AbstractLearningRule
    K::Float64
    W_lim::Float64
    state_pre::Union{Nothing, Symbol}  = nothing
    state_post::Union{Nothing, Symbol} = nothing
    t_pre::Union{Nothing, Float64}     = nothing
    t_post::Union{Nothing, Float64}    = nothing
end


function (hp::HebbianPlasticity)(val_pre, val_post, w, feedback)
    Δw = hp.K * val_pre * val_post * (hp.W_lim - w) * feedback
    return Δw
end

function weight_gradient(hp::HebbianPlasticity, sol, w, feedback)
    val_pre = only(sol(hp.t_pre; idxs = [hp.state_pre]))
    val_post = only(sol(hp.t_post; idxs = [hp.state_post]))

    return hp(val_pre, val_post, w, feedback)
end

get_eval_times(l::HebbianPlasticity) = (l.t_pre, l.t_post)
get_eval_states(l::HebbianPlasticity) = (l.state_pre, l.state_post)

function maybe_set_state_pre(lr::AbstractLearningRule, state)
    if isnothing(lr.state_pre)
        @set lr.state_pre = state
    else
        lr
    end
end

function maybe_set_state_post(lr::AbstractLearningRule, state)
    if isnothing(lr.state_post)
        @set lr.state_post = state
    else
        lr
    end
end
maybe_set_state_pre(lr::NoLearningRule, state) = lr
maybe_set_state_post(lr::NoLearningRule, state) = lr

@kwdef struct HebbianModulationPlasticity <: AbstractLearningRule
    K::Float64
    decay::Float64
    α::Float64
    θₘ::Float64
    state_pre::Union{Nothing, Symbol}  = nothing
    state_post::Union{Nothing, Symbol} = nothing
    t_pre::Union{Nothing, Float64}     = nothing
    t_post::Union{Nothing, Float64}    = nothing
    t_mod::Union{Nothing, Float64}     = nothing
    modulator                          = nothing
end

dlogistic(x) = logistic(x) * (1 - logistic(x)) 

function (hmp::HebbianModulationPlasticity)(val_pre, val_post, val_modulator, w, feedback)
    DA = hmp.modulator(val_modulator)
    DA_baseline = hmp.modulator.κ_DA * hmp.modulator.N_time_blocks
    ϵ = feedback - (hmp.modulator.κ_DA - DA)
    
    # Δw = hmp.K * val_post * val_pre * DA * (DA - DA_baseline) * dlogistic(DA) - hmp.decay * w
    Δw = max((hmp.K * val_post * val_pre * ϵ * (ϵ + hmp.θₘ) * dlogistic(hmp.α * (ϵ + hmp.θₘ)) - hmp.decay * w), -w)

    return Δw
end


function weight_gradient(hmp::HebbianModulationPlasticity, sol, w, feedback)
    state_mod = get_modulator_state(hmp.modulator)
    val_pre = sol(hmp.t_pre; idxs = hmp.state_pre)
    val_post = sol(hmp.t_post; idxs = hmp.state_post)
    val_mod = sol(hmp.t_mod; idxs = state_mod)

    return hmp(val_pre, val_post, val_mod, w, feedback)
end

get_eval_times(l::HebbianModulationPlasticity) = (l.t_pre, l.t_post, l.t_mod)
get_eval_states(l::HebbianModulationPlasticity) = (l.state_pre, l.state_post, get_modulator_state(l.modulator))

mutable struct ClassificationEnvironment{S} <: AbstractEnvironment
    const name::Symbol
    const source::S
    const category::Vector{Int}
    const N_trials::Int
    const t_trial::Float64
    current_trial::Int
    
    function ClassificationEnvironment(stim::ImageStimulus; name)
        N_trials = stim.N_stimuli

        ClassificationEnvironment(stim, N_trials; name)
    end

    function ClassificationEnvironment(stim::ImageStimulus, N_trials; name)
        t_trial = stim.t_stimulus + stim.t_pause

        new{typeof(stim)}(Symbol(name), stim, stim.category, N_trials, t_trial, 1)
    end
end

(env::ClassificationEnvironment)(action) = action == env.category[env.current_trial]
increment_trial!(env::AbstractEnvironment) = env.current_trial = mod(env.current_trial, env.N_trials) + 1
reset!(env::AbstractEnvironment) = env.current_trial = 1

abstract type AbstractActionSelection <: AbstractBlox end

struct GreedyPolicy <: AbstractActionSelection
    name::Symbol
    competitor_states::Vector{Symbol}
    competitor_params::Vector{Symbol}
    t_decision::Float64

    function GreedyPolicy(; name, t_decision,  competitor_states=nothing, competitor_params=nothing)
        sts = isnothing(competitor_states) ? Symbol[] : competitor_states
        ps = isnothing(competitor_states) ? Symbol[] : competitor_params
        new(name, sts, ps, t_decision)
    end
end

function GraphDynamics.system_wiring_rule!(g::GraphSystem, ::AbstractActionSelection; kwargs...)
    #@info "Skipping the wiring of an ActionSelection"
    nothing
end
function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox_src::AbstractBlox, ::AbstractActionSelection; kwargs...)
    # @info "Skipping the wiring of an ActionSelection"
    nothing
end

function (p::GreedyPolicy)(sol::AbstractSciMLSolution)
    comp_vals = map(p.competitor_states) do sym
        sol(p.t_decision; idxs=sym)
    end
    return argmax(comp_vals)
end

function connect_action_selection!(as::AbstractActionSelection, str1::Striatum, str2::Striatum)
    connect_action_selection!(as, str1.matrisome, str2.matrisome)
end

function connect_action_selection!(as::AbstractActionSelection, matr1::Matrisome, matr2::Matrisome)
    @assert length(as.competitor_states) == 0
    push!(as.competitor_states, Symbol(matr1.name, :₊ρ_snapshot), Symbol(matr2.name, :₊ρ_snapshot)) #HACK : accessing values of rho at a specific time after the simulation
    as
end

get_eval_times(gp::GreedyPolicy) = (gp.t_decision,)
get_eval_states(gp::GreedyPolicy) = gp.competitor_states

struct Agent{S,P,A,LR,CM}
    system::S
    problem::P
    action_selection::A
    learning_rules::LR
    connection_matrices::CM
end
function Agent(g::GraphSystem; name, t_block=missing, u0=[], p=[], kwargs...)
    if !ismissing(t_block)
        global_events=[PeriodicCallback(t_block_event(:t_block_early), t_block - √(eps(float(t_block)))),
                       PeriodicCallback(t_block_event(:t_block_late), t_block  +2*√(eps(float(t_block))))]
    else
        global_events=[]
    end
    
    # sys = graphsystem_from_graph(g; global_events)
    sys_par = PartitionedGraphSystem(g)
    prob = ODEProblem(g, u0, (0.,1.), p; global_events, kwargs...)
    policy = action_selection_from_graph(g)
    learning_rules = make_connection_matrices(sys_par.flat_graph,
                                              conn_key=:learning_rule,
                                              pred=(x) -> !(x isa NoLearningRule)).connection_matrices
    conn = prob.p.connection_matrices
    Agent(g, prob, policy, learning_rules, conn)
end

function action_selection_from_graph(g::GraphSystem)
    sels = [blox for blox in nodes(g) if blox isa AbstractActionSelection]
    if isempty(sels)
        @warn "No action selection provided"
    elseif length(sels) == 1
        sel = only(sels)
        srcs = []
        for (;src, dst) ∈ connections(g)
            if dst == sel
                push!(srcs, src)
            end
        end
        if length(srcs) != 2
            error("Two blocks need to connect to the action selection $(sel.name) block")
        end
        connect_action_selection!(sel, srcs[1], srcs[2])
    else
        error("Multiple action selection blocks are detected. Only one must be used in an experiment.")
    end
end
