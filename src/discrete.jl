abstract type AbstractDiscrete <: AbstractBlox end
abstract type AbstractModulator <: AbstractDiscrete end


has_t_block_event(::Type{<:AbstractDiscrete}) = true
is_t_block_event_time(::Type{<:AbstractDiscrete}, key, t) = key == :t_block_early
t_block_event_requires_inputs(::Type{<:AbstractDiscrete}) = true
function apply_t_block_event!(_, vparams, s::Subsystem{<:AbstractDiscrete}, (;jcn), _)
    params = get_params(s)
    vparams[] = @reset params.jcn_t_block = jcn
end



#-------------------------
# Matrisome

@kwdef struct Matrisome <: AbstractDiscrete
    name::Symbol
    H::Int=1
    TAN_spikes::Float64=0
    jcn_t_block::Float64=0.0
    jcn_snapshot::Float64=0.0
    H_snapshot::Int=0
    t_event::Float64=180
end

function GraphDynamics.to_subsystem(m::Matrisome)
    # Default state initial values
    states = SubsystemStates{Matrisome}()
    # Parameter values
    (; name, H, TAN_spikes, jcn_t_block, jcn_snapshot, H_snapshot, t_event) = m
    params = SubsystemParams{Matrisome}(; name, H, TAN_spikes, jcn_t_block, jcn_snapshot, H_snapshot,  t_event)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{Matrisome}) = (; jcn=0.0)

GraphDynamics.subsystem_differential_requires_inputs(::Type{Matrisome}) = false
function GraphDynamics.apply_subsystem_differential!(_, m::Subsystem{Matrisome}, _, _)
    nothing
end
function GraphDynamics.computed_properties(m::Subsystem{Matrisome})
    H_learning((;H)) = H
    ρ_snapshot((;H_snapshot, jcn_snapshot)) = H_snapshot * jcn_snapshot
    ρ((;H, jcn_t_block)) = H * jcn_t_block
    (;H_learning, ρ_snapshot, ρ)
end

GraphDynamics.event_times(m::Subsystem{Matrisome}) = m.t_event + √(eps(m.t_event))
GraphDynamics.has_discrete_events(::Type{Matrisome}) = true
GraphDynamics.discrete_events_require_inputs(::Type{Matrisome}) = false
function GraphDynamics.discrete_event_condition(m::Subsystem{Matrisome}, t, _)
    t == m.t_event + √(eps(m.t_event))
end
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{Matrisome}, _)
    # recording the values of jcn_t_block and H at the event time in the parameters jcn_ and H_
    params = get_params(s)
    @reset params.H_snapshot = s.H
    @reset params.jcn_snapshot = s.jcn_t_block 
    vparams[] = params
    nothing
end

#-------------------------
# Striosome

@kwdef struct Striosome <: AbstractDiscrete
    name::Symbol
    H::Int=1
    jcn_t_block::Float64=0.0
end
function GraphDynamics.to_subsystem(s::Striosome)
    # Default state initial values
    states = SubsystemStates{Striosome}()
    # Parameter values
    (; name, H, jcn_t_block) = s
    params = SubsystemParams{Striosome}(; name, H, jcn_t_block)
    # Total subsystem
    Subsystem(states, params)
end

GraphDynamics.initialize_input(::Subsystem{Striosome}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{Striosome}) = false
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{Striosome}, (;jcn), _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{Striosome})
    H_learning((;H)) = H
    ρ((;H, jcn_t_block)) = H * jcn_t_block
    (;H_learning, ρ)
end

#-------------------------
# TAN

@kwdef struct TAN <: AbstractDiscrete
    name::Symbol
    κ::Float64=100
    λ::Float64=1
    jcn_t_block::Float64=0.0
end
function GraphDynamics.to_subsystem(s::TAN)
    # Default state initial values
    states = SubsystemStates{TAN}()
    # Parameter values
    (; name, κ, λ, jcn_t_block) = s
    params = SubsystemParams{TAN}(; name, κ, λ, jcn_t_block)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{TAN}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{TAN}) = false
function GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{TAN}, _, _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{TAN})
    R((;κ, λ, jcn_t_block)) = min(κ, κ/(λ*jcn_t_block + sqrt(eps())))
    (;R)
end

#-------------------------
# SNc

@kwdef struct SNc <: AbstractDiscrete
    name::Symbol
    N_time_blocks::Int=5
    κ_DA::Float64 = 1
    λ_DA::Float64 = 0.33
    κ::Float64 = κ_DA
    λ::Float64 = λ_DA
    jcn_t_block::Float64 = 0.0
    jcn_snapshot::Float64 = 0.0
    t_event::Float64 = 90
end
function GraphDynamics.to_subsystem(s::SNc)
    # Default state initial values
    states = SubsystemStates{SNc}()
    # Parameter values
    (; name, κ_DA, λ_DA, κ, λ, jcn_t_block, jcn_snapshot, t_event) = s
    params = SubsystemParams{SNc}(; name, κ_DA, λ_DA, κ, λ, jcn_t_block, jcn_snapshot, t_event)
    # Total subsystem
    Subsystem(states, params)
end
GraphDynamics.initialize_input(::Subsystem{SNc}) = (; jcn=0.0)
GraphDynamics.subsystem_differential_requires_inputs(::Type{SNc}) = false
function GraphDynamics.apply_subsystem_differential!(_, ::Subsystem{SNc}, _, _)
    nothing
end
function GraphDynamics.computed_properties(::Subsystem{SNc})
    R_snapshot((;κ, λ, jcn_snapshot)) = min(κ, κ/(λ*jcn_snapshot + sqrt(eps())))
    R((;κ, λ, jcn_t_block)) = min(κ, κ/(λ*jcn_t_block + sqrt(eps())))
    (;R_snapshot, R)
end

function get_modulator_state(s::Subsystem{SNc})
    Symbol(s.name, :₊R_snapshot)
end
function get_modulator_state(s::SNc)
    Symbol(s.name, :₊R_snapshot)
end
(b::SNc)(R_DA) = R_DA #b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward
(b::Subsystem{SNc})(R_DA) = R_DA #b.N_time_blocks * b.κ_DA + R_DA - b.κ_DA + feedback * b.DA_reward

GraphDynamics.has_discrete_events(::Type{SNc}) = true
#GraphDynamics.discrete_events_require_inputs(::Type{SNc}) = true
function GraphDynamics.discrete_event_condition((;t_event,)::Subsystem{SNc}, t, _)
    t == t_event
end
GraphDynamics.event_times((;t_event)::Subsystem{SNc}) = t_event
function GraphDynamics.apply_discrete_event!(integrator, _, vparams, s::Subsystem{SNc}, _)
    # recording the values of jcn_t_block at the event time in the parameters jcn_snapshot
    params = get_params(s)
    vparams[] = @set params.jcn_snapshot = params.jcn_t_block
    nothing
end
