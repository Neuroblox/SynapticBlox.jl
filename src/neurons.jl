@kwdef struct HHExci <: AbstractExciNeuronBlox
    name::Symbol
    G_Na::Float64 =  52
    G_K::Float64  =  20
    G_L::Float64  =  0.1
    E_Na::Float64 =  55
    E_K::Float64  = -90
    E_L::Float64  = -60
    I_bg::Float64 =  0
    E_syn::Float64   = 0.0
    G_syn::Float64   = 3.0
    V_shift::Float64 = 10
    V_range::Float64 = 35
    τ::Float64       =  5
    spk_const::Float64 = 1.127
    default_synapses::Set=Set()
end

function to_subsystem(h::HHExci)
    # Default state initial values
    states = SubsystemStates{HHExci}(
        V = -65.0,
        n = 0.32,
        m = 0.05,
        h = 0.59,
        spikes_cumulative = 0.0,
        spikes_window = 0.0
    )
    # Parameter values
    (; name, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg,
     E_syn, G_syn, V_shift, V_range, spk_const, τ) = h
    params = SubsystemParams{HHExci}(;name, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg,
                                     E_syn, G_syn, V_shift, V_range, τ, spk_const)
    # Total subsystem
    Subsystem(states, params)
end

# HHExci Neurons can take 3 different types of input, I_syn, I_in and I_asc, each of which can be
# incremented by different connection types
GraphDynamics.initialize_input(s::Subsystem{HHExci}) = (;I_syn = 0.0, I_in=0.0, I_asc=0.0)

# Compute the differential of the subsystem's states for a given input and time
function GraphDynamics.subsystem_differential(sys::Subsystem{HHExci}, input, t)
    (; V, n, m, h) = sys                                  # Unpack states
    (; G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg) = sys        # Unpack params
    (; E_syn, G_syn, V_shift, V_range, spk_const) = sys   # Unpack more params
    (; I_syn, I_in, I_asc) = input                        # Unpack inputs

    # Define some helper functions
    αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
	βₙ(v) = 0.125*exp(-(v+44)/80)
	αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
	βₘ(v) = 4*exp(-(v+55)/18)
	αₕ(v) = 0.07*exp(-(v+44)/20)
	βₕ(v) = 1/(1+exp(-(v+14)/10))
	ϕ = 5

    G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))

    # Output the differentials of each states:
    SubsystemStates{HHExci}(
        #=d/dt=#V =-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in,
		#=d/dt=#n = ϕ*(αₙ(V)*(1-n)-βₙ(V)*n),
		#=d/dt=#m = ϕ*(αₘ(V)*(1-m)-βₘ(V)*m),
		#=d/dt=#h = ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
        #=d/dt=#spikes_cumulative = spk_const*G_asymp,
        #=d/dt=#spikes_window     = spk_const*G_asymp,
    )
end
default_synapse(n::HHExci; E_syn=n.E_syn, τ₁=0.1, τ₂=n.τ, kwargs...) =
    Glu_AMPA_Synapse(;name=Symbol("$(n.name)_synapse"), E_syn, τ₁, τ₂, kwargs...)

has_t_block_event(::Type{HHExci}) = true
is_t_block_event_time(::Type{HHExci}, key, t) = key == :t_block_late
t_block_event_requires_inputs(::Type{HHExci}) = false
function apply_t_block_event!(vstates, _, s::Subsystem{HHExci}, _, _)
    vstates[:spikes_window] = 0.0
end

@kwdef struct HHInhi <: AbstractInhiNeuronBlox
    name::Symbol
    G_Na::Float64 =  52 
	G_K::Float64  =  20
	G_L::Float64  =  0.1 
	E_Na::Float64 =  55 
	E_K::Float64  = -90 
	E_L::Float64  = -60
    I_bg::Float64 =  0
    E_syn::Float64   =-70.0
    G_syn::Float64   = 11.5
    V_shift::Float64 = 0
    V_range::Float64 = 35
    τ::Float64       = 70
    default_synapses::Set=Set()
end

function to_subsystem(h::HHInhi)
    states = SubsystemStates{HHInhi}((
        V = -65.0,
        n = 0.32,
        m = 0.05,
        h = 0.59,
    ))
    (; name, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg,
     E_syn, G_syn, V_shift, V_range, τ) = h
    params = SubsystemParams{HHInhi}(; name, G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg, 
                                     E_syn, G_syn, V_shift, V_range, τ)
    Subsystem(states, params)
end
GraphDynamics.initialize_input(s::Subsystem{HHInhi}) = (;I_syn = 0.0, I_in=0.0, I_asc=0.0)
function GraphDynamics.subsystem_differential(sys::Subsystem{HHInhi}, jcn, t)
    (; V, n, m, h) = sys                            # Unpack states
    (; G_Na, G_K, G_L, E_Na, E_K, E_L, I_bg) = sys  # Unpack params
    (;I_syn, I_in, I_asc) = jcn                     # Unpack inputs

    # Define some helper functions
    αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
	βₙ(v) = 0.125*exp(-(v+48)/80)
    αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
	βₘ(v) = 4*exp(-(v+58)/18)
    αₕ(v) = 0.07*exp(-(v+51)/20)
	βₕ(v) = 1/(1+exp(-(v+21)/10))
	ϕ = 5
    
    # Output the differentials of each states:
    SubsystemStates{HHInhi}(
        #=d/dt=#V =-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg+I_syn+I_asc+I_in,
		#=d/dt=#n = ϕ*(αₙ(V)*(1-n)-βₙ(V)*n),
		#=d/dt=#m = ϕ*(αₘ(V)*(1-m)-βₘ(V)*m),
		#=d/dt=#h = ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
    )
end
default_synapse(n::HHInhi; E_syn=n.E_syn, τ₁=0.1, τ₂=n.τ, kwargs...) =
    GABA_A_Synapse(;name=Symbol("$(n.name)_synapse"), E_syn, τ₁, τ₂, kwargs...)

synapse_set(n::Union{HHExci, HHInhi}) = getfield(n, :default_synapses)

