
function get_synapse!(blox_src; synapse_kwargs=(;), kwargs...)
    syn = get(kwargs, :synapse) do
        syn = default_synapse(blox_src; synapse_kwargs...)
        push!(synapse_set(blox_src), syn)
        syn
    end
    syn
end

# default fallbacks
function blox_wiring_rule!(g, blox; kwargs...)
    add_vertex!(g, blox)
end
function blox_wiring_rule!(g, blox_src, blox_dst; weight, conn_type=BasicConnection, kwargs...)
    add_edge!(g, blox_src, blox_dst; conn=conn_type(weight), weight, kwargs...)
end

struct BasicConnection <: ConnectionRule
    weight::Float64
end
Base.zero(::Type{BasicConnection}) = BasicConnection(0.0)


struct EventConnection{NT <: NamedTuple} <: ConnectionRule
    weight::Float64
    event_times::NT
end
EventConnection(w, event_times::NamedTuple) = EventConnection(convert(Float64, w), event_times) 
Base.zero(::Type{EventConnection{N}}) where {N} = EventConnection(0.0, (;))

GraphDynamics.has_discrete_events(::EventConnection) = true
GraphDynamics.has_discrete_events(::Type{EventConnection{NT}}) where {NT} = true
function GraphDynamics.discrete_event_condition((;event_times)::EventConnection, t)
    t ∈ event_times
end
GraphDynamics.event_times((;event_times)::EventConnection) = event_times

#==========================================================================================
Direct connections between Exci/Inhi neurons are implemented with an intermediate syntaptic block by default

i.e.
#   HHExci => HHExci

turns into

#   HHExci => Glu_AMPA_Synapse
#             Glu_AMPA_Synapse => HHExci

and

#   HHInhi => HHExci

turns into

#   HHInhi => GABA_A_Synapse
#             GABA_A_Synapse => HHExci
==========================================================================================#
function Graphs.add_edge!(g::Neurograph, blox_src::Union{HHExci, HHInhi}, blox_dst::Union{HHExci, HHInhi}; kwargs...)
    blox_wiring_rule!(g, blox_src, blox_dst; kwargs...)
end

function blox_wiring_rule!(g::Neurograph, blox_src::HHExci, blox_dst::Union{HHExci, HHInhi};
                           learning_rule=NoLearningRule(), sta = false, kwargs...)
    weight = get_weight(kwargs, blox_src.name, blox_dst.name)

    learning_rule = maybe_set_state_pre(learning_rule, namespaced_name(blox_src.name, "spikes_cumulative"))
    learning_rule = maybe_set_state_post(learning_rule, namespaced_name(blox_dst.name, "spikes_cumulative"))
    if sta
        #============================
        STA synapses require both the pre-synaptic voltage to calculate z, and the post synaptic voltage
        to calculate zₛₜₚ. Hence we make one connection from the presynaptic neuron to the synapse, and a
        ReverseConnection from the postsynaptic neuron to the synapse.
        #[n_pre::HHExci] -->G_asymp_pre--> [syn::Glu_AMPA_STA_Synapse] --->I_syn--> [n_post::HHExci]
        #                                    ↑                                       /
        #                                     \------------G_asymp_post<------------/
        # Because of this, we musn't ever re-use a pre-existing synapse!
        ============================#
        syn = Glu_AMPA_STA_Synapse(;name=Symbol("$(blox_src.name)_$(blox_dst.name)_STA_synapse"),
                                   E_syn=blox_src.E_syn, τ₂=blox_src.τ)
        add_edge!(g, blox_src, syn; kwargs..., weight=1)  # weight=1 marks this as a forward rule 
        add_edge!(g, blox_dst, syn; kwargs..., weight=2)# weight=2 marks this as a reverse rule
        add_edge!(g, syn, blox_dst; kwargs..., weight, conn=BasicConnection(weight), learning_rule)
    else
        # Generate a synapse (or fetch a pre-existing one)
        syn = get_synapse!(blox_src; kwargs...)
        conn = BasicConnection(weight)
        if !has_edge(g, blox_src, syn) # If we're re-using a synapse, don't re-add the connection
            add_edge!(g, blox_src, syn; kwargs..., weight, conn) # Note: connection between src and syn is not learnable!
        end
        add_edge!(g, syn, blox_dst; kwargs..., weight, conn, learning_rule)
    end
    nothing
end

function blox_wiring_rule!(g::Neurograph, blox_src::HHInhi, blox_dst::Union{HHExci, HHInhi}; kwargs...)
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse!(blox_src; kwargs...)
    # wire up the src to syn and syn to dst
    if !has_edge(g, blox_src, syn)
        # If we're re-using a synapse, don't re-add the connection
        blox_wiring_rule!(g, blox_src, syn; kwargs...)
    end
    blox_wiring_rule!(g, syn, blox_dst; kwargs...)
    nothing
end

"""
    (::BasicConnection)(src::Subsystem{<:Union{HHExci, HHInhi}}, dst::Subsystem{<:AbstractSynapse})

This connection simply forwards a presynaptic neuron's voltage to a synaptic block. The synaptic block is then
able to use that presyntaptic voltage in its connections with a postsynaptic neuron
"""
function (::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{<:Union{Glu_AMPA_Synapse, GABA_B_Synapse, Glu_NMDA_Synapse}})
    input = initialize_input(sys_dst)
    (; V, G_syn, V_shift, V_range) = sys_src
    @reset input.G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
end

function (::BasicConnection)(sys_src::Subsystem{HHInhi}, sys_dst::Subsystem{<:Union{GABA_A_Synapse, GABA_B_Synapse}})
    input = initialize_input(sys_dst)
    # @set input.V = sys_src.V
    (; V, G_syn, V_shift, V_range) = sys_src
    @reset input.G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
end

function (c::BasicConnection)(sys_src::Subsystem{<:Union{GABA_A_Synapse, GABA_B_Synapse, Glu_AMPA_Synapse}},
                              sys_dst::Subsystem{<:Union{HHExci, HHInhi}})
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_src.g * sys_src.G * (sys_dst.V - sys_src.E_syn)
end
function (c::BasicConnection)(sys_src::Subsystem{NGNMM_theta}, sys_dst::Subsystem{<:Union{HHExci, HHInhi}}, t)
    w = c.weight
    a = sys_src.aₑ
    b = sys_src.bₑ
    acc = initialize_input(sys_dst)
    @reset acc.I_asc = w * (1/(sys_src.Cₑ*π))*(1-a^2-b^2)/(1+2*a+a^2+b^2)
end

function (c::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{Glu_AMPA_STA_Synapse})
    input = initialize_input(sys_dst)
    (; V, G_syn, V_shift, V_range) = sys_src
    G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))
    if c.weight == 1
        @reset input.G_asymp_pre = G_asymp
    elseif c.weight == 2
        @reset input.G_asymp_post = G_asymp
    else
        error("Weight flag for connections leading to GLU_AMPA_STA_synapse must be either 1 (pesynaptic) or 2 (postsynaptic)")
    end
end

function (c::BasicConnection)(sys_src::Subsystem{Glu_AMPA_STA_Synapse}, sys_dst::Subsystem{<:Union{HHExci, HHInhi}})
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.I_syn = -w * sys_src.g * sys_src.Gₛₜₚ * sys_src.G * (sys_dst.V - sys_src.E_syn)
end

#---------------------------------------------------------------------
# L-FLIC

function blox_wiring_rule!(g, blox_src::L_FLICBlox, blox_dst::L_FLICBlox; kwargs...)
    # users can supply a :connection_matrix to the graph edge, where
    # connection_matrix[i, j] determines if neurons_src[i] is connected to neurons_src[j]
    namespaced_nameof(x) = x.name
    connection_matrix = get_connection_matrix(namespaced_nameof(blox_src), namespaced_nameof(blox_dst),
                                              length(blox_src.excis), length(blox_dst.excis); kwargs...)
    
    for (j, neuron_postsyn) in enumerate(blox_dst.excis)
        name_postsyn = neuron_postsyn.name
        for (i, neuron_presyn) in enumerate(blox_src.excis)
            if name_postsyn != neuron_presyn.name && connection_matrix[i, j]
                blox_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        end
    end
end

function blox_wiring_rule!(g, blox_src::HHInhi, blox_dst::L_FLICBlox; kwargs...)
    for neuron_postsyn in blox_dst.excis
        blox_wiring_rule!(g, blox_src, neuron_postsyn; kwargs...)
    end
end



#---------------------------------------------------------------------
# Cortical Blox, STN, Thalamus

function hypergeometric_connections!(g, neurons_src, neurons_dst, name_src, name_dst; kwargs...)
    density = get_density(kwargs, name_src, name_dst)
    N_connects = density * length(neurons_dst) * length(neurons_src)
    if length(neurons_dst) == 0
        @show name_dst
    end
    out_degree = Int(ceil(N_connects / length(neurons_src)))
    in_degree =  Int(ceil(N_connects / length(neurons_dst)))
    wt = get_weight(kwargs, name_src, name_dst)
    outgoing_connections = zeros(Int, length(neurons_src))
    for neuron_postsyn in neurons_dst
        rem = findall(x -> x < out_degree, outgoing_connections)
        idx = sample(rem, min(in_degree, length(rem)); replace=false)
        if length(wt) == 1
            for neuron_presyn in neurons_src[idx]
                blox_wiring_rule!(g, neuron_presyn, neuron_postsyn; kwargs...)
            end
        else
            for i in idx
                blox_wiring_rule!(g, neurons_src[i], neuron_postsyn; kwargs..., weight=wt[i])
            end
        end
        outgoing_connections[idx] .+= 1
    end
end

function blox_wiring_rule!(g, blox_src::Union{CorticalBlox,STN,Thalamus}, blox_dst::Union{CorticalBlox,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function blox_wiring_rule!(g, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{CorticalBlox,STN,Thalamus}; kwargs...)
    neurons_dst = get_exci_neurons(blox_dst)
    neurons_src = get_inhi_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function blox_wiring_rule!(g, blox_src::NGNMM_theta, blox_dst::CorticalBlox; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    blox_wiring_rule!(g, blox_src, blox_dst.n_ff_inh; kwargs...)
end


#---------------------------------------------------------------------
# GPi

function blox_wiring_rule!(g, blox_src::Union{Striatum, GPi, GPe}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    neurons_src = get_inhi_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function blox_wiring_rule!(g, blox_src::Union{CorticalBlox,STN,Thalamus}, blox_dst::Union{GPi, GPe}; kwargs...)
    neurons_dst = get_inhi_neurons(blox_dst)
    neurons_src = get_exci_neurons(blox_src)
    hypergeometric_connections!(g, neurons_src, neurons_dst, blox_src.name, blox_dst.name; kwargs...)
end

function blox_wiring_rule!(g, blox_src::HHExci, blox_dst::Union{Striatum, GPi}; kwargs...)
    for neuron_dst ∈ get_inhi_neurons(blox_dst)
        blox_wiring_rule!(g, blox_src, neuron_dst; kwargs...)
    end
end

#---------------------------------------------------------------------
# Striatum

function blox_wiring_rule!(g, cb::CorticalBlox, str::Striatum; kwargs...)
    neurons_dst = get_inhi_neurons(str)
    neurons_src = get_exci_neurons(cb)
    
    w = get_weight(kwargs, cb.name, str.name)

    dist = Uniform(0, 1)
    wt_ar = 2*w*rand(dist, length(neurons_src)) # generate a uniform distribution of weight with average value w
    kwargs = (kwargs..., weight=wt_ar)
    if haskey(kwargs, :learning_rule)
        lr = kwargs.learning_rule
        matr = str.matrisome
        lr = maybe_set_state_post(lr, namespaced_name(matr.name, "H_learning"))
        kwargs = (kwargs..., learning_rule=lr)
    end
    hypergeometric_connections!(g, neurons_src, neurons_dst, cb.name, str.name; kwargs...)

    algebraic_parts = (str.matrisome, str.striosome)
    for (i, neuron_presyn) ∈ enumerate(neurons_src)
        kwargs = (kwargs...,weight=wt_ar[i])
        for part ∈ algebraic_parts
            blox_wiring_rule!(g, neuron_presyn, part; kwargs...)
        end
    end
end

function blox_wiring_rule!(g, sys_src::Striatum, sys_dst::Striatum; kwargs...)
    t_event = get(kwargs, :t_event) do
        error("No `t_event` provided for the connection between $(sys_src.name) and $(sys_dst.name)")
    end
    blox_wiring_rule!(g, sys_src.matrisome, sys_dst.matrisome; t_event=t_event +   √(eps(t_event)), kwargs...)
    blox_wiring_rule!(g, sys_src.matrisome, sys_dst.striosome; t_event=t_event + 2*√(eps(t_event)), kwargs...)
    for inhib ∈ sys_dst.inhibs
        blox_wiring_rule!(g, sys_src.matrisome, inhib; t_event=t_event+2*√(eps(t_event)), kwargs...)
    end
    nothing
end

function blox_wiring_rule!(g, sys_src::TAN, sys_dst::Striatum; kwargs...)
    blox_wiring_rule!(g, sys_src, sys_dst.matrisome; kwargs...)
end



#---------------------------------------------------------------------
# Discrete blox

function blox_wiring_rule!(g, sys_src::Striatum, sys_dst::Union{TAN, SNc}; kwargs...)
    blox_wiring_rule!(g, sys_src.striosome, sys_dst; kwargs...)
end

function blox_wiring_rule!(g, sys_src::Striosome, sys_dst::Union{TAN,SNc};
                           weight, kwargs...)
    conn = BasicConnection(weight)
    if haskey(kwargs, :learning_rule)
        @info "" kwargs.learning_rule
    end
    add_edge!(g, sys_src, sys_dst; weight, conn, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{Striosome}, sys_dst::Subsystem{<:Union{TAN, SNc}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.H * sys_src.jcn_t_block
end

function blox_wiring_rule!(g, sys_src::HHExci, sys_dst::Union{Matrisome, Striosome};
                           weight, learning_rule=NoLearningRule(), kwargs...)
    
    conn = BasicConnection(weight)
    learning_rule = maybe_set_state_pre( learning_rule, Symbol(sys_src.name, :₊spikes_cumulative))
    learning_rule = maybe_set_state_post(learning_rule, Symbol(sys_dst.name, :₊H_learning))
    add_edge!(g, sys_src, sys_dst; weight, conn, learning_rule, kwargs...)
end
function (c::BasicConnection)(sys_src::Subsystem{HHExci}, sys_dst::Subsystem{<:Union{Matrisome, Striosome}}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_src.spikes_window
end



function blox_wiring_rule!(g, sys_src::Matrisome, sys_dst::Union{Matrisome, Striosome, HHInhi}; weight=1.0, t_event, kwargs...)
    conn = EventConnection(weight, (;t_init=0.1, t_event))
    add_edge!(g, sys_src, sys_dst; conn, kwargs...)
end

function (c::EventConnection)(src::Subsystem{Matrisome}, dst::Subsystem{<:Union{Matrisome, Striosome, HHInhi}}, t)
    initialize_input(dst)
end


function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Matrisome})

    (;params_partitioned, state_types_val, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = to_vec_o_states(u.x, state_types_val)

    (;t_event) = ec.event_times
            
    params_dst = vparams_dst[]
    if haskey(ec.event_times, :t_init) && t == ec.event_times.t_init
        @reset params_dst.H = 1
    end
    if t == t_event
        @reset params_dst.H = m_src.ρ > m_dst.ρ ? 0 : 1
    end
    vparams_dst[] = params_dst
    nothing
end

using Base: isstored

function find_competitor_matrisome(integrator, m::Subsystem{Matrisome}, j)
    (;params_partitioned, state_types_val, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = to_vec_o_states(u.x, state_types_val)
    i = findfirst(v -> eltype(v) <: SubsystemStates{Matrisome}, states_partitioned)
    l = findfirst(eachindex(states_partitioned[i])) do l
        found = false
        if l != j
            for nc ∈ 1:length(connection_matrices)
                M = connection_matrices[nc].data[i][i]
                if !(M isa NotConnected)
                    found = !iszero(M[l, j]) && !iszero(M[j, l])
                end
            end
        end
        found
    end
    if !isnothing(l)
        Subsystem(states_partitioned[i][l], params_partitioned[i][l])
    else
        @warn "No competitor found for" m.name
    end
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{HHInhi})
    t = integrator.t
    params_src = vparams_src[]
    params_dst = vparams_dst[]
    (;t_init, t_event) = ec.event_times
    if t == t_init
        # @info "M-I Init"
        vparams_dst[] = @reset params_dst.I_bg = 0.0
    elseif t == t_event
        # @info "M-I Event"
        m_comp = find_competitor_matrisome(integrator, m_src, only(vparams_src.indices))
        if !isnothing(m_comp)
            vparams_dst[] = @reset params_dst.I_bg = m_src.ρ > m_comp.ρ ? -2.0 : 0.0
        end
    else
        error("Invalid event time, this shouldn't be possible")
    end
    nothing
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             m_src::Subsystem{Matrisome},
                                             m_dst::Subsystem{Striosome})
    t = integrator.t
    (;t_init, t_event) = ec.event_times
    params_dst = vparams_dst[]
    if t == t_init
        @reset params_dst.H = 1
    else
        m_comp = find_competitor_matrisome(integrator, m_src, only(vparams_src.indices))
        if !isnothing(m_comp)
            @reset params_dst.H = m_src.ρ > m_comp.ρ ? 0 : 1
        end
    end
    vparams_dst[] = params_dst
    nothing
end


function blox_wiring_rule!(g, sys_src::TAN, sys_dst::Matrisome; weight=1.0, t_event, kwargs...)
    conn = EventConnection(weight, (; t_event))
    add_edge!(g, sys_src, sys_dst; conn, kwargs...)
end
function (c::EventConnection)(sys_src::Subsystem{TAN}, sys_dst::Subsystem{Matrisome}, t)
    w = c.weight
    input = initialize_input(sys_dst)
    @reset input.jcn = w * sys_dst.TAN_spikes
end

function GraphDynamics.apply_discrete_event!(integrator,
                                             _, vparams_src,
                                             _, vparams_dst,
                                             ec::EventConnection,
                                             sys_src::Subsystem{TAN},
                                             sys_dst::Subsystem{Matrisome})

    params_dst = vparams_dst[]
    w = ec.weight
    vparams_dst[] = @reset params_dst.TAN_spikes = w * rand(Poisson(sys_src.R))
    nothing
end


#--------------------
# ImageStimulus
function blox_wiring_rule!(g, stim::ImageStimulus, neuron::Union{HHInhi, HHExci}; current_pixel, weight, kwargs...)
    add_edge!(g, stim, neuron; conn=StimConnection(weight, current_pixel), weight, kwargs...)
end

struct StimConnection <: ConnectionRule
    weight::Float64
    pixel_index::Int
end

function (c::StimConnection)(src::Subsystem{ImageStimulus}, dst::Subsystem{<:Union{HHInhi, HHExci}}, t)
    w = c.weight
    input = initialize_input(dst)
    @reset input.I_in = w * src.current_image[c.pixel_index]
end

function blox_wiring_rule!(g, src::ImageStimulus, dst::CorticalBlox; kwargs...)
    current_pixel = 1
    for n_dst ∈ vertices(dst.graph)
        if n_dst isa HHExci
            blox_wiring_rule!(g, src, n_dst; current_pixel, kwargs...)
            current_pixel = mod(current_pixel, src.N_pixels) + 1
        end
    end
end
