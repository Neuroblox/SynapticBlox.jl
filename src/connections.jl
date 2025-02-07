function Neuroblox.Connector(blox_src::Union{HHExci, HHInhi},
                             blox_dst::HHExci;
                             kwargs...)
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse!(blox_src, Glu_AMPA_Synapse; kwargs...)
    # wire up the src to syn and syn to dst
    c1 = Connector(blox_src, syn; kwargs...)
    c2 = Connector(syn, blox_dst; kwargs...)
    merge!(c1, c2)
    # Store the synapse in the connector
    push!(c1.connection_blox, syn)
    c1
end
function Neuroblox.Connector(blox_src::Union{HHExci, HHInhi},
                             blox_dst::HHInhi;
                             kwargs...)
    # Generate a synapse (or fetch a pre-existing one)
    syn = get_synapse!(blox_src, GABA_A_Synapse; kwargs...)
    # wire up the src to syn and syn to dst
    c1 = Connector(blox_src, syn; kwargs...)
    c2 = Connector(syn, blox_dst; kwargs...)
    merge!(c1, c2)
    # Store the synapse in the connector
    push!(c1.connection_blox, syn)
    c1
end

function Neuroblox.Connector(blox_src::Union{HHExci, HHInhi},
                             blox_dest::AbstractSynapse; kwargs...)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)

    # All we need to do here is send the presynaptic neuron's voltage into the synapse
    eq = sys_dest.V ~ sys_src.V 
    Connector(nameof(sys_src), nameof(sys_dest), equation=eq)
end

function Neuroblox.Connector(blox_src::Union{GABA_A_Synapse, GABA_B_Synapse},
                             blox_dest::HHInhi; kwargs...)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    g = get_g(blox_src)
    
    eq = sys_dest.I_syn ~ -w * g * sys_src.G * (sys_dest.V - sys_src.E_syn)
    Connector(nameof(sys_src), nameof(sys_dest), equation=eq, weight=w)
end

function Neuroblox.Connector(blox_src::Union{Glu_AMPA_Synapse, Glu_NMDA_Synapse},
                             blox_dest::HHExci; kwargs...)
    sys_src = get_namespaced_sys(blox_src)
    sys_dest = get_namespaced_sys(blox_dest)
    w = generate_weight_param(blox_src, blox_dest; kwargs...)

    g = get_g(blox_src)
    
    eq = sys_dest.I_syn ~ -w * g * sys_src.G * (sys_dest.V - sys_src.E_syn)
    Connector(nameof(sys_src), nameof(sys_dest), equation=eq, weight=w)
end

# function Graphs.add_edge!(g::MetaDiGraph,
#                           (blox_src, blox_dst)::Pair{<:Union{HHExci, HHInhi},
#                                                      <:Union{HHExci}}; kwargs...)
#     syn = get_synapse!(blox_src, Glu_AMPA_Synapse; kwargs...)
#     add_edge!(g, blox_src => syn; kwargs...)
#     add_edge!(g, syn => blox_dst; kwargs...)
# end

# function Graphs.add_edge!(g::MetaDiGraph,
#                           (blox_src, blox_dst)::Pair{<:Union{HHExci, HHInhi},
#                                                      <:Union{HHInhi}}; kwargs...)
#     syn = get_synapse!(blox_src, GABA_A_Synapse; kwargs...)
#     add_edge!(g, blox_src => syn; kwargs...)
#     add_edge!(g, syn => blox_dst; kwargs...)
# end

function get_synapse!(blox_src, default_synapse_type, default_synapse_kwargs=(;); kwargs...)
    syn = get(kwargs, :synapse, default_synapse_type)
    if syn isa Type
        name_src = namespaced_nameof(blox_src)
        syn_kwargs = merge((;name = Symbol("$(syn)_$(name_src)"),),
                           get(kwargs, :synapse_kwargs, default_synapse_kwargs))
        
        syn = get!(synapse_dict(blox_src), (syn, syn_kwargs)) do
            syn(; syn_kwargs...)
        end
    end
    syn
end
