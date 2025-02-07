module SynapticBlox

using Neuroblox:
    Neuroblox,
    Connector,
    AbstractBlox,
    CompositeBlox,
    AbstractNeuronBlox,
    AbstractExciNeuronBlox,
    AbstractInhNeuronBlox,
    get_namespaced_sys,
    generate_weight_param,    
    namespaced_nameof

using Neuroblox.GraphDynamicsInterop:
    define_basic_connection, define_neuron

using ModelingToolkit:
    ModelingToolkit,
    @variables,
    @parameters,
    ODESystem

using ModelingToolkit: t_nounits as t, D_nounits as D

using Graphs:
    Graphs,
    add_edge!

using MetaGraphs:
    MetaGraphs,
    MetaDiGraph

export HHExci, HHInhi, GABA_A_Synapse, GABA_B_Synapse, Glu_AMPA_Synapse, Glu_NMDA_Synapse

include("neurons.jl")
include("synapses.jl")
include("connections.jl")
include("graph_dynamics_interop.jl")

end # module SynapticBlox
