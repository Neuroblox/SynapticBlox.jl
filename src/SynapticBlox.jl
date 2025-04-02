module SynapticBlox


using Base: @kwdef
using Base.Iterators: map as imap

using Graphs:
    Graphs,
    AbstractEdge,
    add_edge!,
    edges,
    vertices,
    add_vertex!,
    add_edge!,
    src,
    dst,
    has_edge

using MetaGraphs:
    MetaGraphs,
    MetaDiGraph,
    AbstractMetaGraph,
    props,
    set_prop!

using OrderedCollections:
    OrderedCollections,
    OrderedDict

using GraphDynamics:
    GraphDynamics,
    NotConnected,
    ConnectionRule,
    ConnectionMatrices,
    ConnectionMatrix,
    SDEGraphSystem,
    ODEGraphSystem,
    initialize_input,
    subsystem_differential,
    Subsystem,
    SubsystemParams,
    SubsystemStates,
    isstochastic,
    event_times,
    get_tag,
    get_states,
    get_params,
    calculate_inputs,
    partitioned,
    maybe_sparse_enumerate_col

using Accessors:
    Accessors,
    @set,
    @reset

using SparseArrays:
    SparseArrays,
    sparse

using CSV:
    CSV

using Distributions:
    Distributions,
    Bernoulli,
    Uniform,
    Poisson

using Random:
    Random

using StatsBase:
    StatsBase,
    sample

using MacroTools:
    MacroTools,
    @capture

using LogExpFunctions:
    LogExpFunctions,
    logistic

using SciMLBase:
    SciMLBase,
    AbstractSciMLSolution,
    ODEProblem,
    remake,
    solve,
    getp

using DiffEqCallbacks:
    DiffEqCallbacks,
    PeriodicCallback

abstract type AbstractBlox end
abstract type AbstractExciNeuronBlox <: AbstractBlox end
abstract type AbstractInhiNeuronBlox <: AbstractBlox end
abstract type NeuralMassBlox <: AbstractBlox end
abstract type CompositeBlox <: AbstractBlox end
abstract type StimulusBlox  <: AbstractBlox end

abstract type AbstractEnvironment end
abstract type AbstractLearningRule end

include("utils.jl")
include("neurographs.jl")
include("neurons.jl")
include("neural_mass.jl")
include("synapses.jl")
include("discrete.jl")
include("composite_structures.jl")
include("sources.jl")
include("connections.jl")
include("learning_structures.jl")
include("graph_dynamics_interop.jl")


export
    Neurograph,
    add_vertex!,
    add_edge!,
    graphsystem_from_graph

export
    HHExci,
    HHInhi,
    GABA_A_Synapse,
    GABA_B_Synapse,
    Glu_AMPA_Synapse,
    Glu_NMDA_Synapse,
    NGNMM_theta,
    L_FLICBlox,
    CorticalBlox,
    Striatum,
    GPi,
    GPe,
    Thalamus,
    STN,
    Matrisome,
    Striosome,
    TAN,
    SNc,
    ImageStimulus

export
    HebbianPlasticity,
    HebbianModulationPlasticity,
    CSEventHandler,
    GreedyPolicy,
    Agent,
    ClassificationEnvironment,
    run_experiment!

export
    @named


end # module SynapticBlox
