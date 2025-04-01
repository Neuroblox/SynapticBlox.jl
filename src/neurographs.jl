
struct NeurographEdge <: AbstractEdge{Any}
    src
    dst
    data::NamedTuple
end
NeurographEdge(src, dst; kwargs...) = NeurographEdge(src, dst, NamedTuple(kwargs))

struct Neurograph <: AbstractMetaGraph{Any}
    data::OrderedDict{Any, OrderedDict{Any, NeurographEdge}}
end

Neurograph() = Neurograph(OrderedDict{Any, OrderedDict{Any, NeurographEdge}}())


Graphs.edges(g::Neurograph) = (Iterators.flatten ∘ Iterators.map)(g.data) do (_, d)
    Iterators.map(((_, e),) -> e, d)
end
Graphs.vertices(g::Neurograph) = keys(g.data)
function Graphs.add_vertex!(g::Neurograph, blox)
    get!(g.data, blox) do
        OrderedDict{Any, NeurographEdge}()
    end
end

function Graphs.add_edge!(g::Neurograph, src, dst; kwargs...)
    d_src = add_vertex!(g, src)
    d_dst = add_vertex!(g, dst)

    if haskey(d_src, dst)
        error("Attempted to add an edge between two vertices that already have an edge. Not allowed.")
    end
    d_src[dst] = NeurographEdge(src, dst, NamedTuple(kwargs))
end
Graphs.add_edge!(g::Neurograph, src, dst, d::AbstractDict) = add_edge!(g, src, dst; d...)
Graphs.has_edge(g::Neurograph, src, dst) = haskey(g.data, src) && haskey(g.data[src], dst)
function MetaGraphs.props(g::Neurograph, src, dst)
    g.data[src][dst]
end

function Base.merge!(g1::Neurograph, g2::Neurograph)
    for x ∈ vertices(g2)
        add_vertex!(g1, x)
    end
    for (;src, dst, data) ∈ edges(g2)
        add_edge!(g1, src, dst; data...)
    end
    g1
end

function Graphs.add_edge!(g::Neurograph, (src, dst)::Pair; kwargs...)
    add_edge!(g, src, dst; kwargs...)
end
