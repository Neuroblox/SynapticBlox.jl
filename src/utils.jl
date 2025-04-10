namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(::Nothing, name) = Symbol(name)

macro named(ex)
    if @capture(ex, name_ = f_(args__; kwargs__))
        :($name = $f($(args...); $(kwargs...), name = $(QuoteNode(name))))
    elseif @capture(ex, name_ = f_(args__))
        :($name = $f($(args...); name = $(QuoteNode(name))))
    elseif @capture(ex, name_ = f_(;kwargs__))
        :($name = $f(; $(kwargs)..., name = $(QuoteNode(name))))
    else
        error("Malformed expression to @named, must be of the form `name = f(args...; kwargs...)`")
    end |> esc
end

function get_connection_matrix(name_src, name_dst, N_out, N_in; kwargs...)
    sz = (N_out, N_in)
    connection_matrix = get(kwargs, :connection_matrix) do
        density = get(kwargs, :density) do 
            error("Connection density from $name_src to $name_dst is not specified.")
        end
        dist = Bernoulli(density)
        rng = get(kwargs, :rng, Random.default_rng())
        rand(rng, dist, sz...)
    end
    if size(connection_matrix) != sz
        error(ArgumentError("The supplied connection matrix between $(name_src) and $(name_dst) is an "
                            * "incorrect size. Got $(size(connection_matrix)), whereas $(name_out) has "
                            * "$N_out excitatory neurons, and $name_in has $N_in excitatory neurons."))
    end
    if eltype(connection_matrix) != Bool
        error(ArgumentError("The supplied connection matrix between $(name_src) and $(name_dst) must "
                            * "be an array of Bool, got $(eltype(connection_matrix)) instead."))
    end
    connection_matrix
end

get_exci_neurons(blox_src::CompositeBlox) = [blox for blox ∈ nodes(blox_src.graph) if blox isa HHExci]
get_inhi_neurons(blox_src::CompositeBlox) = [blox for blox ∈ nodes(blox_src.graph) if blox isa HHInhi]

function get_density(kwargs, name_src, name_dst)
    density = get(kwargs, :density) do
        error("No connection density specified between $name_src and $name_dst")
    end
end
function get_weight(kwargs, name_src, name_dst)
    get(kwargs, :weight) do
        error("No connection weight specified between $name_src and $name_dst")
    end
end
