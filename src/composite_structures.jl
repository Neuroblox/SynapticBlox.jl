function GraphDynamics.system_wiring_rule!(g::GraphSystem, blox::CompositeBlox; kwargs...)
    merge!(g, blox.graph)
end

"""
    L_FLICBlox

L-FLIC (local-feedback lateral inhibition circuits) 
"""
struct L_FLICBlox <: CompositeBlox
    name::Symbol
    inhi::HHInhi
    excis::Vector{HHExci}
    graph::GraphSystem
    function L_FLICBlox(;name,
                        N_exci = 5,
                        E_syn_exci=0.0,
                        E_syn_inhib=-70,
                        G_syn_exci=3.0,
                        G_syn_inhib=3.0,
                        I_bg=zeros(N_exci),
                        # phase=0.0,
                        τ_exci=5,
                        τ_inhib=70)

        inhi = HHInhi(name = namespaced_name(name, :inhi), E_syn = E_syn_inhib, G_syn = G_syn_inhib, τ = τ_inhib)
        
        excis = map(1:N_exci) do i
            excii = HHExci(
                name = namespaced_name(name, "exci$i"),
                I_bg = (I_bg isa AbstractArray) ? I_bg[i] : I_bg*rand(), # behave differently if I_bg is array
                E_syn = E_syn_exci,
                G_syn = G_syn_exci,
                τ = τ_exci
                # phase = phase
            )
        end
        g = GraphSystem()
        for excii ∈ excis
            system_wiring_rule!(g, inhi, excii; weight=1.0)
            system_wiring_rule!(g, excii, inhi; weight=1.0)
        end
        new(name, inhi, excis, g)
    end
end


struct CorticalBlox <: CompositeBlox
    name::Symbol
    l_flics
    n_ff_inh
    graph::GraphSystem
    function CorticalBlox(;name,
                          N_l_flic=20,
                          N_exci=5,
                          E_syn_exci=0.0,
                          E_syn_inhib=-70,
                          G_syn_exci=3.0,
                          G_syn_inhib=4.0,
                          G_syn_ff_inhib=3.5,
                          I_bg_ar=0,
                          τ_exci=5,
                          τ_inhib=70,
                          kwargs...)


        
        n_ff_inh = HHInhi(name = namespaced_name(name, :ff_inh), E_syn=E_syn_inhib, G_syn=G_syn_ff_inhib, τ=τ_inhib)
        
        l_flics = map(1:N_l_flic) do i
            if I_bg_ar isa AbstractArray
                I_bg = I_bg_ar[i]
            else
                I_bg = I_bg_ar
            end
            L_FLICBlox(;name = namespaced_name(name, "l_flic$i"),
                       N_exci,
                       E_syn_exci,
                       E_syn_inhib,
                       G_syn_exci,
                       G_syn_inhib,
                       I_bg = I_bg,
                       τ_exci,
                       τ_inhib)
        end
        
        g = GraphSystem()
        system_wiring_rule!(g, n_ff_inh)
        for i ∈ 1:N_l_flic
            system_wiring_rule!(g, l_flics[i])
            for j ∈ 1:N_l_flic
                if j != i
                    # users can supply a matrix of connection matrices.
                    # connection_matrices[i,j][k, l] determines if neuron k from l-flic i is connected to
                    # neuron l from l-flic j.
                    if haskey(kwargs, :connection_matrices)
                        kwargs_ij = merge(kwargs, (; connection_matrix = kwargs[:connection_matrices][i, j]))
                    else
                        kwargs_ij = kwargs
                    end

                    # connect l_flics[i] to l_flics[j]
                    system_wiring_rule!(g, l_flics[i], l_flics[j]; kwargs_ij...)
                end
            end
            # connect the inhibitory neuron to the i-th l_flic
            system_wiring_rule!(g, n_ff_inh, l_flics[i]; weight=1.0)
        end
        new(name, l_flics, n_ff_inh, g)
    end
end


struct Striatum <: CompositeBlox
    name::Symbol
    inhibs::Vector{HHInhi}
    matrisome::Matrisome
    striosome::Striosome
    graph::GraphSystem
    function Striatum(; name,
                      N_inhib = 25,
                      E_syn_inhib=-70,
                      G_syn_inhib=1.2,
                      I_bg=zeros(N_inhib),
                      τ_inhib=70
                      )

        inhibs = map(1:N_inhib) do i
            HHInhi(
                name = namespaced_name(name, "inh$i"),
                E_syn = E_syn_inhib, 
                G_syn = G_syn_inhib, 
                τ = τ_inhib,
                I_bg = I_bg[i],
            ) 
        end
        (E_syn = E_syn_inhib,  G_syn = G_syn_inhib,  τ₂ = τ_inhib)

        matrisome = Matrisome(; name=namespaced_name(name, :matrisome))
        striosome = Striosome(; name=namespaced_name(name, :striosome))
        
        g = GraphSystem()

        for n ∈ inhibs
            system_wiring_rule!(g, n)
        end
        system_wiring_rule!(g, matrisome)
        system_wiring_rule!(g, striosome)

        new(name, inhibs, matrisome, striosome, g)
    end
end


struct GPi <: CompositeBlox
    name::Symbol
    inhibs::Vector{HHInhi}
    graph
    function GPi(; name,
                 N_inhib = 25,
                 E_syn_inhib =-70,
                 G_syn_inhib=8,
                 I_bg=4*ones(N_inhib),
                 τ_inhib=70)

        graph = GraphSystem()
        inhibs = map(1:N_inhib) do i
            inhib = HHInhi(; name=namespaced_name(name, Symbol(:inh, i)),
                           E_syn = E_syn_inhib,
                           G_syn = G_syn_inhib,
                           τ = τ_inhib,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, inhibs, graph)
    end
end

struct GPe <: CompositeBlox
    name::Symbol
    inhibs::Vector{HHInhi}
    graph
    function GPe(;name,
                 N_inhib=15,
                 E_syn_inhib=-70,
                 G_syn_inhib=3,
                 I_bg=2*ones(N_inhib),
                 τ_inhib=70)
        graph = GraphSystem()
        inhibs = map(1:N_inhib) do i
            inhib = HHInhi(; name=namespaced_name(name, Symbol(:inh, i)),
                           E_syn = E_syn_inhib,
                           G_syn = G_syn_inhib,
                           τ = τ_inhib,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, inhib)
            inhib
        end
        new(name, inhibs, graph)
    end
end

struct Thalamus <: CompositeBlox
    name::Symbol
    excis::Vector{HHExci}
    graph
    function Thalamus(;name,
                      N_exci = 25,
                      E_syn_exci=0,
                      G_syn_exci=3,
                      I_bg=3*ones(N_exci),
                      τ_exci=5)
        graph = GraphSystem()
        excis = map(1:N_exci) do i
            exci = HHExci(; name=namespaced_name(name, Symbol(:exci, i)),
                           E_syn = E_syn_exci,
                           G_syn = G_syn_exci,
                           τ = τ_exci,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, exci)
            exci
        end
        new(name, excis, graph)
    end
end

struct STN <: CompositeBlox
    name::Symbol
    excis::Vector{HHExci}
    graph
    function STN(;name,
                 N_exci = 25,
                 E_syn_exci=0,
                 G_syn_exci=3,
                 I_bg=3*ones(N_exci),
                 τ_exci=5)
        graph = GraphSystem()
        excis = map(1:N_exci) do i
            exci = HHExci(; name=namespaced_name(name, Symbol(:exci, i)),
                           E_syn = E_syn_exci,
                           G_syn = G_syn_exci,
                           τ = τ_exci,
                           I_bg = I_bg[i])
            system_wiring_rule!(graph, exci)
            exci
        end
        new(name, excis, graph)
    end
end
