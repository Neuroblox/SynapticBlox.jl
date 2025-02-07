abstract type AbstractSynapse <: AbstractBlox end

for S ∈ ("GABA_A", "GABA_B", "Glu_AMPA", "Glu_NMDA")
    g = Symbol("g_", S)
    T = Symbol(S, "_Synapse")
    @eval begin
        struct $T <: AbstractSynapse
            system
            namespace
            function $T(
                ;name,
                namespace=nothing,
                E_syn=0.0,
                G_syn=3,
                τ=5,
                $g=1.0)
                sts = @variables begin
                    V(t)
                    [input=true]
                    G(t)=0.0
                    [output=true]
                    z(t)=0.0
                end
                ps = @parameters begin
                    E_syn=E_syn
                    G_syn=G_syn
                    V_shift=10
                    V_range=35
                    τ₁=0.1
                    τ₂=τ
                    $g=$g
                end
                G_asymp(v,G_syn) = (G_syn/(1 + exp(-4.394*((v-V_shift)/V_range))))
                eqs = [
                    D(G)~(-1/τ₂)*G + z,
			        D(z)~(-1/τ₁)*z + G_asymp(V,G_syn)
                ]
                sys = ODESystem(eqs, t, sts, ps; name=Symbol(name))
                new(sys, namespace)
            end
        end
        get_g(s::$T) = getproperty(s.system, $(QuoteNode(g)))
    end
end
