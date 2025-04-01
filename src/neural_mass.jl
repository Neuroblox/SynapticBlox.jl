function to_nt(blox; exclude=())
    fnames = Tuple((name for name ∈ propertynames(blox) if name ∉ exclude))
    NamedTuple{fnames}(Tuple(getproperty(blox, name) for name ∈ fnames))
end

@kwdef struct NGNMM_theta <: NeuralMassBlox
    name
    Cₑ=30.0
    Cᵢ=30.0
    Δₑ=0.5
    Δᵢ=0.5
    η_0ₑ=10.0
    η_0ᵢ=0.0
    v_synₑₑ=10.0
    v_synₑᵢ=-10.0
    v_synᵢₑ=10.0
    v_synᵢᵢ=-10.0
    alpha_invₑₑ=10.0
    alpha_invₑᵢ=0.8
    alpha_invᵢₑ=10.0
    alpha_invᵢᵢ=0.8
    kₑₑ=0
    kₑᵢ=0.5
    kᵢₑ=0.65
    kᵢᵢ=0
end

function to_subsystem(nm::NGNMM_theta)
    # Default state initial values
    states = SubsystemStates{NGNMM_theta}(
        aₑ=-0.6,
        bₑ= 0.18,
        aᵢ= 0.02,
        bᵢ= 0.21,
        gₑₑ=0.0,
        gₑᵢ=0.23,
        gᵢₑ=0.26,
        gᵢᵢ=0.0,
    )
    # Parameter values
    params = SubsystemParams{NGNMM_theta}(;to_nt(nm)...)
    # Total subsystem
    Subsystem(states, params)
end

# HHExci Neurons can take 3 different types of input, I_syn, I_in and I_asc, each of which can be
# incremented by different connection types
GraphDynamics.initialize_input(s::Subsystem{NGNMM_theta}) = (;)
GraphDynamics.subsystem_differential_requires_inputs(::Type{NGNMM_theta}) = false
function GraphDynamics.subsystem_differential(sys::Subsystem{NGNMM_theta}, _, t)
    (;name, Cₑ, Cᵢ, Δₑ, Δᵢ, η_0ₑ, η_0ᵢ,
     v_synₑₑ, v_synₑᵢ, v_synᵢₑ, v_synᵢᵢ,
     alpha_invₑₑ, alpha_invₑᵢ,
     alpha_invᵢₑ, alpha_invᵢᵢ, kₑₑ, kₑᵢ, kᵢₑ, kᵢᵢ)  = sys
    (;aₑ, bₑ, aᵢ, bᵢ, gₑₑ, gₑᵢ, gᵢₑ, gᵢᵢ)           = sys

    SubsystemStates{NGNMM_theta}(
        #=d/dt=#aₑ = (1/Cₑ)*(bₑ*(aₑ-1)
                             - (Δₑ/2)*((aₑ+1)^2-bₑ^2)
                             - η_0ₑ*bₑ*(aₑ+1)
                             - (v_synₑₑ*gₑₑ+v_synₑᵢ*gₑᵢ)*(bₑ*(aₑ+1))
                             - (gₑₑ/2+gₑᵢ/2)*(aₑ^2-bₑ^2-1)),
        #=d/dt=#bₑ = (1/Cₑ)*((bₑ^2-(aₑ-1)^2)/2
                             - Δₑ*bₑ*(aₑ+1)
                             + (η_0ₑ/2)*((aₑ+1)^2-bₑ^2)
                             + (v_synₑₑ*(gₑₑ/2)+v_synₑᵢ*(gₑᵢ/2))*((aₑ+1)^2-bₑ^2)
                             - aₑ*bₑ*(gₑₑ+gₑᵢ)),
        #=d/dt=#aᵢ = (1/Cᵢ)*(bᵢ*(aᵢ-1)
                             - (Δᵢ/2)*((aᵢ+1)^2-bᵢ^2)
                             - η_0ᵢ*bᵢ*(aᵢ+1)
                             - (v_synᵢₑ*gᵢₑ+v_synᵢᵢ*gᵢᵢ)*(bᵢ*(aᵢ+1))
                             - (gᵢₑ/2+gᵢᵢ/2)*(aᵢ^2-bᵢ^2-1)),
        #=d/dt=#bᵢ = (1/Cᵢ)*((bᵢ^2-(aᵢ-1)^2)/2
                             - Δᵢ*bᵢ*(aᵢ+1)
                             + (η_0ᵢ/2)*((aᵢ+1)^2-bᵢ^2)
                             + (v_synᵢₑ*(gᵢₑ/2)+v_synᵢᵢ*(gᵢᵢ/2))*((aᵢ+1)^2-bᵢ^2)
                             - aᵢ*bᵢ*(gᵢₑ+gᵢᵢ)),
        #=d/dt=#gₑₑ = alpha_invₑₑ*((kₑₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gₑₑ),
        #=d/dt=#gₑᵢ = alpha_invₑᵢ*((kₑᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gₑᵢ),
        #=d/dt=#gᵢₑ = alpha_invᵢₑ*((kᵢₑ/(Cₑ*pi))*((1-aₑ^2-bₑ^2)/(1+2*aₑ+aₑ^2+bₑ^2)) - gᵢₑ),
        #=d/dt=#gᵢᵢ = alpha_invᵢᵢ*((kᵢᵢ/(Cᵢ*pi))*((1-aᵢ^2-bᵢ^2)/(1+2*aᵢ+aᵢ^2+bᵢ^2)) - gᵢᵢ)
    )
end
