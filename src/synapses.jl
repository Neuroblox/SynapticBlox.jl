abstract type AbstractSynapse <: AbstractBlox end

for S ∈ ("GABA_A", "GABA_B")
    T = Symbol(S, "_Synapse")
    @eval begin
        @kwdef struct $T <: AbstractSynapse
            name::Symbol
            E_syn::Float64   = -70.0
            τ₁::Float64      = 0.1
            τ₂::Float64      = 70
            g::Float64       = 1.0
        end

        # Convert a synapse of type $T to a GraphDynamics.Subsystem
        function GraphDynamics.to_subsystem((;name, E_syn, #G_syn, V_shift, V_range,
                               τ₁, τ₂, g)::$T)
            states = SubsystemStates{$T}(G=0.0, z=0.0)
            params = SubsystemParams{$T}(;name, E_syn, #G_syn, V_shift, V_range,
                                         τ₁, τ₂, g)
            Subsystem(states, params)
        end
        
        GraphDynamics.initialize_input(s::Subsystem{$T}) = (;G_asymp = 0.0)

        function GraphDynamics.subsystem_differential(sys::Subsystem{$T}, input, t)
            (; G, z) = sys                                       # Unpack states
            (; #E_syn, G_syn, V_shift, V_range,
             τ₁, τ₂, g) = sys  # Unpack params
            (; G_asymp) = input                                        # Unpack input

            # # Define helper function
            # G_asymp = (G_syn/(1 + exp(-4.394*((V - V_shift)/V_range))))

            # Return the differentials of each state
            SubsystemStates{$T}(
                #=d/dt=#G = -G/τ₂ + z,
                #=d/dt=#z = -z/τ₁ + G_asymp
            )
        end
    end
end


@kwdef struct Glu_AMPA_Synapse <: AbstractSynapse
    name::Symbol
    E_syn::Float64   = 0.0
    τ₁::Float64      = 0.1
    τ₂::Float64      = 5
    g::Float64       = 1
end

# Convert a synapse of type $T to a GraphDynamics.Subsystem
function GraphDynamics.to_subsystem((;name, E_syn, #G_syn, V_shift, V_range, spk_const,
                       τ₁, τ₂, g)::Glu_AMPA_Synapse)
    states = SubsystemStates{Glu_AMPA_Synapse}(G=0.0, z=0.0)
    params = SubsystemParams{Glu_AMPA_Synapse}(;name, E_syn, τ₁, τ₂, g)
    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{Glu_AMPA_Synapse}) = (;G_asymp = 0.0)

function GraphDynamics.subsystem_differential(sys::Subsystem{Glu_AMPA_Synapse}, input, t)
    (; G, z) = sys        # Unpack states
    (; τ₁, τ₂, g) = sys   # Unpack params
    (; G_asymp) = input   # Unpack input
    # Return the differentials of each state
    SubsystemStates{Glu_AMPA_Synapse}(
        #=d/dt=#G = -G/τ₂ + z,
        #=d/dt=#z = -z/τ₁ + G_asymp,
    )
end

@kwdef struct Glu_AMPA_STA_Synapse <: AbstractSynapse
    name::Symbol
    E_syn::Float64 = 0.0
    τ₁::Float64    = 0.1
    τ₂::Float64    = 5
    τ₃::Float64    = 2000
    τ₄::Float64    = 0.1
    kₛₜₚ::Float64  = 0.5
    g::Float64     = 1
end

# Convert a synapse of type $T to a GraphDynamics.Subsystem
function GraphDynamics.to_subsystem((;name, E_syn, τ₁, τ₂, τ₃, τ₄, kₛₜₚ, g)::Glu_AMPA_STA_Synapse)
    states = SubsystemStates{Glu_AMPA_STA_Synapse}(
        G=0.0,
        z=0.0,
        Gₛₜₚ=0.0,
        zₛₜₚ=0.0,
    )
    params = SubsystemParams{Glu_AMPA_STA_Synapse}(;name, E_syn, τ₁, τ₂, τ₃, τ₄, kₛₜₚ, g)
    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{Glu_AMPA_STA_Synapse}) = (;G_asymp_pre = 0.0, G_asymp_post=0.0)

function GraphDynamics.subsystem_differential(sys::Subsystem{Glu_AMPA_STA_Synapse}, input, t)
    (; Gₛₜₚ, zₛₜₚ, G, z) = sys           # Unpack states
    (; τ₁, τ₂, τ₃, τ₄, kₛₜₚ, g) = sys    # Unpack params
    (;G_asymp_pre, G_asymp_post) = input # Unpack input

    # Return the differentials of each state
    SubsystemStates{Glu_AMPA_STA_Synapse}(
        #=d/dt=#G    = -G/τ₂ + z,
        #=d/dt=#z    = -z/τ₁ + G_asymp_pre,
        #=d/dt=#Gₛₜₚ = -Gₛₜₚ/τ₃ + (zₛₜₚ/5)*(kₛₜₚ-Gₛₜₚ),
        #=d/dt=#zₛₜₚ = -zₛₜₚ/τ₄ + G_asymp_post,
    )
end


struct Glu_NMDA_Synapse <: AbstractSynapse
    function Glu_NMDA_Synapse(;)
        error("Not yet implemented")
    end
end




