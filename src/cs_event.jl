
# For each striatum connected to another striatum
# #  strio_dst.matrisome.H = 1
# #  strio_dst.striosome.H = 1
# #
# #  for neuron in exci_neurons_dst
# #      neuron.I_bg = 0    
# #  end

#  For each Striatum => Striatum connection (t_event + sqrt(eps(t_event)))
#    strio_dst.matrisome.H = strio_src.matrisome.H * strio_str.matrisome.jcn > strio_dst.matrisome.H * strio_dst.matrisome.jcn ? 0 : 1
#    strio_dst.striosome.H = strio_src.matrisome.H * strio_str.matrisome.jcn > strio_dst.matrisome.H * strio_dst.matrisome.jcn ? 0 : 1
# 
#    for neuron in exci_neurons_dst
#        neuron.I_bg = strio_src.matrisome.H * strio_str.matrisome.jcn > strio_dst.matrisome.H * strio_dst.matrisome.jcn ? -2 : 0
#    end

#================
t = 0.1 event:
When the event triggers, we should do

#  For each Matrisome-Matrisome connection
#    matrisome_dst.is_loser = false
#    matrisome_dst.H = 1

#  For each Matrisome => Striosome connection (this comes from the old Striatum => Striatum connection)
#    striosome_dst.H = 1

#  For each Matrisome => HHExci connection
#    neuron.I_bg = 0


================#
#================

t_block event:
When the event triggers, we should do 

================================
# t_event actions:

#  For each matrisome
#    jcn_snapshot = jcn
#    H_snapshot = H

#  For each SNc 
#    jcn_snapshot = jcn

#  For each Striosome
#    jcn_ref = jcn

#  For each Matrisome-Matrisome connection
#    matrisome_dst.is_loser = matrisome_src.H * matrisome_src.jcn > matrisome_dst.H * matrisome_dst.jcn
#    matrisome_dst.H = matrisome_dst.is_loser ? 0 : 1

================================
t_event + sqrt(eps(t_event)) actions:

#  For each TAN => Matrisome connection (t_event + sqrt(eps(t_event)))
#    (;κ,) = tan_src
#    (;H, jcn_snapshot, H_snapshot) = mat_dst
#    R = min(κ/(jcn_src + sqrt(eps())), κ)
#    mat_dst.TAN_spikes = float(rand(Poisson(R)))

#  For each Matrisome => Striosome connection (this comes from the old Striatum => Striatum connection)
#    striosome_dst.H = matrisome_src.is_loser ? 0.0 : 1.0

#  For each Matrisome => HHExci connection
#    neuron.I_bg = matrisome_src.is_loser ? -2.0 : 0.0

================#

@kwdef struct CSEventHandler
    name::Symbol
    t_event::Float64
    striatum_striatum_data::Vector{Pair} = Pair[]
end

function to_subsystem(e::CSEventHandler)
    states = SubsystemStates{CSEventHandler}()
    (;name, t_event, striatum_striatum_data) = e
    params = SubsystemParams{CSEventHandler}(;name, t_event, striatum_striatum_data)
    Subsystem(states, params)
end

GraphDynamics.initialize_input(::Subsystem{CSEventHandler}) = (;)
GraphDynamics.subsystem_differential_requires_inputs(::Type{CSEventHandler}) = false
function GraphDynamics.apply_subsystem_differential!(_, m::Subsystem{CSEventHandler}, _, _)
    nothing
end

GraphDynamics.has_discrete_events(::Type{CSEventHandler}) = true
GraphDynamics.event_times((;t_event)::Subsystem{CSEventHandler}) = (0.1, t_event)
function GraphDynamics.discrete_event_condition((;t_event)::Subsystem{CSEventHandler}, t, _)
    t == t_event || t == 0.1
end

function GraphDynamics.apply_discrete_event!(integrator, _, _, e::Subsystem{CSEventHandler}, _)
    (;params_partitioned, state_types_val, connection_matrices) = integrator.p
    u = integrator.u
    t = integrator.t
    states_partitioned = to_vec_o_states(u.x, state_types_val)

    if t == 0.1

        matri_matri_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
        matri_strio_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
        matri_inhi_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
        
    elseif t == e.t_event

        # First take snapshots of the states *before* we start modifying stuff
        matri_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
        snc_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
        strio_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
        
        # Matri-event has to happen here because it decides which matrisomes are winners and which are losers
        matri_competition_event(states_partitioned, params_partitioned, connection_matrices, t)

        # Then we can do all the other events that depend on whether the matrisome won or lost
        matri_strio_event(states_partitioned, params_partitioned, connection_matrices, t)
        matri_inhi_event(states_partitioned, params_partitioned, connection_matrices, t)
    else
        error("Invalid time $t, expected either t=0.1 or t=$(e.t_event)")
    end
end


function matri_matri_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Matrisome})
            states_matrisome = states_partitioned[i]
            params_matrisome = params_partitioned[i]
            for j ∈ eachindex(states_matrisome)
                
                params_dst = params_matrisome[j]

                for nc ∈ eachindex(connection_matrices.matrices)
                    M = connection_matrices[nc][i, i]
                    if !(M isa NotConnected)
                        for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                            # If we hit this block, that means there's a connection from a Matrisome to params_dst
                            @reset params_dst.H = 1
                            params_matrisome[j] = params_dst
                        end
                    end
                end
            end
        end
    end
end

function matri_strio_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Striosome})
            states_striosome = states_partitioned[i]
            params_striosome = params_partitioned[i]
            for j ∈ eachindex(states_striosome)
                
                params_dst = params_striosome[j]

                for k ∈ eachindex(states_partitioned)
                    if eltype(states_partitioned[k]) <: SubsystemStates{Matrisome}
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                    # If we hit this block, that means there's a connection from a Matrisome to params_dst
                                    @reset params_dst.H = 1
                                    params_striosome[j] = params_dst
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function matri_inhi_initial_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{HHInhi})
            states_exci = states_partitioned[i]
            params_exci = params_partitioned[i]
            for j ∈ eachindex(states_exci)
                
                params_dst = params_exci[j]

                has_connection_from_matrisome = false

                for k ∈ eachindex(states_partitioned)
                    if eltype(states_partitioned[k]) <: SubsystemStates{Matrisome}
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                    # If we hit this block, that means there's a connection from a Matrisome to params_dst
                                    @reset params_dst.I_bg = 0.0
                                    params_exci[j] = params_dst
                                    
                                end
                            end
                        end
                    end
                end                
            end
        end
    end
end

function matri_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Matrisome})
            states_matrisome = states_partitioned[i]
            params_matrisome = params_partitioned[i]
            for j ∈ eachindex(states_matrisome)
                params = params_matrisome[j]
                input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)

                @reset params.jcn_snapshot = input.jcn

                params_matrisome[j] = params
            end
        end
    end
end
function snc_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{SNc})
            states_snc = states_partitioned[i]
            params_snc = params_partitioned[i]
            for j ∈ eachindex(states_snc)
                
                params = params_snc[j]
                input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)

                @reset params.jcn_snapshot = input_dst.jcn

                params_snc[j] = params
                
            end
        end
    end
end
function strio_snapshot_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Striosome})
            params_strio = params_partitioned[i]
            for j ∈ eachindex(params_strio)

                params = params_strio[j]
                input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)

                @reset params.jcn_snapshot = input.jcn

                params_strio[j] = params
            end
        end
    end
end



function matri_competition_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Matrisome})
            states_matrisome = states_partitioned[i]
            params_matrisome = params_partitioned[i]
            for j ∈ eachindex(states_matrisome)
                
                params_dst = params_matrisome[j]
                input_dst = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)


                for k ∈ eachindex(states_partitioned)
                    if eltype(k) <: SubsystemStates{Matrisome}
                        # search for competitor matrisomes, and if there are any, decide who loses
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)

                                    params_src = params_partitioned[k][l]
                                    input_src = calculate_inputs(Val(k), l, states_partitioned, params_partitioned, connection_matrices, t)
                                    
                                    @reset params_dst.is_loser = params_src.H * input_src.jcn > params_dst.H * input_dst.jcn
                                    @reset params_dst.H = params_dst.is_loser ? 0 : 1

                                    params_matrisome[j] = params_dst

                                    @info "Winner!" params_dst.name params_src.name params_dst.is_loser
                                    
                                end
                            end
                        end
                    elseif eltype(states_partitioned[k]) <: SubsystemStates{TAN}
                        #Find any TANs connected to the matrisome, and do the poisson sampling affect
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                    params_src = params_partitioned[k][l]
                                    input_src = calculate_inputs(Val(k), l, states_partitioned, params_partitioned, connection_matrices, t)
                                    R = min(params_src.κ/(input_src.jcn, + √(eps())) , params_src.κ)
                                    params_matrisome[j] = @reset params_dst.TAN_spikes = rand(Poisson(R))
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function matri_strio_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{Striosome})
            states_striosome = states_partitioned[i]
            params_striosome = params_partitioned[i]
            for j ∈ eachindex(states_striosome)
                
                params_dst = params_striosome[j]

                for k ∈ eachindex(states_partitioned)
                    if eltype(states_partitioned[k]) <: SubsystemStates{Matrisome}
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                    # If we hit this block, that means there's a connection from a Matrisome to params_dst
                                    params_src = params_partitioned[k][l]
                                    params_striosome[j] = @set params_dst.H = params_src.is_loser ? 0 : 1
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end



function matri_inhi_event(states_partitioned, params_partitioned, connection_matrices, t)
    for i ∈ eachindex(states_partitioned)
        if (eltype(states_partitioned[i]) <: SubsystemStates{HHInhi})
            states_exci = states_partitioned[i]
            params_exci = params_partitioned[i]
            for j ∈ eachindex(states_exci)
                
                params_dst = params_exci[j]

                has_connection_from_matrisome = false

                for k ∈ eachindex(states_partitioned)
                    if eltype(states_partitioned[k]) <: SubsystemStates{Matrisome}
                        for nc ∈ eachindex(connection_matrices.matrices)
                            M = connection_matrices[nc][k, i]
                            if !(M isa NotConnected)
                                for (l, Mlj) ∈ maybe_sparse_enumerate_col(M, j)
                                    params_src = params_partitioned[k][l]
                                    # If we hit this block, that means there's a connection from a Matrisome to params_dst
                                    @reset params_dst.I_bg = params_src.is_loser ? -2.0 : 0.0
                                    params_exci[j] = params_dst
                                    
                                end
                            end
                        end
                    end
                end                
            end
        end
    end
end
