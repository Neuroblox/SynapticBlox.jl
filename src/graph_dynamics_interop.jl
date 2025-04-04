function graphsystem_from_graph(g::Neurograph; t_block=missing, name=nothing, global_events=())
    g_flat = Neurograph()
    for blox ∈ vertices(g)
        blox_wiring_rule!(g_flat, blox; is_flattening=true)
    end
    for (;src, dst, data) ∈ edges(g)
        blox_wiring_rule!(g_flat, src, dst; is_flattening=true, data...)
    end

    tstops = Float64[]
    system_is_stochastic = false

    #==================================================================================================
    Create a list of lists of the lowest level blox in the flattened graph, partitioned by their type
    so different types can be handled efficiently

    e.g. if we have
    @named n1 = NeuronType1(x=1, y=2)
	@named n2 = NeuronType1(x=1, y=3)
	@named n3 = NeuronType2(a=1, b=2, c=3)

    in the graph, then we'd end up with

    blox_paritioned = [NeuronType1[n1, n2], NeuronType2[n3]]
    
    ===================================================================================================#
    
    blox_types = (unique ∘ imap)(typeof, vertices(g_flat))
    blox_partitioned = map(blox_types) do T
        if isstochastic(T)
            system_is_stochastic = true
        end
        filter(collect(vertices(g_flat))) do blox
            blox isa T
        end
    end
    
    #==================================================================================================
    Create a ConnectionMatrices object containing structured information about how each lowest level blox 
    is connected to other blox, partitioned by the types of the blox, and the types of the connections for
    type stability.
    e.g. if we have
    
    @named n1 = NeuronType1(x=1, y=2)
	@named n2 = NeuronType1(x=1, y=3)
	@named n3 = NeuronType2(a=1, b=2, c=3)
    
    add_edge!(g, n1, n2; conn=C1(1))
    add_edge!(g, n2, n3; conn=C1(2))
    add_edge!(g, n3, n1; conn=C2(3))
    add_edge!(g, n3, n2; conn=C3(4))
    
    we'd get
    connection_matrix_1 = Conn1[⎡. 1⎤⎡.⎤
	                            ⎣. .⎦⎣2⎦
	                            [. .][.]]
	
	connection_matrix_2 = Conn2[⎡. .⎤⎡.⎤
	                            ⎣. .⎦⎣.⎦
	                            [3 4][.]]
	
	ConnectionMatrices((connection_matrix_1, connection_matrix_2))

    where the sub-matrices are sparse arrays.

    This allows for type-stable calculations involving the neurons and their connections
    ===================================================================================================#
    # connection_types = (unique ∘ imap)(edges(g_flat)) do (; src, dst, data)
    #     if !haskey(data, :conn)
    #         @info "" src.name dst.name data
    #     end
    #     typeof(data.conn)
    # end

    # learning_rules = Dict{@NamedTuple{nc::Int, k::Int, i::Int, l::Int, j::Int}, AbstractLearningRule}()    

    # connection_matrices = (ConnectionMatrices ∘ Tuple ∘ map)(enumerate(connection_types)) do (nc, CT)
    #     (ConnectionMatrix ∘ Tuple ∘ map)(enumerate(blox_partitioned)) do (k, bloxks)
    #         (Tuple ∘ map)(enumerate(blox_partitioned)) do (i, bloxis)
    #             ls = Int[]
    #             js = Int[]
    #             conns = CT[]
    #             for (j, bloxij) ∈ enumerate(bloxis)
    #                 for (l, bloxkl) ∈ enumerate(bloxks)
    #                     if has_edge(g_flat, bloxkl, bloxij)
    #                         (; data) = props(g_flat, bloxkl, bloxij)
    #                         (; conn) = data
    #                         if conn isa CT
    #                             push!(js, j)
    #                             push!(ls, l)
    #                             push!(conns, conn)

    #                             if haskey(data, :learning_rule) && !(data.learning_rule isa NoLearningRule)
    #                                 learning_rules[(;nc, k, i, l, j)] = data.learning_rule
    #                             end
                                
    #                             for t ∈ event_times(conn)
    #                                 push!(tstops, t)
    #                             end
    #                         end
    #                     end
    #                 end
    #             end
    #             if isempty(conns)
    #                 NotConnected()
    #             else
    #                 rule_matrix_sparse = sparse(ls, js, conns, length(bloxks), length(bloxis))
    #                 # Maybe do something if totally dense
    #                 rule_matrix_sparse
    #             end
    #         end
    #     end
    # end

    function make_connection_matrices(conn_key=:conn; pred=(_) -> true)
        connection_types = (imap)(edges(g_flat)) do (; src, dst, data)
            if haskey(data, conn_key) && pred(data[conn_key])
                typeof(data[conn_key])
            else
                nothing
            end
        end |> unique |> x -> filter(!isnothing, x)
        (ConnectionMatrices ∘ Tuple ∘ map)(enumerate(connection_types)) do (nc, CT)
            (ConnectionMatrix ∘ Tuple ∘ map)(enumerate(blox_partitioned)) do (k, bloxks)
                (Tuple ∘ map)(enumerate(blox_partitioned)) do (i, bloxis)
                    ls = Int[]
                    js = Int[]
                    conns = CT[]
                    for (j, bloxij) ∈ enumerate(bloxis)
                        for (l, bloxkl) ∈ enumerate(bloxks)
                            if has_edge(g_flat, bloxkl, bloxij)
                                (; data) = props(g_flat, bloxkl, bloxij)
                                if haskey(data, conn_key)
                                    conn = data[conn_key]
                                    if conn isa CT && pred(conn)
                                        push!(js, j)
                                        push!(ls, l)
                                        push!(conns, conn)
                                        
                                        for t ∈ event_times(conn)
                                            push!(tstops, t)
                                        end
                                    end
                                end
                            end
                        end
                    end
                    if isempty(conns)
                        NotConnected()
                    else
                        rule_matrix_sparse = sparse(ls, js, conns, length(bloxks), length(bloxis))
                        # Maybe do something if totally dense
                        rule_matrix_sparse
                    end
                end
            end
        end
    end
    connection_matrices = make_connection_matrices(:conn)
    learning_rules      = make_connection_matrices(:learning_rule; pred=(x) -> !(x isa NoLearningRule))

    composite_discrete_events_partitioned = nothing
    composite_continuous_events_partitioned = nothing
    
    subsystems_partitioned = (Tuple ∘ map)(v -> map(to_subsystem, v), blox_partitioned)
    states_partitioned = (Tuple ∘ map)(v -> map(get_states, v), subsystems_partitioned)
    params_partitioned = (Tuple ∘ map)(v -> map(get_params, v), subsystems_partitioned)
    names_partitioned  = (Tuple ∘ map)(v -> map(x -> x.name, v), blox_partitioned)

    for v ∈ subsystems_partitioned
        for sys in v
            for t ∈ event_times(sys)
                push!(tstops, t)
            end
        end
    end

    
    extra_params = (; learning_rules)
    
    gsys_args = (;connection_matrices,
                 states_partitioned,
                 params_partitioned,
                 tstops=unique!(tstops),
                 composite_discrete_events_partitioned,
                 composite_continuous_events_partitioned,
                 global_events,
                 names_partitioned,
                 extra_params)
    
    if system_is_stochastic
        SDEGraphSystem(;gsys_args...)
    else
        ODEGraphSystem(;gsys_args...)
    end
end


function t_block_event(key)
    function _apply_t_block_event!(integrator)
        (; params_partitioned, connection_matrices, partition_plan) = integrator.p
        states_partitioned = partitioned(integrator.u, partition_plan)
        t = integrator.t
        # Some t_block events need to happen before others, so we split them into two categories: 'early' and 'late'.
        # the old implementation did this as separate events, but here we can just force the early ones to happen before
        # the late ones without having to have extra events.

        for i ∈ eachindex(states_partitioned)
            states_partitioned_i = states_partitioned[i]
            params_partitioned_i = params_partitioned[i]
            tag = get_tag(eltype(states_partitioned_i))
            if has_t_block_event(tag)
                if is_t_block_event_time(tag, key, t)
                    for j ∈ eachindex(states_partitioned_i)
                        sys_dst = Subsystem(states_partitioned_i[j], params_partitioned_i[j])
                        if t_block_event_requires_inputs(tag)
                            input = calculate_inputs(Val(i), j, states_partitioned, params_partitioned, connection_matrices, t)
                        else
                            input = initialize_input(sys_dst)
                        end
                        apply_t_block_event!(@view(states_partitioned_i[j]), @view(params_partitioned_i[j]), sys_dst, input, t)
                    end
                end
            end
        end
    end
end

has_t_block_event(::Type{Union{}}) = error("Something went very wrong. This error should only exist for method disambiguation")
t_block_event_requires_inputs(::Type{Union{}}) = error("Something went very wrong. This error should only exist for method disambiguation")
has_t_block_event(::Type{T}) where {T} = false
