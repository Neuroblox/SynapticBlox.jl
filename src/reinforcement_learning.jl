function run_experiment!(agent::Agent, env::ClassificationEnvironment; t_warmup=0, verbose=false, kwargs...)
    N_trials = env.N_trials
    t_trial = env.t_trial
    tspan = (0, t_trial)
    @time begin 
        if t_warmup > 0
            u0 = @noinline run_warmup(agent, env, t_warmup; kwargs...)
            @reset agent.problem = remake(agent.problem; tspan, u0=u0)
        else
            @reset agent.problem = remake(agent.problem; tspan, p=init_params)
        end
        print("Warmed up in: ")
    end
    trace = @NamedTuple{trial::Int, iscorrect::Bool, action::Int, time::Float64}[]
    prog = Progress(N_trials; showspeed=true, enabled=verbose)
    try
        for trial ∈ 1:N_trials
            (;time, gctime) = @timed begin
                _, iscorrect, action = @noinline run_trial!(agent, env; kwargs...)
            end
            push!(trace, (;trial, iscorrect, action, time))            
            next!(prog, showvalues=showvalues(;trace, N_trials))
        end
    catch e;
        @warn "Error during run_experiment! Terminating early" e
    finally
        finish!(prog)
    end
    trace
end

maybe_show_plot(args...; kwargs...) = []
showvalues(;trace, N_trials) = () -> begin
    pct_correct = round(sum(row -> row.iscorrect, trace)*100/length(trace); digits=3)
    len=min(100, length(trace))
    pct_correct_recent = let
        pct = sum(1:len) do i
            trace[end-i+1].iscorrect
        end / len
        (round(pct*100; digits=3))
    end
    N = 8
    last_N = map(min(length(trace)-1, N-1):-1:0) do i
        (;trial, iscorrect, action, time) = trace[end-i]
        Response = iscorrect ? "\e[0;32mCorrect\e[0m," : "\e[0;31mFalse\e[0m,  "
        trial_str = rpad("$trial,", textwidth(string(N_trials))+1)
        ("Trial", "$trial_str Category choice = $(action), Response = $Response Time = $(round(time, digits=3)) seconds")
    end
    [
        last_N
        ("Accuracy", "$(pct_correct)% total, $(pct_correct_recent)% last $len trials")
        maybe_show_plot(trace)
    ]
end


function runningperf(trace; len=min(150, length(trace)))
    sum(1:len) do i
        trace[end-i+1].iscorrect
    end / len
end

function run_warmup(agent::Agent, env::ClassificationEnvironment, t_warmup; alg, kwargs...)
    prob = remake(agent.problem; tspan=(0, t_warmup))
    sol = solve(prob, alg; save_everystep=false, kwargs...)
    u0 = sol[:,end] # last value of state vector
    return u0
end

function run_trial!(agent, env; alg, kwargs...)
    prob = agent.problem
    action_selection = agent.action_selection
    learning_rules = agent.learning_rules

    update_trial_stimulus!(prob, env)
    
    sol = solve(prob, alg; kwargs...)
    if isnothing(action_selection)
        feedback = 1
        action = 0
    else
        action = action_selection(sol)
        feedback = env(action)
    end
    apply_learning_rules!(sol, prob, learning_rules, feedback)
    increment_trial!(env)
    
    return sol, feedback, action
end


function update_trial_stimulus!(prob, env::ClassificationEnvironment)
    (;params_partitioned) = prob.p
    for i ∈ eachindex(params_partitioned)
        if eltype(params_partitioned[i]) <: SubsystemParams{ImageStimulus}
            stim = only(params_partitioned[i])
            stim.current_image .= stim.IMG[:, env.current_trial]
        end
    end
end

function apply_learning_rules!(sol, prob, learning_rules, feedback)
    (;connection_matrices, params_partitioned) = prob.p
    _apply_learning_rules!(sol, params_partitioned, connection_matrices, learning_rules, feedback)
end
function _apply_learning_rules!(sol,
                                params_partitioned::NTuple{Len, Any},
                                connection_matrices::ConnectionMatrices{NConn},
                                learning_rules::ConnectionMatrices{NLearn},
                                feedback) where {Len, NConn, NLearn}
    for i ∈ eachindex(params_partitioned)
        for k ∈ eachindex(params_partitioned)
            for ncl ∈ 1:length(learning_rules)
                M_learning = learning_rules[ncl].data[k][i]  
                if !(M_learning isa NotConnected)
                    for nc ∈ 1:length(connection_matrices)
                        M = connection_matrices[nc].data[k][i]
                        if !(M isa NotConnected)
                            for j ∈ eachindex(params_partitioned[i])
                                for (l, rule) ∈ maybe_sparse_enumerate_col(M_learning, j)
                                    conn = M[l, j]
                                    Δw = weight_gradient(rule, sol, conn.weight, feedback)
                                    if !isfinite(Δw)
                                        name_dst = params_partitioned[i][j].name
                                        name_src = params_partitioned[k][l].name
                                        @warn "non-finite gradient" name_dst name_src Δw
                                    end
                                    M[l, j] = @reset conn.weight += Δw
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
