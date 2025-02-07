using Test, Neuroblox, OrdinaryDiffEq
using SynapticBlox
using GraphDynamics

using Neuroblox: edges, src, dst
using Neuroblox.GraphDynamicsInterop: get_blox
using SymbolicIndexingInterface


#todo, deuplicate this from Neuroblox
function test_compare_du_and_sols(::Type{ODEProblem}, g, tspan;
                                  rtol,
                                  parallel=true, mtk=true, alg=nothing)
    if g isa Tuple
        (gl, gr) = g
    else
        gl = g
        gr = g
    end
    @named gsys = system_from_graph(gl; graphdynamics=true)
    state_names = variable_symbols(gsys)
    sol_grp, du_grp = let sys = gsys
        prob = ODEProblem(sys, [], tspan)
        (; f, u0, p) = prob
        du = similar(u0)
        f(du, u0, p, 1.0)

        sol = solve(prob, alg)
        @test sol.retcode == ReturnCode.Success
        sol_u_reordered = map(state_names) do name
            sol[name][end]
        end
        du_reordered = map(state_names) do name
            getu(sys, name)(du)
        end
        sol_u_reordered, du_reordered
    end
   
    if mtk
        sol_mtk, du_mtk = let @named sys = system_from_graph(gr)
            prob = ODEProblem(sys, [], tspan)
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)

            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            # For some reason getu is erroring here, this is some sort of MTK bug I think
            # du_reordered = map(state_names) do name 
            #     getu(sys, name)(du)
            # end
            du_reordered = du
            sol_u_reordered, du_reordered
        end
        @debug "" norm(sol_grp .- sol_mtk) / norm(sol_mtk)
        for i ∈ eachindex(state_names)
            if !isapprox(sol_grp[i], sol_mtk[i]; rtol=rtol)
                @debug  "" i state_names[i] sol_grp[i] sol_mtk[i]
            end
        end
        @test sort(du_grp) ≈ sort(du_mtk) # due to the MTK getu bug, we'll compare the sorted versions
        @test sol_grp ≈ sol_mtk rtol=rtol
    end
    if parallel
        sol_grp_p, du_grp_p = let sys = gsys
            prob = ODEProblem(sys, [], tspan, scheduler=StaticScheduler())
            (; f, u0, p) = prob
            du = similar(u0)
            f(du, u0, p, 1.0)

            sol = solve(prob, alg)
            @test sol.retcode == ReturnCode.Success
            sol_u_reordered = map(state_names) do name
                sol[name][end]
            end
            du_reordered = map(state_names) do name
                getu(sys, name)(du)
            end
            sol_u_reordered, du_reordered
        end
        @test du_grp ≈ du_grp_p
        @test sol_grp ≈ sol_grp_p rtol=rtol
    end
end


function synaptic_blox_tests()
    @testset "Syntaptic blox basics" begin
        g = MetaDiGraph()

        #-----------------------
        @named n1 = HHExci(;I_bg=0.5)
        @named n2 = HHExci(;)
        @named n3 = HHInhi(;)
        #-----------------------
        
        add_edge!(g, n1 => n3; weight=1.0) # defaults to a GABA_A_Synapse
        add_edge!(g, n1 => n2; weight=1.0) # defailts to a Glu_AMDA_Synapse

        #-----------------------
        #add an edge from n2 to n1 with an explicit GABA_A_Synapse
        add_edge!(g, n2 => n3; synapse=GABA_A_Synapse(name=:gas), weight=1.0)
        
        # a second connection from n1 to n3 using a GABA_B_Synapse
        add_edge!(g, n1 => n3; synapse=GABA_B_Synapse(name=:gbs), weight=1.0)
        
        # a second connection from n1 to n3 using a Glu_NMDA_Synapse
        add_edge!(g, n1 => n2; synapse=Glu_NMDA_Synapse(name=:gnms), weight=1.0)

        add_edge!(g, n3 => n2; synapse=Glu_NMDA_Synapse(name=:gnms2), weight=0.5)
        add_edge!(g, n3 => n1; synapse=Glu_AMPA_Synapse(name=:gams), weight=0.5)
        
        #-----------------------
        for use_gdy ∈ [true, false ]
            @named sys = system_from_graph(g; graphdynamics=use_gdy)

            tspan = (0.0, 1000.0)
            prob = ODEProblem(sys, [], (0.0, 1000.0), [])

            #-----------------------
            
            sol1 = solve(prob, Tsit5())
            @test sol1.retcode == ReturnCode.Success

            #-----------------------
            # Now lets mess with some of these synapses
            prob2 = remake(prob)
            setp(prob2, :gnms₊g_Glu_NMDA)(prob, 2.0)
            setp(prob2, :gbs₊g_GABA_B)(prob, 2.0)
            setp(prob2, :gas₊g_GABA_A)(prob, 0.0)
            setp(prob2, :gnms₊E_syn)(prob, -3.0)
            setp(prob2, :gnms2₊g_Glu_NMDA)(prob, 0.0)
            setp(prob2, :gams₊g_Glu_AMPA)(prob, 0.0)
            
            sol2 = solve(prob2, Tsit5())
            @test sol2.retcode == ReturnCode.Success

            #-----------------------

            @test !(sol1[end] ≈ sol2[end])
            # f1 = stackplot([n1, n2, n3], sol1; threshold=20.0, title="Before")
            # f2 = stackplot([n1, n2, n3], sol2; threshold=20.0, title="After")
            # # save("before.png", f1)
            # # save("after.png", f2)
            # display(f1)
            # display(f2)
        end
        
        test_compare_du_and_sols(ODEProblem, g, (0.0, 1000.0); rtol=1e-7, alg=Tsit5())

    end
end


function synaptic_blox_tests()
    @testset "Syntaptic blox basics" begin
        g = MetaDiGraph()

        #-----------------------
        @named n1 = HHExci(;I_bg=0.5)
        @named n2 = HHExci(;)
        @named n3 = HHInhi(;)
        #-----------------------
        
        add_edge!(g, n1 => n3; weight=1.0) # defaults to a GABA_A_Synapse
        add_edge!(g, n1 => n2; weight=1.0) # defailts to a Glu_AMDA_Synapse

        #-----------------------
        #add an edge from n2 to n1 with an explicit GABA_A_Synapse
        add_edge!(g, n2 => n3; synapse=GABA_A_Synapse(name=:gas), weight=1.0)
        
        # a second connection from n1 to n3 using a GABA_B_Synapse
        add_edge!(g, n1 => n3; synapse=GABA_B_Synapse(name=:gbs), weight=1.0)
        
        # a second connection from n1 to n3 using a Glu_NMDA_Synapse
        add_edge!(g, n1 => n2; synapse=Glu_NMDA_Synapse(name=:gnms), weight=1.0)

        add_edge!(g, n3 => n2; synapse=Glu_NMDA_Synapse(name=:gnms2), weight=0.5)
        add_edge!(g, n3 => n1; synapse=Glu_AMPA_Synapse(name=:gams), weight=0.5)
        
        #-----------------------
        for use_gdy ∈ [true, false ]
            @named sys = system_from_graph(g; graphdynamics=use_gdy)

            tspan = (0.0, 1000.0)
            prob = ODEProblem(sys, [], (0.0, 1000.0), [])

            #-----------------------
            
            sol1 = solve(prob, Tsit5())
            @test sol1.retcode == ReturnCode.Success

            #-----------------------
            # Now lets mess with some of these synapses
            prob2 = remake(prob)
            setp(prob2, :gnms₊g_Glu_NMDA)(prob, 2.0)
            setp(prob2, :gbs₊g_GABA_B)(prob, 2.0)
            setp(prob2, :gas₊g_GABA_A)(prob, 0.0)
            setp(prob2, :gnms₊E_syn)(prob, -3.0)
            setp(prob2, :gnms2₊g_Glu_NMDA)(prob, 0.0)
            setp(prob2, :gams₊g_Glu_AMPA)(prob, 0.0)
            
            sol2 = solve(prob2, Tsit5())
            @test sol2.retcode == ReturnCode.Success

            #-----------------------

            @test !(sol1[end] ≈ sol2[end])
            # f1 = stackplot([n1, n2, n3], sol1; threshold=20.0, title="Before")
            # f2 = stackplot([n1, n2, n3], sol2; threshold=20.0, title="After")
            # # save("before.png", f1)
            # # save("after.png", f2)
            # display(f1)
            # display(f2)
        end
        
        test_compare_du_and_sols(ODEProblem, g, (0.0, 1000.0); rtol=1e-7, alg=Tsit5())

    end
end




# function synaptic_blox_bench(;N=10, NN=10, seed=1234)
#     # @testset "Syntaptic blox basics" begin
#     let 
#         g = MetaDiGraph()
#         neurons = map(1:N) do i
#             HHExci(name=Symbol("n", i))
#         end
#         Random.seed!(seed)
#         for i ∈ 1:N
#             for j ∈ shuffle(1:N)[1:NN]
#                 add_edge!(g, neurons[i] => neurons[j]; weight=randn())
#             end
#         end
#         for use_gdy ∈ [true, false]
#             @named sys = system_from_graph(g; graphdynamics=use_gdy)

#             tspan = (0.0, 1000.0)
#             prob = ODEProblem(sys, [], (0.0, 1000.0), [])

#             #-----------------------
            
#             @btime solve($prob, Tsit5())
#         end
#     end
#     println()
#     let
#         g = MetaDiGraph()
#         neurons = map(1:N) do i
#             HHExci(name=Symbol("n", i))
#         end
#         Random.seed!(seed)
#         for i ∈ 1:N
#             s = Glu_AMPA_Synapse(name=Symbol(:s, i))
#             add_edge!(g, neurons[i] => s; weight=0.0)
#             for j ∈ shuffle(1:N)[1:NN]
#                 add_edge!(g, s => neurons[j]; weight=randn())
#             end
#         end
#         for use_gdy ∈ [true, false]
#             @named sys = system_from_graph(g; graphdynamics=use_gdy)

#             tspan = (0.0, 1000.0)
#             prob = ODEProblem(sys, [], (0.0, 1000.0), [])

#             #-----------------------
            
#             @btime solve($prob, Tsit5())
#         end
#     end
#     println()
#     let 
#         g = MetaDiGraph()
#         neurons = map(1:N) do i
#             HHNeuronExciBlox(name=Symbol("n", i))
#         end
#         Random.seed!(seed)
#         for i ∈ 1:N
#             for j ∈ shuffle(1:N)[1:NN]
#                 add_edge!(g, neurons[i] => neurons[j]; weight=randn())
#             end
#         end
#         for use_gdy ∈ [true, false]
#             @named sys = system_from_graph(g; graphdynamics=use_gdy)

#             tspan = (0.0, 1000.0)
#             prob = ODEProblem(sys, [], (0.0, 1000.0), [])

#             #-----------------------
            
#             @btime solve($prob, Tsit5())
#         end
#     end

#     # end
# end

