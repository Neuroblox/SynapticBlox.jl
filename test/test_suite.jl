using Test, OrdinaryDiffEq
using SynapticBlox
using GraphDynamics
using OhMyThreads
using CSV
using SymbolicIndexingInterface
using Random


function simple_synaptic_blox_tests()
    # @testset "HH Neuron excitatory & inhibitory network"
    begin
        @named nn1 = HHExci(I_bg=3)
        @named nn2 = HHExci(I_bg=2)
        @named nn3 = HHInhi(I_bg=2)
        
        g = Neurograph()
        
        # Adjacency matrix : 
        #adj = [0   1 0
        #       0   0 1
        #       0.2 0 0]
        SynapticBlox.add_vertex!(g, nn1)
        add_edge!(g, nn1, nn2; weight=1)
        add_edge!(g, nn2, nn1; weight=1)
        add_edge!(g, nn2, nn3; weight=1)
        add_edge!(g, nn3, nn1; weight=0.2)
        
        neuron_net = graphsystem_from_graph(g)
        prob = ODEProblem(neuron_net, [], (0.0, 20.0), [])
        sol = solve(prob, Tsit5())

        @test sol.retcode == ReturnCode.Success
    end
end



function synaptic_blox_tests()
    begin
        g = Neurograph()
        #-----------------------
        @named n1 = HHExci(;I_bg=0.5)
        @named n2 = HHExci(;)
        @named n3 = HHInhi(;)
        #-----------------------
        add_edge!(g, n1, n3; weight=1.0) # defaults to a GABA_A_Synapse
        add_edge!(g, n1, n2; weight=1.0) # defailts to a Glu_AMDA_Synapse

        add_edge!(g, n2, n3; synapse=Glu_AMPA_Synapse(name=:gams, τ₁=0.5, τ₂=100, E_syn=-80, g=10), weight=1.0)
        add_edge!(g, n3, n1; synapse=GABA_B_Synapse(name=:gbs), weight=0.5)
        
        #-----------------------

        sys = graphsystem_from_graph(g)
        
        tspan = (0.0, 1000.0)
        prob = ODEProblem(sys, [], (0.0, 1000.0), [])

        #-----------------------
        # # return sys.states_partitioned
        sol1 = @time solve(prob, Tsit5())

        # p = plot(sol1.t, sol1[:n1₊V])
        # plot!(sol1.t, sol1[:n2₊V])
        # plot!(sol1.t, sol1[:n3₊V])

        # return p
        
        @test sol1.retcode == ReturnCode.Success

        #-----------------------
        # Now lets mess with some of these synapses
        prob2 = remake(prob)
        setp(prob2, [:gbs₊g, :gams₊g])(prob2, [2.0, 3.0])
        
        # setp(prob2, :gnms₊g_Glu_NMDA)(prob, 2.0)
        # setp(prob2, :gbs₊g_GABA_B)(prob, 2.0)
        # setp(prob2, :gas₊g_GABA_A)(prob, 0.0)
        # setp(prob2, :gnms₊E_syn)(prob, -3.0)
        # setp(prob2, :gnms2₊g_Glu_NMDA)(prob, 0.0)
        # setp(prob2, :gams₊g_Glu_AMPA)(prob, 0.0)
        
        sol2 = solve(prob2, Tsit5())
        @test sol2.retcode == ReturnCode.Success

        #-----------------------

        @test !(sol1[end] ≈ sol2[end])
    end
end

function lflic_tests()
    begin
        let g = Neurograph() 
            @named lf1 = L_FLICBlox()
            add_vertex!(g, lf1)
            sys = graphsystem_from_graph(g)
            prob = ODEProblem(sys, [], (0.0, 10.0), [])
            sol = solve(prob, Tsit5())
            @test sol.retcode == ReturnCode.Success
        end
        let g = Neurograph() 
            @named lf1 = L_FLICBlox()
            @named lf2 = L_FLICBlox()
            
            add_edge!(g, lf1, lf2; weight=1.0, density=1.0)

            sys = graphsystem_from_graph(g)
            prob = ODEProblem(sys, [], (0.0, 10.0), [])
            sol = solve(prob, Tsit5())
            @test sol.retcode == ReturnCode.Success
        end
        let g = Neurograph() 
            @named inh = HHInhi()
            @named lf2 = L_FLICBlox()
            
            add_edge!(g, inh, lf2; weight=1.0)

            sys = graphsystem_from_graph(g)
            prob = ODEProblem(sys, [], (0.0, 10.0), [])
            sol = solve(prob, Tsit5())
            @test sol.retcode == ReturnCode.Success
        end
    end
end

function cortical_tests()
    begin
        let g = Neurograph()
            @named cb = CorticalBlox(N_wta=5, N_exci=5, density=0.1, weight=1)
            add_vertex!(g, cb)
            sys = graphsystem_from_graph(g)
            prob = ODEProblem(sys, [], (0.0, 10.0),[])
            sol = solve(prob, Vern7())
            @test sol.retcode == ReturnCode.Success
        end
        let g = Neurograph()
            @named cb1 = CorticalBlox(N_wta=5, N_exci=5, density=0.1, weight=1)
            @named cb2 = CorticalBlox(N_wta=2, N_exci=3, density=0.15, weight=1)
            add_edge!(g, cb1, cb2; weight=1.0, density=0.1)
            sys = graphsystem_from_graph(g)
            prob = ODEProblem(sys, [], (0.0, 10.0),[])
            sol = solve(prob, Vern7())
            @test sol.retcode == ReturnCode.Success
        end
    end
end

