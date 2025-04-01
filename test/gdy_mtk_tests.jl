module Test1
using Test

module MTK
using Neuroblox, OrdinaryDiffEq, Test
using Random: seed!

function test1(t_dur)
    g = MetaDiGraph()
    seed!(1234)
    @named n1 = HHNeuronExciBlox(I_bg=1.0, freq=0.0)
    @named n2 = HHNeuronExciBlox(I_bg=0.0, freq=0.0)
    @named n3 = HHNeuronInhibBlox(I_bg=0.0, freq=0.0, E_syn=-70, G_syn=1.2, τ=70)
    @named n4 = HHNeuronInhibBlox(I_bg=0.0, freq=0.0, E_syn=-70, G_syn=8,   τ=70)
    
    add_edge!(g, n1 => n2; weight=1.0, sta=true)
    # add_edge!(g, n2 => n3; weight=0.5)
    # add_edge!(g, n3 => n1; weight=1.5)
    #add_edge!(g, n3 => n4; weight=1.0)
    # add_blox!.((g,), [n1, n2, n3])
    #add_blox!.((g,), [n4])
    
    @named sys = system_from_graph(g; )
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Tsit5())

    # sol(t_dur; idxs=[n1.V, n2.V, n3.V,
    #                  n4.V
    #                  ])
end

end

module GDY
using SynapticBlox, OrdinaryDiffEq, Test
using Random: seed!
function test1(t_dur)
    g = Neurograph()
    seed!(1234)
    @named n1 = HHExci(I_bg=1.0)
    @named n2 = HHExci(I_bg=0.0)
    @named n3 = HHInhi(I_bg=0.0, E_syn=-70, G_syn=1.2, τ=70)
    @named n4 = HHInhi(I_bg=0.0, E_syn=-70, G_syn=8,   τ=70)

    add_edge!(g, n1 => n2; weight=1.0, sta=true)
    # add_edge!(g, n2 => n3; weight=0.5)
    # add_edge!(g, n3 => n1; weight=1.5)
    #add_edge!(g, n3 => n4; weight=1.0)
    # add_vertex!.((g,), [n1, n2, n3])
    # add_vertex!.((g,), [n4])
    
    sys = graphsystem_from_graph(g;)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Tsit5())

    # sol(t_dur; idxs=[:n1₊V, :n2₊V, :n3₊V, :n4₊V])
    
end

end 

comp(t_dur) = let solgdy=GDY.test1(t_dur), solmtk=MTK.test1(t_dur)
    gdy = solgdy(t_dur; idxs=[:n1₊V, :n2₊V, # :n3₊V, :n4₊V
                              ])
    mtk = solmtk(t_dur; idxs=[:n1₊V, :n2₊V, # :n3₊V, :n4₊V
                              ])
    
    [gdy mtk (gdy-mtk)]
end
test1(t_dur) = @test GDY.test1(t_dur) ≈ MTK.test1(t_dur) rtol=1e-6

using Plots
function plt(t_dur)
    gdy=GDY.test1(t_dur)
    mtk=MTK.test1(t_dur)
    plot(gdy; idxs=[#:n1_synapse_STA₊Gₛₜₚ
                    :n2₊V,
                    # :n3₊V, :n4₊V
                    ])
    plot!(mtk; idxs=[#:n1₊Gₛₜₚ
                     :n2₊V,
                     # :n3₊V, :n4₊V
                     ])
end
end


# ======================================================================
# ======================================================================

module Test2
using Test

module MTK
using Neuroblox, OrdinaryDiffEq, Test
using Neuroblox: getp
using Random: seed!

function test()
    t_dur = 11.0
    namespace = :g
    g = MetaDiGraph()
    seed!(1234)
    @named cb = CorticalBlox(;N_wta=3, N_exci=3, density=0.5, weight=1.0, namespace)
    add_blox!(g, cb)

    @named sys = system_from_graph(g;)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    sol(t_dur; idxs=[sys.cb.ff_inh.V,
                     sys.cb.wta1.inh.V, sys.cb.wta1.exci1.V, sys.cb.wta1.exci2.V,
                     sys.cb.wta2.inh.V, sys.cb.wta2.exci1.V, sys.cb.wta2.exci2.V,
                     sys.cb.wta3.inh.V, sys.cb.wta2.exci3.V, sys.cb.wta3.exci2.V,
                     
                     sys.cb.wta1.exci1.spikes_window, sys.cb.wta1.exci2.spikes_window,
                     sys.cb.wta2.exci1.spikes_window, sys.cb.wta2.exci2.spikes_window,
                     sys.cb.wta3.exci1.spikes_window, sys.cb.wta3.exci2.spikes_window, 
                     ])

end


end

module GDY
using SynapticBlox, OrdinaryDiffEq, Test
using SynapticBlox: getp
using Random: seed!

function test()
    t_dur = 11.0
    
    g = Neurograph()
    seed!(1234)
    @named cb = CorticalBlox(;N_l_flic=3, N_exci=3, density=0.5, weight=1.0,)
    add_vertex!(g, cb)
    
    @named sys = graphsystem_from_graph(g)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    sol(t_dur; idxs=[:cb₊ff_inh₊V,
                     :cb₊l_flic1₊inhi₊V, :cb₊l_flic1₊exci1₊V, :cb₊l_flic1₊exci2₊V,
                     :cb₊l_flic2₊inhi₊V, :cb₊l_flic2₊exci1₊V, :cb₊l_flic2₊exci2₊V,
                     :cb₊l_flic3₊inhi₊V, :cb₊l_flic3₊exci1₊V, :cb₊l_flic3₊exci2₊V,
                     
                     :cb₊l_flic1₊exci1₊spikes_window, :cb₊l_flic1₊exci2₊spikes_window,
                     :cb₊l_flic2₊exci1₊spikes_window, :cb₊l_flic2₊exci2₊spikes_window,
                     :cb₊l_flic3₊exci1₊spikes_window, :cb₊l_flic3₊exci2₊spikes_window,
                     ])

end

end 

test() = @test GDY.test() ≈ MTK.test() rtol=1e-6
comp() = GDY.test() - MTK.test()
end


# ======================================================================
# ======================================================================

module Test3
using Test

module MTK
using Neuroblox, OrdinaryDiffEq, Test
using Neuroblox: getp
using Random: seed!

function test(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    namespace = :g
    g = MetaDiGraph()
    seed!(1234)
    @named cb = CorticalBlox(;N_wta=2, N_exci=2, density=0.5, weight=1.0, namespace)
    @named str1 = Striatum(; N_inhib=2, namespace)
    @named str2 = Striatum(; N_inhib=2, namespace)
    add_edge!(g, cb => str1; density=0.5, weight=1.0)
    add_edge!(g, str1 => str2; density=0.5, t_event=181.0, weight=1.0)
    add_edge!(g, str2 => str1; density=0.5, t_event=181.0, weight=1.0)

    @named sys = system_from_graph(g; t_block=100)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    [sol(t_dur; idxs=[# sys.cb.ff_inh.V,
                      # sys.cb.wta1.inh.V, sys.cb.wta1.exci1.V, sys.cb.wta1.exci2.V,
                      # sys.cb.wta2.inh.V, sys.cb.wta2.exci1.V, sys.cb.wta2.exci2.V,
                      sys.cb.wta1.exci1.spikes_window, sys.cb.wta1.exci2.spikes_window,
                      sys.cb.wta2.exci1.spikes_window, sys.cb.wta2.exci2.spikes_window,
                      # sys.str1.inh1.V, sys.str1.inh2.V,
                      # sys.str2.inh1.V, sys.str2.inh2.V,
                      ])
     # sol[sys.str1.matrisome.ρ, end]
     # sol[sys.str1.matrisome.ρ_, end]
     # sol[sys.str2.matrisome.ρ, end]
     # sol[sys.str2.matrisome.ρ_, end]

     getp(sol, sys.str1.matrisome.jcn)(sol)
     getp(sol, sys.str1.matrisome.jcn_)(sol)
     getp(sol, sys.str2.matrisome.jcn)(sol)
     getp(sol, sys.str2.matrisome.jcn_)(sol)

     # getp(sol, sys.str1.matrisome.t_store)(sol)
     # getp(sol, sys.str2.matrisome.t_store)(sol)
     
     
     # getp(sol, sys.str1.matrisome.H)(sol)
     # getp(sol, sys.str1.matrisome.H_)(sol)
     # getp(sol, sys.str2.matrisome.H)(sol)
     # getp(sol, sys.str2.matrisome.H_)(sol)

     
     # getp(sol, sys.str1.inh1.I_bg)(sol)
     # getp(sol, sys.str1.inh2.I_bg)(sol)
     # getp(sol, sys.str2.inh1.I_bg)(sol)
     # getp(sol, sys.str2.inh2.I_bg)(sol)
     ]

end


end

module GDY
using SynapticBlox, OrdinaryDiffEq, Test
using SynapticBlox: getp
using Random: seed!

function test(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    g = Neurograph()
    seed!(1234)
    @named cb = CorticalBlox(;N_l_flic=2, N_exci=2, density=0.5, weight=1.0,)
    @named str1 = Striatum(N_inhib=2)
    @named str2 = Striatum(N_inhib=2)
    add_edge!(g, cb, str1; density=0.5, weight=1.0)
    add_edge!(g, str1 => str2; density=0.5, t_event=181.0, weight=1.0)
    add_edge!(g, str2 => str1; density=0.5, t_event=181.0, weight=1.0)
    
    @named sys = graphsystem_from_graph(g; t_block=100, N_t_block=8)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    [sol(t_dur; idxs=[# :cb₊ff_inh₊V,
                      # :cb₊l_flic1₊inhi₊V, :cb₊l_flic1₊exci1₊V, :cb₊l_flic1₊exci2₊V,
                      # :cb₊l_flic2₊inhi₊V, :cb₊l_flic2₊exci1₊V, :cb₊l_flic2₊exci2₊V,
                      :cb₊l_flic1₊exci1₊spikes_window, :cb₊l_flic1₊exci2₊spikes_window,
                      :cb₊l_flic2₊exci1₊spikes_window, :cb₊l_flic2₊exci2₊spikes_window,
                      # :str1₊inh1₊V, :str1₊inh2₊V,
                      # :str2₊inh1₊V, :str2₊inh2₊V,
                      # :str1₊matrisome₊ρ
                      # :str1₊matrisome₊ρ 
                      ])
     # sol[:str1₊matrisome₊ρ,  end]
     # sol[:str1₊matrisome₊ρ_snapshot,  end]
     # sol[:str2₊matrisome₊ρ,  end]
     # sol[:str2₊matrisome₊ρ_snapshot,  end]

     getp(sol, :str1₊matrisome₊jcn_t_block)(sol)
     getp(sol, :str1₊matrisome₊jcn_snapshot)(sol)
     getp(sol, :str2₊matrisome₊jcn_t_block)(sol)
     getp(sol, :str2₊matrisome₊jcn_snapshot)(sol)

     # getp(sol, :str1₊matrisome₊H)(sol)
     # getp(sol, :str1₊matrisome₊H_snapshot)(sol)
     # getp(sol, :str2₊matrisome₊H)(sol)
     # getp(sol, :str2₊matrisome₊H_snapshot)(sol)
     
     
     # getp(sol, :str1₊inh1₊I_bg)(sol)
     # getp(sol, :str1₊inh2₊I_bg)(sol)
     # getp(sol, :str2₊inh1₊I_bg)(sol)
     # getp(sol, :str2₊inh2₊I_bg)(sol)
     ]
end

end 

test(t_dur) = @test GDY.test(t_dur) ≈ MTK.test(t_dur) rtol=1e-6
comp(t_dur) = let mtk=MTK.test(t_dur), gdy=GDY.test(t_dur)
    [gdy mtk (gdy-mtk)]
end

end


# ======================================================================
# ======================================================================

module Test4
using Test

module MTK
using Neuroblox, OrdinaryDiffEq, Test
using Neuroblox: getp
using Random: seed!

function test(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    namespace = :g
    g = MetaDiGraph()
    seed!(1234)
    @named str1 = Striatum(; N_inhib=1, namespace)
    @named gpi1 = GPi(N_inhib=1, namespace, I_bg=zeros(3))
    add_edge!(g, str1 => gpi1, weight = 0.0, density = 1.5) #str1->gpi1

    @named sys = system_from_graph(g; t_block=100)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    sol(t_dur; idxs=[sys.str1.inh1.V,
                     sys.gpi1.inh1.V,
                     # sys.gpi1.inh2.V,
                     # sys.gpi1.inh3.V
                     ])
    sol
end

function test2(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    sol1 = let g = MetaDiGraph()
        seed!(1234)
        namespace =:g
        @named str1 = Striatum(; namespace, N_inhib=1, )
        @named gpi1 = GPi(; namespace, N_inhib=1, I_bg=zeros(3))
        add_edge!(g, str1 => gpi1, weight = 4, density = 1.5) #str1->gpi1
        
        @named sys = system_from_graph(g; t_block=100, N_t_block=8)
        prob = ODEProblem(sys, [], (0.0, t_dur), [])
        sol = solve(prob, Vern7())

        sol(t_dur; idxs=[:str1₊inh1₊V,
                         :gpi1₊inh1₊V,
                         # :gpi1₊inh2₊V,
                         # :gpi1₊inh3₊V,
                         ])
    end
    sol2 = let g = MetaDiGraph()
        seed!(1234)
        namespace =:g
        n1 = let E_syn_inhib=-70, G_syn_inhib=1.2, I_bg=zeros(1), freq=zeros(1), phase=zeros(1), τ_inhib=70
            i = 1
            @named str1₊inh1 = HHNeuronInhibBlox(
                E_syn = E_syn_inhib, 
                G_syn = G_syn_inhib, 
                τ = τ_inhib,
                I_bg = I_bg[i],
                freq = freq[i],
                phase = phase[i]
            )
        end
        n2 = let E_syn_inhib=-70, G_syn_inhib=8, I_bg=zeros(1), freq=zeros(1), phase=zeros(1), τ_inhib=70
            i = 1
            @named gpi1₊inh1 = HHNeuronInhibBlox( 
                E_syn = E_syn_inhib, 
                G_syn = G_syn_inhib, 
                τ = τ_inhib,
                I_bg = I_bg[i],
                freq = freq[i],
                phase = phase[i]
            ) 
        end
        add_edge!(g, n1 => n2; weight=4)
        # add_edge!(g, str1 => gpi1, weight = 4, density = 1.5) #str1->gpi1

        #SynapticBlox.hypergeometric_connections!(g, str1.inhibs, gpi1.inhibs, :str1, :gpi1; density=1, weight=4.0)

        @named sys = system_from_graph(g; t_block=100, N_t_block=8)
        prob = ODEProblem(sys, [], (0.0, t_dur), [])
        sol = solve(prob, Vern7())

        sol(t_dur; idxs=[:str1₊inh1₊V,
                         :gpi1₊inh1₊V,
                         # :gpi1₊inh2₊V,
                         # :gpi1₊inh3₊V,
                         ])
    end
    [sol1 sol2 (sol1 .- sol2)]
    # sol
end

end

module GDY
using SynapticBlox, OrdinaryDiffEq, Test
using SynapticBlox: getp
using Random: seed!

function test(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    g = Neurograph()
    seed!(1234)
    @named str1 = Striatum(N_inhib=1)
    @named gpi1 = GPi(N_inhib=1, I_bg=zeros(3))
    add_edge!(g, str1 => gpi1, weight = 0.0, density = 1.5) #str1->gpi1
    
    @named sys = graphsystem_from_graph(g; t_block=100, N_t_block=8)
    prob = ODEProblem(sys, [], (0.0, t_dur), [])
    sol = solve(prob, Vern7())

    sol(t_dur; idxs=[:str1₊inh1₊V,
                     :gpi1₊inh1₊V,
                      # :gpi1₊inh2₊V,
                      # :gpi1₊inh3₊V,
                      ])
     
    sol
end

function test2(t_dur)
    # t_dur = 160 #180.0 + √(eps(180.0))
    
    sol1 = let g = Neurograph()
        seed!(1234)
        @named str1 = Striatum(N_inhib=1)
        @named gpi1 = GPi(N_inhib=1, I_bg=zeros(3))
        add_edge!(g, str1 => gpi1, weight = 4, density = 1.5) #str1->gpi1
        
        @named sys = graphsystem_from_graph(g; t_block=100, N_t_block=8)
        prob = ODEProblem(sys, [], (0.0, t_dur), [])
        sol = solve(prob, Vern7())

        @info "" getp(sol, :str1₊inh1₊G_syn)(sol)
        
        sol(t_dur; idxs=[:str1₊inh1₊V,
                         :gpi1₊inh1₊V,
                         # :gpi1₊inh2₊V,
                         # :gpi1₊inh3₊V,
                         ])
    end
    sol2 = let g = Neurograph()
        seed!(1234)
        @named str1 = Striatum(N_inhib=1, G_syn_inhib=100.0)
        @named gpi1 = GPi(N_inhib=1, I_bg=zeros(3))
        # add_edge!(g, str1 => gpi1, weight = 4, density = 1.5) #str1->gpi1
        n1=str1.inhibs[1]
       
        #SynapticBlox.hypergeometric_connections!(g, str1.inhibs, gpi1.inhibs, :str1, :gpi1; density=1, weight=4.0)
        
        add_edge!(g, str1.inhibs[1], gpi1.inhibs[1], weight=4)
        
        @named sys = graphsystem_from_graph(g; t_block=100, N_t_block=8)
        prob = ODEProblem(sys, [], (0.0, t_dur), [])
        sol = solve(prob, Vern7())
        @info "" n1.G_syn SynapticBlox.to_subsystem(n1).G_syn getp(sol, :str1₊inh1₊G_syn)(sol)

        
        sol(t_dur; idxs=[:str1₊inh1₊V,
                         :gpi1₊inh1₊V,
                         # :gpi1₊inh2₊V,
                         # :gpi1₊inh3₊V,
                         ])
    end
    [sol1 sol2 (sol1 .- sol2)]
    # sol
end

end 

test(t_dur) = @test GDY.test(t_dur) ≈ MTK.test(t_dur) rtol=1e-6
comp(t_dur) = let mtk=MTK.test(t_dur), gdy=GDY.test(t_dur)
    [gdy mtk (gdy-mtk)]
end

using Plots
function plt(t_dur)
    gdy=GDY.test(t_dur)
    mtk=MTK.test(t_dur)
    plot( gdy; idxs=[#:gpi1₊inh1₊V,
                     :str1₊inh1_synapse₊G
                     #:gpi1₊inh1₊V, # :n2₊V, :n3₊V
                     ])
    plot!(mtk; idxs=[#:gpi1₊inh1₊V,
                     :str1₊inh1₊G
                     #:gpi1₊inh1₊V, # :n2₊V, :n3₊V
                     ])
end

end



module Test5
using Test

module MTK
using Neuroblox, OrdinaryDiffEq, Test
using Random: seed!

function test1(t_stimulus, t_pause)
    g = MetaDiGraph()
    seed!(1234)
    namespace = :g
    @named n1 = HHNeuronExciBlox(I_bg=0.0, freq=0.0)
    @named n2 = HHNeuronExciBlox(I_bg=0.0, freq=0.0)
    @named n3 = HHNeuronExciBlox(I_bg=0.0, freq=0.0)

    fn = joinpath(@__DIR__(), "stimuli_set.csv")
    @named stim = ImageStimulus(fn; t_stimulus=600, t_pause=1000, namespace) 
	trial_dur = stim.t_stimulus + stim.t_pause
    @show size(stim.IMG)
    
    stim.current_pixel = 12
    add_edge!(g, stim => n1; weight=14.0)
    add_edge!(g, stim => n2; weight=14.0)
    add_edge!(g, stim => n3; weight=14.0)
    
    #add_edge!(g, n3 => n4; weight=1.0)
    # add_blox!.((g,), [n1, n2, n3])
    #add_blox!.((g,), [n4])
    
    @named sys = system_from_graph(g; )
    prob = ODEProblem(sys, [], (0.0, trial_dur), [])
    sol = solve(prob, Tsit5())
end

end

module GDY
using SynapticBlox, OrdinaryDiffEq, Test
using Random: seed!
function test1(t_stimulus, t_pause)
    g = Neurograph()
    seed!(1234)
    @named n1 = HHExci(I_bg=0.0)
    @named n2 = HHExci(I_bg=0.0)
    @named n3 = HHExci(I_bg=0.0)

    fn = joinpath(@__DIR__(), "stimuli_set.csv")
    @named stim = ImageStimulus(fn; t_stimulus=600, t_pause=1000) 
	trial_dur = stim.t_stimulus + stim.t_pause
    @show size(stim.IMG)
    
    add_edge!(g, stim => n1; weight=14.0, current_pixel=12)
    add_edge!(g, stim => n2; weight=14.0, current_pixel=13)
    #add_vertex!(g, n2)
    add_edge!(g, stim => n3; weight=14.0, current_pixel=14)

    @info "" findfirst(x -> x != 0, stim.current_image)
    
    # add_edge!(g, n1 => n2; weight=1.0)
    # add_edge!(g, n2 => n3; weight=0.5)
    # add_edge!(g, n3 => n1; weight=1.5)
    # add_edge!(g, n3 => n4; weight=1.0)
    # add_vertex!.((g,), [n1, n2, n3])
    # add_vertex!.((g,), [n4])
    
    sys = graphsystem_from_graph(g;)
    prob = ODEProblem(sys, [], (0.0, trial_dur), [])
    sol = solve(prob, Tsit5())

    
end

end 

comp(t_stimulus, t_pause) = let solgdy=GDY.test1(t_stimulus, t_pause), solmtk=MTK.test1(t_stimulus, t_pause)
    t_dur = t_stimulus + t_pause
    gdy = solgdy(t_dur; idxs=[:n1₊V, :n2₊V, :n3₊V,  ])
    mtk = solmtk(t_dur; idxs=[:n1₊V, :n2₊V, :n3₊V,  ])
    
    [gdy mtk (gdy-mtk)]
end
test1(t_dur) = @test GDY.test1(t_dur) ≈ MTK.test1(t_dur) rtol=1e-6

using Plots
function plt(t_dur)
    gdy=GDY.test1(t_dur)
    mtk=MTK.test1(t_dur)
    plot(gdy; idxs=[
                    ])
    plot!(mtk; idxs=[
                     ])
end
end

# ======================================================================
# ======================================================================
