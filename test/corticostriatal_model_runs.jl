using
    Test,
    OrdinaryDiffEq,
    SynapticBlox,
    GraphDynamics,
    OhMyThreads,
    CSV,
    SymbolicIndexingInterface,
    Random


function smaller_cortiostriatal_learning_run(;time_block_dur = 90.0, ## ms (size of discrete time blocks)
                                             N_trials = 700, ## number of trials
                                             trial_dur = 1000 ## ms
                                             )

    
    # download the stimulus images
    #image_set = joinpath(@__DIR__, "image_example.csv") #stimulus image file
    @time begin
    image_set = joinpath(@__DIR__, "stimuli_set_small.csv")
    #CSV.File(Downloads.download("raw.githubusercontent.com/Neuroblox/NeurobloxDocsHost/refs/heads/main/data/stimuli_set.csv")) ## reading data into DataFrame format

    # define stimulus Blox
    # t_stimulus: how long the stimulus is on (in ms)
    # t_pause : how long the stimulus is off (in ms)
    @named stim = ImageStimulus(image_set;  t_stimulus=trial_dur, t_pause=0);

    # Cortical Bloxs
    @named VAC = CorticalBlox(;  N_l_flic=4, N_exci=5,  density=0.05, weight=1)
    @named AC = CorticalBlox(;  N_l_flic=2, N_exci=5, density=0.05, weight=1)
    # ascending system Blox, modulating frequency set to 16 Hz
    @named ASC1 = NGNMM_theta(;  Cₑ=2*26,Cᵢ=1*26, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑᵢ=0.6*26, kᵢₑ=0.6*26)

    # additional Striatum Bloxs
    @named STR1 = Striatum(;  N_inhib=5)
    @named STR2 = Striatum(;  N_inhib=5)

    @named tan_pop1 = TAN(κ=10; )
    @named tan_pop2 = TAN(κ=10; )

    @named SNcb = SNc(κ_DA=1; )

    # action selection Blox, necessary for making a choice
    @named AS = GreedyPolicy(;  t_decision=2*time_block_dur)

    # learning rules
    hebbian_mod = HebbianModulationPlasticity(K=0.06, decay=0.01, α=2.5, θₘ=1, modulator=SNcb, t_pre=trial_dur, t_post=trial_dur, t_mod=time_block_dur)
    hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=7, t_pre=trial_dur, t_post=trial_dur)

    g = Neurograph()

    add_edge!(g, stim => VAC, weight=14)
    add_edge!(g, ASC1 => VAC, weight=44)
    add_edge!(g, ASC1 => AC, weight=44)
    add_edge!(g, VAC => AC, weight=3, density=0.1, learning_rule = hebbian_cort)
    add_edge!(g, AC => STR1, weight = 0.075, density =  0.04, learning_rule =  hebbian_mod)
    add_edge!(g, AC => STR2, weight =  0.075, density =  0.04, learning_rule =  hebbian_mod)
    add_edge!(g, tan_pop1 => STR1, weight = 1, t_event = time_block_dur)
    add_edge!(g, tan_pop2 => STR2, weight = 1, t_event = time_block_dur)
    add_edge!(g, STR1 => tan_pop1, weight = 1)
    add_edge!(g, STR2 => tan_pop1, weight = 1)
    add_edge!(g, STR1 => tan_pop2, weight = 1)
    add_edge!(g, STR2 => tan_pop2, weight = 1)
    add_edge!(g, STR1 => STR2, weight = 1, t_event = 2*time_block_dur)
    add_edge!(g, STR2 => STR1, weight = 1, t_event = 2*time_block_dur)
    add_edge!(g, STR1 => SNcb, weight = 1)
    add_edge!(g, STR2 => SNcb, weight = 1)
    # action selection connections
    add_edge!(g, STR1 => AS);
    add_edge!(g, STR2 => AS);

    @named env = ClassificationEnvironment(stim, N_trials)
    
    N_t_block=round(Int, env.t_trial/time_block_dur, RoundUp)
        @named agent = Agent(g; t_block = time_block_dur, N_t_block); ## define agent
        print("Construction:  "); 
    end
    #trace = run_experiment!(agent, env; t_warmup=200.0, alg=Vern7(), verbose=true, save_everystep=false)
    agent.problem
end


function big_cortiocostriatal_learning_run(; sta_thal=true, scheduler=SerialScheduler(),
                                           time_block_dur = 90, # ms (size of discrete time blocks)
                                           N_trials = 700, #number of trials
                                           )
    @time begin
	    fn = joinpath(@__DIR__, "stimuli_set_big.csv") #stimulus image file
        
	    #define the circuit blox
        @named stim = ImageStimulus(fn; t_stimulus=600, t_pause=1000) 
	    trial_dur = stim.t_stimulus + stim.t_pause
        @named LC = NGNMM_theta(;Cₑ=2*26,Cᵢ=1*26, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/26, alpha_invₑᵢ=0.8/26, alpha_invᵢₑ=10.0/26, alpha_invᵢᵢ=0.8/26, kₑₑ=0.0*26, kₑᵢ=0.6*26, kᵢₑ=0.6*26, kᵢᵢ=0*26) 
	    
        @named ITN = NGNMM_theta(;Cₑ=2*36,Cᵢ=1*36, Δₑ=0.5, Δᵢ=0.5, η_0ₑ=10.0, η_0ᵢ=0.0, v_synₑₑ=10.0, v_synₑᵢ=-10.0, v_synᵢₑ=10.0, v_synᵢᵢ=-10.0, alpha_invₑₑ=10.0/36, alpha_invₑᵢ=0.8/36, alpha_invᵢₑ=10.0/36, alpha_invᵢᵢ=0.8/36, kₑₑ=0.0*36, kₑᵢ=0.6*36, kᵢₑ=0.6*36, kᵢᵢ=0*36) 

        @named VC = CorticalBlox(N_l_flic=45, N_exci=5,  density=0.01, weight=1,I_bg_ar=0) 
        @named PFC = CorticalBlox(N_l_flic=20, N_exci=5, density=0.01, weight=1,I_bg_ar=0) 
        @named STR1 = Striatum(N_inhib=25) 
	    @named STR2 = Striatum(N_inhib=25)
	    
	    @named tan_nrn = HHExci()
	    
	    @named gpi1 = GPi(N_inhib=25) 
	    @named gpi2 = GPi(N_inhib=25) 
	    
	    @named gpe1 = GPe(N_inhib=15) 
	    @named gpe2 = GPe(N_inhib=15) 
	    
	    @named STN1 = STN(N_exci=15,I_bg=3*ones(25)) 
        @named STN2 = STN(N_exci=15,I_bg=3*ones(25)) 
	    
	    @named Thal1 = Thalamus(N_exci=25) 
	    @named Thal2 = Thalamus(N_exci=25)
        
        @named tan_pop1 = TAN() 
        @named tan_pop2 = TAN() 
	    
	    @named AS = GreedyPolicy(t_decision=180.5) 
        @named SNcb = SNc() 
	    

	    #define learning rules
	    hebbian_mod = HebbianModulationPlasticity(K=0.04, decay=0.01, α=2.5, θₘ=1.0, modulator=SNcb, t_pre=1600-eps(), t_post=1600-eps(), t_mod=90.0)
	    
        hebbian_cort = HebbianPlasticity(K=5e-4, W_lim=5, t_pre=1600-eps(), t_post=1600-eps())
	    
	    hebbian_thal_cort = HebbianPlasticity(K=1.7e-5, W_lim=6, t_pre=1600-eps(), t_post=1600-eps())


	    g = Neurograph()
        add_edge!(g, LC, VC; weight = 44) #LC->VC
	    add_edge!(g, LC, PFC; weight = 44) #LC->pfc
	    add_edge!(g, ITN, tan_nrn; weight = 100) #ITN->tan
	    add_edge!(g, VC, PFC; weight = 1, density = 0.08, learning_rule = hebbian_cort) #VC->pfc
	    add_edge!(g, PFC, STR1; weight = 0.075, density = 0.04, learning_rule = hebbian_mod) #pfc->str1
	    add_edge!(g, PFC, STR2; weight = 0.075, density = 0.04, learning_rule = hebbian_mod) #pfc->str2
	    add_edge!(g, tan_nrn, STR1; weight = 0.17) #tan->str1
	    add_edge!(g, tan_nrn, STR2; weight = 0.17) #tan->str2
	    add_edge!(g, STR1, gpi1, weight = 4, density = 0.04) #str1->gpi1
	    add_edge!(g, STR2, gpi2; weight = 4, density = 0.04) #str2->gpi2
	    add_edge!(g, gpi1, Thal1; weight = 0.16, density = 0.04) #gpi1->thal1
	    add_edge!(g, gpi2, Thal2; weight = 0.16, density = 0.04) #gpi2->thal2
        add_edge!(g, Thal1, PFC; weight = 0.2, density = 0.32, learning_rule = hebbian_thal_cort, sta=sta_thal) #thal1->pfc
	    add_edge!(g, Thal2, PFC; weight = 0.2, density = 0.32, learning_rule = hebbian_thal_cort, sta=sta_thal) #thal2->pfc
        # @warn "Disabling STA connections!"
	    add_edge!(g, STR1, gpe1; weight = 4, density = 0.04)   #str1->gpe1
	    add_edge!(g, STR2, gpe2; weight = 4.0, density = 0.04) #str2->gpe2
	    add_edge!(g, gpe1, gpi1; weight = 0.2, density = 0.04) #gpe1->gpi1
	    add_edge!(g, gpe2, gpi2; weight = 0.2, density = 0.04) #gpe2->gpi2
	    add_edge!(g, gpe1, STN1; weight = 3.5, density = 0.04) #gpe1->stn1
	    add_edge!(g, gpe2, STN2; weight = 3.5, density = 0.04) #gpe2->stn2
	    add_edge!(g, STN1, gpi1; weight = 0.1, density = 0.04) #stn1->gpi1
	    add_edge!(g, STN2, gpi2; weight = 0.1, density = 0.04) #stn2->gpi2
	    add_edge!(g, stim, VC; weight = 14) #stim->VC
	    add_edge!(g, tan_pop1, STR1; weight = 1, t_event = 90.0) #TAN pop1 -> str1
	    add_edge!(g, tan_pop2, STR2; weight = 1, t_event = 90.0) #TAN pop2 -> str2
	    add_edge!(g, STR1, tan_pop1; weight = 1) #str1 -> TAN pop1 
	    add_edge!(g, STR2, tan_pop1; weight = 1) #str2 -> TAN pop1
	    add_edge!(g, STR1, tan_pop2; weight = 1) #str1 -> TAN pop2 
	    add_edge!(g, STR2, tan_pop2; weight = 1) #str2 -> TAN pop2
	    add_edge!(g, STR1, STR2; weight = 1, t_event = 181.0) #str1 -> str2
	    add_edge!(g, STR2, STR1; weight = 1, t_event = 181.0) #str2 -> str1
	    add_edge!(g, STR1, AS)# str1->AS
	    add_edge!(g, STR2, AS)# str2->AS
	    add_edge!(g, STR1, SNcb; weight = 1.0) # str1->Snc
        add_edge!(g, STR2, SNcb; weight = 1.0)  # str2->Snc


        @named env = ClassificationEnvironment(stim, N_trials)
        t_block = 90
        N_t_block=round(Int, env.t_trial/t_block, RoundUp)
        @named agent = Agent(g; t_block, N_t_block, scheduler);
        print("Construction:  "); 
    end 
    #agent
    @time run_experiment!(agent, env; alg=Vern7(), t_warmup=800.0, save_everystep=false, verbose=true)
end

