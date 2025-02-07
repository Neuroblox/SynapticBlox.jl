
let syss = [
    HHExci(name=:ne) => [GABA_A_Synapse(name=:gas),
                         GABA_B_Synapse(name=:gbs),
                         Glu_AMPA_Synapse(name=:gams),
                         Glu_NMDA_Synapse(name=:gnms)]
    HHInhi(name=:ni) => [GABA_A_Synapse(name=:gas),
                         GABA_B_Synapse(name=:gbs),
                         Glu_AMPA_Synapse(name=:gams),
                         Glu_NMDA_Synapse(name=:gnms)]
    GABA_A_Synapse(name=:gas) => [HHInhi(name=:ni)]
    GABA_B_Synapse(name=:gbs) => [HHInhi(name=:ni)]
    Glu_AMPA_Synapse(name=:gams) => [HHExci(name=:ne)]
    Glu_NMDA_Synapse(name=:gnms) => [HHExci(name=:ne)]
    ]
    for (sys_src, _) ∈ syss
        define_neuron(sys_src; mod=@__MODULE__())
    end
    
    for (sys_src, sys_dsts) ∈ syss
        for sys_dst ∈ sys_dsts
            conn = Connector(sys_src, sys_dst; connection_rule="basic", weight=1.0)
            invokelatest(define_basic_connection, conn, sys_src, sys_dst; mod=@__MODULE__())
        end
    end
end
