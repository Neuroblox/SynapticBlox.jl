

include("test_suite.jl")

@testset "HH Neuron excitatory & inhibitory network" begin 
    simple_synaptic_blox_tests()
    synaptic_blox_tests()
end

@testset "Composite structures" begin
    @testset "L-FLIC tests" begin
        lflic_tests()
    end
    @testset "CorticalBlox tests" begin
        cortical_tests()
    end
end

include("./corticostriatal_model_runs.jl")

@testset "Cortical Striatal model" begin
    trace = smaller_cortiostriatal_learning_run(N_trials=500)[100:end]
    accuracy = sum(row -> row.iscorrect, trace)/length(trace)
    @test accuracy >= 0.7
end
