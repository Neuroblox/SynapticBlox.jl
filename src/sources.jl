struct ImageStimulus <: StimulusBlox
    name::Symbol
    current_image::Vector{Float64}
    IMG::Matrix{Float64} # Matrix[pixels X stimuli]
    category::Vector{Float64}
    t_stimulus::Float64
    t_pause::Float64
    N_pixels::Int
    N_stimuli::Int

    function ImageStimulus(data::CSV.File; name, t_stimulus, t_pause)

        N_pixels = count(x -> x != :category, propertynames(data))
        N_stimuli = length(data)

        # Stack the columns into a matrix and then transpose them
        IMG = (data[col] for col in propertynames(data) if col != :category) |> stack |> transpose

        # Append a column of zeros at the end of data so that indexing can work
        # on the final simulation time step when the index will be `nrow(data)+1`.
        IMG = [IMG zeros(N_pixels)]
        
        category = data[:category]
        current_image = IMG[:, 1]
        
        new(name,  current_image, IMG, category, t_stimulus, t_pause, N_pixels, N_stimuli)
    end
end
function ImageStimulus(file::String; name,  t_stimulus, t_pause)
    @assert last(split(file, '.')) == "csv" "Image file must be a CSV file."
    data = CSV.File(file)
    ImageStimulus(data; name,  t_stimulus, t_pause)
end

function to_subsystem(s::ImageStimulus)
    states = SubsystemStates{ImageStimulus}()
    (; name, current_image, IMG, category, t_stimulus, t_pause, N_pixels, N_stimuli) = s
    params = SubsystemParams{ImageStimulus}(; name, current_image, IMG, category, t_stimulus, t_pause, N_pixels, N_stimuli)
    Subsystem(states, params)
end

GraphDynamics.initialize_input(s::Subsystem{ImageStimulus}) = (;) # ImageStimulus has no inputs!
function GraphDynamics.apply_subsystem_differential!(_, s::Subsystem{ImageStimulus}, jcn, t)
    nothing
end

GraphDynamics.has_discrete_events(::Type{ImageStimulus}) = true
GraphDynamics.event_times((;t_stimulus)::Subsystem{ImageStimulus}) = t_stimulus
GraphDynamics.discrete_event_condition(s::Subsystem{ImageStimulus}, t, _) = s.t_stimulus == t
function GraphDynamics.apply_discrete_event!(integrator, sview, pview, s::Subsystem{ImageStimulus}, _)
    # zero out the current image
    s.current_image .= 0.0
    nothing
end
