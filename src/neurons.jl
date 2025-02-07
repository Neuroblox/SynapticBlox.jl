
struct HHExci <: AbstractExciNeuronBlox
    system
    namespace
    default_synapses

	function HHExci(;
        name, 
        namespace=nothing,
        I_bg=0.0,
        freq=0
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)
			[input=true] 
            I_in(t)
            [input=true]
			I_asc(t)
			[input=true]
		end

		ps = @parameters begin
			G_Na = 52 
			G_K  = 20 
			G_L = 0.1 
			E_Na = 55 
			E_K = -90 
			E_L = -60 
			I_bg=I_bg
			freq = freq 
		end

		αₙ(v) = 0.01*(v+34)/(1-exp(-(v+34)/10))
	    βₙ(v) = 0.125*exp(-(v+44)/80)
	    αₘ(v) = 0.1*(v+30)/(1-exp(-(v+30)/10))
	    βₘ(v) = 4*exp(-(v+55)/18)
		αₕ(v) = 0.07*exp(-(v+44)/20)
	    βₕ(v) = 1/(1+exp(-(v+14)/10))
		ϕ = 5
		eqs = [ 
			D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in,
			D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n),
			D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m) ,
			D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
		]
        
		sys = ODESystem(
            eqs, t, sts, ps; 
			name = Symbol(name)
			)
		new(sys, namespace, Dict())
	end
end

struct HHInhi <: AbstractInhNeuronBlox
    system
    namespace
    default_synapses
	function HHInhi(;
        name, 
        namespace = nothing, 
        I_bg=0.0,
        freq=0,
    )
		sts = @variables begin 
			V(t)=-65.00 
			n(t)=0.32 
			m(t)=0.05 
			h(t)=0.59 
			I_syn(t)
			[input=true] 
			I_asc(t)
			[input=true]
			I_in(t)
			[input=true]
		end

		ps = @parameters begin
			G_Na = 52 
			G_K  = 20 
			G_L = 0.1 
			E_Na = 55 
			E_K = -90 
			E_L = -60 
			I_bg=I_bg 
			freq = freq
		end

	   	αₙ(v) = 0.01*(v+38)/(1-exp(-(v+38)/10))
		βₙ(v) = 0.125*exp(-(v+48)/80)
        αₘ(v) = 0.1*(v+33)/(1-exp(-(v+33)/10))
		βₘ(v) = 4*exp(-(v+58)/18)
        αₕ(v) = 0.07*exp(-(v+51)/20)
		βₕ(v) = 1/(1+exp(-(v+21)/10))   	
		ϕ = 5
	 	eqs = [ 
			   D(V)~-G_Na*m^3*h*(V-E_Na)-G_K*n^4*(V-E_K)-G_L*(V-E_L)+I_bg*(sin(t*freq*2*pi/1000)+1)+I_syn+I_asc+I_in,
			   D(n)~ϕ*(αₙ(V)*(1-n)-βₙ(V)*n) ,
			   D(m)~ϕ*(αₘ(V)*(1-m)-βₘ(V)*m) ,
			   D(h)~ϕ*(αₕ(V)*(1-h)-βₕ(V)*h),
		]

        sys = ODESystem(
            eqs, t, sts, ps; 
			name = Symbol(name)
        )
        
		new(sys, namespace, Dict())
	end
end

synapse_dict(n::Union{HHExci, HHInhi}) = getfield(n, :default_synapses)


struct WTABlox <: CompositeBlox
    namespace
    parts
    system
    connector
    function WTABlox(
        ;  name, 
        namespace = nothing,
        N_exci = 5,
        E_syn_exci=0.0,
        E_syn_inhib=-70,
        G_syn_exci=3.0,
        G_syn_inhib=3.0,
        I_bg=zeros(N_exci),
        freq=0.0,
        phase=0.0,
        τ_exci=5,
        τ_inhib=70)
        
        n_inhi = HHInhi(
            name = "inh",
            namespace = namespaced_name(namespace, name)
        )
        s_inhi = get_synapse!(n_inhi, Glu_AMPA_Synapse, (;E_syn = E_syn_inhib, 
                                                         G_syn = G_syn_inhib, 
                                                         τ = τ_inhib))
        # E_syn = E_syn_exci, 
        # G_syn = G_syn_exci, 
        # τ = τ_exci,
        n_excis = map(1:N_exci) do i
            HHExci(
                name = Symbol("exci$i"),
                namespace = namespaced_name(namespace, name), 
                I_bg = (I_bg isa Array) ? I_bg[i] : I_bg*rand(), # behave differently if I_bg is array
                freq = freq,
                phase = phase)
        end

        g = MetaDiGraph()
        add_blox!(g, n_inhi)
        for n_exci in n_excis
            add_blox!(g, n_excis[i])
            add_edge!(g, 1, i+1, :weight, 1.0)
            add_edge!(g, i+1, 1, :weight, 1.0)
        end

        parts = vcat(n_inhi, n_excis, )
        
        bc = connectors_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name, simplify=false) : system_from_parts(parts; name)

        new(namespace, parts, sys, bc)
    end
end
