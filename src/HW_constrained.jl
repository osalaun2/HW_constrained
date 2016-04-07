

# constrained maximization exercises

## portfolio choice problem

module HW_constrained

		using JuMP, NLopt, DataFrames, FactCheck

		export data, table_NLopt, table_JuMP

		function data(a = 0.5)

				n = 3 # number of assets from i=1 to n=3
				p = [1. ; 1. ; 1.] # asset prices
				e = [2. ; 0. ; 0.] # initial endowment of each asset
				# number of states = 4
				z = hcat(ones(16),repeat([0.72,0.92,1.12,1.32],inner=[4],outer=[1]),repeat([0.86,0.96,1.06,1.16],inner=[1],outer=[4])) # payoffs for each asset
				# z1 = [1.0 ; 1.0 ; 1.0 ; 1.0] # payoffs for asset 1
				# z2 = [0.72 ; 0.92 ; 1.12 ; 1.32] # payoffs for asset 2
				# z3 = [0.86 ; 0.96 ; 1.06 ; 1.16] # payoffs for asset 3
				pi = 1/16 # probability of the uniformally distributed s states

		return Dict("a"=>a,"asset number"=>n,"initial endowment"=>e,"asset price"=>p,"asset payoff"=>z,"probability that state s occurs"=>pi)

	end

	######################## MAXIMIZATION JuMP ##############################

	function max_JuMP(a = 0.5)

		let
				d = data(a)
		    m = Model()

		    @defVar(m, c >= 0)
		    @defVar(m, omega[1:d[3]])

				@setNLObjective(m,:Max,-exp(-a * c) + sum{d[pi][s] * (-exp(-a * sum{omega[j] * d[z][s,j],j=1:d[3]})),s=1:d[16]}) # function to be maximized
				@addNLConstraint(m,c + sum{d[p][i]*(omega[i] - d[e][i]),i=1:d[3]} == 0.0) # constraint

		    solve(m)

		    println("c = ", getValue(c), " omega = ", getValue(omega))
		end
	end

	######################## TABLE JuMP ##############################

	function table_JuMP()

				d = DataFrame(a=[0.5;1.0;5.0],
				c = zeros(3),
				omega1=zeros(3), # creates an empty vector of length 3
				omega2=zeros(3), # creates an empty vector of length 3
				omega3=zeros(3), # creates an empty vector of length 3
				fval=zeros(3)) # creates an empty vector of length 3

		 				for i in 1:nrow(d)
		 						xx = max_JuMP(d[i,:a])
					 			d[i,:c] = xx["c"] # creates colum for c values
					 			d[i,:omega1] = xx["omega"][1] # creates colum for omega1 values
					 			d[i,:omega2] = xx["omega"][2] # creates colum for omega2 values
					 			d[i,:omega3] = xx["omega"][3] # creates colum for omega3 values
					 			d[i,:fval] = xx["obj"] # creates colum for fval values
						end
				return d

	end

######################## FUNCTION OBJECT ##############################

	function obj(x::Vector,grad::Vector,data::Dict)
			if length(grad) > 0
	        grad[1] = a * exp(-a * c) # derivative of maximized function with respect to c
	        grad[2] = sum(pi[s]) * a * sum(z[i]) * exp(-a * sum(omega[i] * z[i])) # derivative of maximized with respect to omega
	    end
	    return (-exp(-a * c)) + sum(pi[s] * -exp(-a * sum(omega[i] * z[i])))
	end

	######################## FUNCTION CONSTRAINT ##############################

	function constr(x::Vector,grad::Vector,data::Dict, )
			if length(grad) > 0
		    	grad[1] = 1. # derivative of constraint with respect to c
		    	grad[2] = sum(p[i]) # derivative of constraint with respect to omega
		  end
		  return c + sum(p[i] * (omega[i] - e[i]))
	end

	######################## MAXIMIZATION NLopt ##############################

	function max_NLopt(a=0.5)

		d = data(a)
		opt = NLopt.Opt(:LD_MMA,d[4])
		lower_bounds!(opt,[0;[-Inf for i=1:d[3]]])
		max_objective!(opt, (x,g) -> obj(x,g,d))
		constraint!(opt, (x,g) -> r_constraint(x,g,d))
		ftol_rel!(opt,1e-9)
		(maxf,maxx,ret) = optimize(opt, rand(d[4]))

	end

	######################## TABLE NLopt ##############################

	function table_NLopt()

		d = DataFrame(a=[0.5;1.0;5.0],
		c = zeros(3), # creates an empty vector of length 3
		omega1=zeros(3), # creates an empty vector of length 3
		omega2=zeros(3), # creates an empty vector of length 3
		omega3=zeros(3), # creates an empty vector of length 3
		fval=zeros(3)) # creates an empty vector of length 3

		for i in 1:nrow(d)
					xx = max_NLopt(d[i,:a])
							for j in 2:ncol(d)-1
											d[i,j] = xx[2][j-1]
							end
					d[i,end] = xx[1]
		end
		return d

	end

	# function `f` is for the NLopt interface, i.e.
	# it has 2 arguments `x` and `grad`, where `grad` is
	# modified in place
	# if you want to call `f` with more than those 2 args, you need to
	# specify an anonymous function as in
	# other_arg = 3.3
	# test_finite_diff((x,g)->f(x,g,other_arg), x )
	# this function cycles through all dimensions of `f` and applies
	# the finite differencing to each. it prints some nice output.
	function test_finite_diff(f::Function,x::Vector{Float64},tol=1e-6)

				# get gradient from f
				grad = similar(x)
				y = f(x,grad)
				# get finite difference approx
				fdiff = finite_diff(f,x)
				r = hcat(1:length(x),grad,fdiff,abs(grad-fdiff))
				errors = find(abs(grad-fdiff).>tol)
				if length(errors) >0
								println("elements with errors:")
								println("id    supplied gradient    finite difference    abs diff")
								for i in 1:length(errors)
												@printf("%d    %f3.8    %f3.8    %f1.8\n",r[errors[i],1],r[i,2],r[i,3],r[i,4])
								end
								return (false,errors)
				else
		 						println("no errors.")
								return true
		 		end
	end

	# do this for each dimension of x
	# low-level function doing the actual finite difference
	function finite_diff(f::Function,x::Vector)

				h = sqrt(eps())
		 		fgrad = similar(x)
		 		tgrad = similar(x)
		 		for i in 1:length(x)
		 							step = abs(x[i]) > 1
		 							newx = copy(x)
		 							newx[i] = x[i]+step
		 							fgrad[i] = (f(newx) - f(x))/step
		 		end
		 		return fgrad

	end

	function runAll()
		println("running tests:")
		include("test/runtests.jl")
		println("")
		println("JumP:")
		table_JuMP()
		println("")
		println("NLopt:")
		table_NLopt()
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end
