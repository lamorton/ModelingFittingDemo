using ModelingToolkit, OrdinaryDiffEq, Plots
using CSV, DataFrames, Optim

MT = ModelingToolkit

include("auxilliary.jl")

@parameters t
D = Differential(t)
pars = @parameters(
   m = 0.57,  # mass
   k = 3.4, # spring constant
   b = 0.2, # drag
)

vars = @variables begin
    x(t), # position of mass
    v(t), # velocity of mass
    F(t) # force of spring on mass
end


eqs = [
       F ~ -k * x - b * v,
       m * D(v) ~ F,
       D(x) ~ v, 
    ]

sys = ODESystem(eqs, t; name = :sys)
sys = structural_simplify(sys)

u0 = [ #Default Initial conditions
    v => 10.,
    x => 7.5,
    ]

testdata = CSV.read("testdata.csv", DataFrame)
tspan = (minimum(testdata.time), maximum(testdata.time)) 
prob = ODEProblem(sys, u0, tspan)

guess = [k => 3.5, b=> .2] # Listing a parameter/value pair here allows the optimizer to adjust the parameter, starting with the specified value as the initial guess.
pfixed = [m => 0.57] # Listing a parameter/value pair here over-rides the default value.
ufixed = [] # Override the default value of any fixed initial conditions here.
uguess = [v => 10., x=> 3.0] # Initial conditions that the optimizer is allowed to vary.

#Plot the solution using the initial guess + fixed values for the parameters & initial conditions vs the data
gsol = error_plot(prob, guess, pfixed, uguess, ufixed, testdata);

#Optimize the model & repeat the plot with the best-fit parameters & initial conditions
bestfit = reoptimize(prob, guess, pfixed, uguess, ufixed,testdata)
bestp = bestfit[1][1:length(guess)]
bestu = bestfit[1][length(guess)+1:end]
fsol = error_plot(prob, bestp, pfixed, bestu, ufixed, testdata);

#To re-create the testdata.csv file:
# p = [
#    m => 0.57,  # mass
#    k => 2.1,  # spring constant
#    b => 0.16,
# ]

# u0 = [
#     v => 15,
#     x => 5.0,
#     ]

# tspan = (0, 10) 
# prob = ODEProblem(sys, u0, tspan, p)
# usol = solve(prob, Tsit5(), saveat = range(0,10,step=0.1))
# df = DataFrames.DataFrame(usol)
# CSV.write("testdata.csv", df)
# Then manually change the header of the testdata.csv file to "time, x, v" instead of "timestamp, x(t), v(t)"