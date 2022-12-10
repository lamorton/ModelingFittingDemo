# Example of fitting a ModelingToolkit model to data

A common use-case for [ModelingToolkit](https://docs.sciml.ai/ModelingToolkit/dev/) is to fit the model to data, thereby inferring the values of parameter and initial conditions. Often, one would
like to change whether a parameter or initial condition is allowed to 
be adjusted in the optimization process. This is an example of how to 
accomplish this task. It also helps work around the slowness of access to solutions via the `sol[x]` approach, which makes naive loss function
 evaluations too slow for optimization use.
