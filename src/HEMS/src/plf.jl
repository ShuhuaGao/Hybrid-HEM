#===============================================
Piecewise linearization 
===============================================#


"""
    linearize(f::Function, x_low, x_up, L::Int)
PLF approximation of a function `f` defined on [`x_low`, `x_up`] with `L` segments. 

The x coordinate of each endpoint, and the slope and intercept of each segment are returned.
"""
function linearize(f::Function, x_low, x_up, L::Int)
    @assert x_low < x_up && L > 0
    x = range(x_low, x_up; length=L+1)
    k = zeros(L)
    b = zeros(L)
    for i = 1:L
        k[i] = (f(x[i+1]) - f(x[i])) / (x[i+1] - x[i])
        b[i] = f(x[i]) - k[i] * x[i]
    end
    return x, k, b
end