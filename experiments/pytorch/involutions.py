import torch

continuous = "continuous"
discrete = "discrete"

def polar_to_cartesian(r, theta):
    x = torch.cos(r) * theta
    y = torch.sin(r) * theta
    return (x, y)

def cartesian_to_polar(x, y):
    theta = torch.atan2(y, x)
    y = torch.sqrt(x * x + y * y)
    return (theta, y)

def write_cont(trace, addr, value):
    trace[addr] = (value, continuous)

def write_disc(trace, addr, value):
    trace[addr] = (value, discrete)

def f(input_trace):
    output_trace = {}
    if input_trace["polar"]:
        (x, y) = polar_to_cartesian(input_trace["r"], input_trace["theta"])
        write_cont(output_trace, "x", x)
        write_cont(output_trace, "y", y)
    else:
        (r, theta) = cartesian_to_polar(input_trace["x"], input_trace["y"])
        write_cont(output_trace, "r", r)
        write_cont(output_trace, "theta", theta)
    write_disc(output_trace, "polar", not input_trace["polar"])
    return output_trace

def involution_with_grad(f, input_trace):
    stripped_input_trace = {}
    for var in input_trace.keys():
        if input_trace[var][1] == continuous:
            stripped_input_trace[var] = torch.tensor(input_trace[var][0], requires_grad=True)
        else:
            stripped_input_trace[var] = input_trace[var][0]
    output_trace = f(stripped_input_trace)
    grads = []
    for (output_var, (output_val, output_label)) in output_trace.items():
        if output_label == continuous:
            output_val.backward(retain_graph=True)
            grad = []
            for (input_var, input_val) in stripped_input_trace.items():
                if input_trace[input_var][1] == continuous:
                    grad.append(input_val.grad.clone())
            for (input_var, input_val) in stripped_input_trace.items():
                if input_trace[input_var][1] == continuous:
                    input_val.grad.zero_()
            grads.append(grad)
    (_, logabsdet) = torch.tensor(grads).slogdet()
    print(logabsdet)
    
input_trace = {
        "polar" : (True, discrete),
        "r": (1.2, continuous),
        "theta" : (0.12, continuous)
}

print(involution_with_grad(f, input_trace))
