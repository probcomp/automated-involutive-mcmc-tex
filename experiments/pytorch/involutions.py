import torch

##########################
# generic implementation #
##########################

continuous = "continuous"
discrete = "discrete"

def write_cont(trace, addr, value):
    trace[addr] = (value, continuous)

def write_disc(trace, addr, value):
    trace[addr] = (value, discrete)

def involution_with_jacobian_det(f, input_trace):
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
    return (output_trace, logabsdet)

def strip_type_labels_for_logpdf(input_trace):
    output_trace = {}
    for (var, (val, label)) in input_trace.items():
        output_trace[var] = val
    return output_trace

def involution_mcmc_step(p, q, f, input_trace):
    prev_log_score = p(strip_type_labels_for_logpdf(input_trace))
    (output_trace, logabsdet) = involution_with_jacobian_det(f, input_trace)
    new_log_score = p(strip_type_labels_for_logpdf(output_trace))
    alpha = new_log_score - prev_log_score + logabsdet
    if torch.distributions.bernoulli.Bernoulli(min(1, torch.exp(alpha))).sample():
        return (output_trace, True)
    else:
        return (input_trace, False)

###########
# example #
###########

pi = 3.1415927410125732

def p(trace):
    logpdf = torch.tensor(0.0)
    dist = torch.distributions.bernoulli.Bernoulli(0.5)
    logpdf += dist.log_prob(trace["polar"])
    if trace["polar"]:
        dist = torch.distributions.gamma.Gamma(1.0, 1.0)
        logpdf += dist.log_prob(trace["r"])
        dist = torch.distributions.uniform.Uniform(-pi/2, pi/2)
        logpdf += dist.log_prob(trace["theta"])
    else:
        dist = torch.distributions.normal.Normal(0.0, 1.0)
        logpdf += dist.log_prob(trace["x"])
        dist = torch.distributions.normal.Normal(0.0, 1.0)
        logpdf += dist.log_prob(trace["y"])
    return logpdf

def q(trace, model_trace):
    logpdf = torch.tensor(0.0)
    return logpdf

def polar_to_cartesian(r, theta):
    x = torch.cos(r) * theta
    y = torch.sin(r) * theta
    return (x, y)

def cartesian_to_polar(x, y):
    theta = torch.atan2(y, x)
    y = torch.sqrt(x * x + y * y)
    return (theta, y)

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

trace = {
        "polar" : (True, discrete),
        "r": (1.2, continuous),
        "theta" : (0.12, continuous)
}

for it in range(1, 100):
    (trace, acc) = involution_mcmc_step(p, q, f, trace)
    print(trace, acc)
