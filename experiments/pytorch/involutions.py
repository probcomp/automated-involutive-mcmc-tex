import torch

##########################
# generic implementation #
##########################

CONTINUOUS = "CONTINUOUS"
DISCRETE = "DISCRETE"

InvolutionRunnerState = collections.namedtuple("InvolutionRunnerState",
    [   "input_model_trace", "input_aux_trace",
        "output_model_trace", "output_aux_trace",
        "input_grad_tensors"
    ])

def read_model_cont(state, addr):
    val = torch.tensor(state.input_model_trace.[addr], requires_grad=True)
    state.input_grad_tensors.append(val.grad)
    return val

def read_model_disc(state, addr):
    return state.input_model_trace[addr]

def write_model_cont(state, addr, value):
    state.output_model_trace.trace[addr] = (value, CONTINUOUS)

def write_model_disc(state, addr, value):
    state.output_model_trace[addr] = (value, DISCRETE)

def read_aux_cont(state, addr):
    val = torch.tensor(state.input_aux_trace.[addr], requires_grad=True)
    state.input_grad_tensors.append(val.grad)
    return val

def read_aux_disc(state, addr):
    return state.input_aux_trace[addr]

def write_aux_cont(state, addr, value):
    state.output_aux_trace.trace[addr] = (value, CONTINUOUS)

def write_aux_disc(state, addr, value):
    state.output_aux_trace[addr] = (value, DISCRETE)


from itertools import chain

def involution_with_jacobian_det(f, input_model_trace, input_auxiliary_trace):
    state = InvolutionRunnerState(input_model_trace, input_auxiliary_trace, {}, {}, [])
    f(state)
    grads = []
    for (output_var, (output_val, output_label)) in chain(
            state.output_model_trace.items(), state.output_aux_trace.items())
        if output_label == CONTINUOUS:
            output_val.backward(retain_graph=True)
            grad = []
            for input_grad in state.input_grad_tensors:
                grad.append(input_grad.clone())
                input_grad.zero_()
            grads.append(grad)
    (_, logabsdet) = torch.tensor(grads).slogdet()
    return (output_trace, logabsdet)

def sample(p, args):
    trace = {}
    def trace_choice(dist, addr):
        val = dist.sample()
        trace[addr] = val
        return val
    p(trace_choice, args...)
    return trace

def logpdf(trace, p, args):
    lpdf = torch.tensor(0.0)
    def trace_choice(dist, addr):
        val = trace[addr]
        lpdf += dist.log_prob(val)
        return val
    p(trace_choice, args...)
    return lpdf

def involution_mcmc_step(p, q, f, input_model_trace):

    # sample from auxiliary program
    input_auxiliary_trace = sample(q, input_model_trace)

    # run involution
    (output_model_trace, output_auxiliary_trace, logabsdet) = involution_with_jacobian_det(
        f, input_model_trace, input_auxiliary_trace)

    # compute acceptance probability
    prev_score = logpdf(input_model_trace, p)
    new_score = logpdf(output_model_trace, p)
    fwd_score = logpdf(input_auxiliary_trace, q, input_model_trace)
    bwd_score = logpdf(output_auxiliary_trace, q, output_model_trace)
    prob_accept = min(1, torch.exp(new_score - prev_score + logabsdet + bwd_score - fwd_score))

    # accept or reject
    if Bernoulli(prob_accept).sample():
        return (output_trace, True)
    else:
        return (input_trace, False)


###########
# example #
###########

Bernoulli = torch.distributions.bernoulli.Bernoulli
Gamma = torch.distributions.gamma.Gamma
Normal = torch.distributions.gamma.Normal
Uniform = torch.distributions.uniform.Uniform

pi = 3.1415927410125732

def p(trace):
    if trace(Bernoulli(0.5), "polar")
        trace(Gamma(1.0, 1.0), "r")
        trace(Uniform(-pi/2, pi/2), "theta")
    else:
        trace(Normal(0.0, 1.0), "x")
        trace(Normal(0.0, 1.0), "y")
    return None

def q(trace, model_trace):
    return None

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
        "polar" : (True, DISCRETE),
        "r": (1.2, CONTINUOUS),
        "theta" : (0.12, CONTINUOUS)
}

for it in range(1, 100):
    (trace, acc) = involution_mcmc_step(p, q, f, trace)
    print(trace, acc)
