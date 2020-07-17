import torch
from collections import namedtuple
from itertools import chain


##############################################
# minimal probabilistic programming language #
##############################################

def sample(p, *args):
    trace = {}
    def trace_choice(dist, addr):
        val = dist.sample()
        trace[addr] = val
        return val
    p(trace_choice, *args)
    return trace

def logpdf(trace, p, *args):
    lpdf = torch.tensor(0.0)
    def trace_choice(dist, addr):
        nonlocal lpdf
        val = trace[addr]
        lpdf += dist.log_prob(val)
        return val
    p(trace_choice, *args)
    return lpdf


###############################################
# minimal differentiable programming language #
###############################################

InvolutionRunnerState = namedtuple("InvolutionRunnerState",
    [   "input_model_trace", "input_aux_trace",
        "output_model_trace", "output_aux_trace",
        "input_cont_tensors", "output_cont_tensors"
    ])

# model and address identifiers
MODEL = "model"
AUX = "aux"

def check_which(which):
    if which != MODEL and which != AUX:
        raise Exception("address argument must be (MODEL, *) or (AUX, *)")

# user-provided type information for reads and writes
CONTINUOUS = "continuous"
DISCRETE = "discrete"

def check_type_label(type_label):
    if type_label != CONTINUOUS and type_label != DISCRETE:
        raise Exception("type label argument must be CONTINUOUS or DISCRETE")

def read(state, addr, type_label):
    (which, addr) = addr
    check_which(which)
    check_type_label(type_label)
    if type_label == CONTINUOUS:
        if which == MODEL:
            val = torch.tensor(state.input_model_trace[addr], requires_grad=True)
        else:
            val = torch.tensor(state.input_aux_trace[addr], requires_grad=True)
        state.input_cont_tensors.append(val)
        return val
    else:
        if which == MODEL:
            return state.input_model_trace[addr]
        else:
            return state.input_aux_trace[addr]

def write(state, addr, val, type_label):
    (which, addr) = addr
    check_which(which)
    check_type_label(type_label)
    if type_label == CONTINUOUS:
        if which == MODEL:
            state.output_model_trace[addr] = val 
        else:
            state.output_aux_trace[addr] = val
        state.output_cont_tensors.append(val)
    else:
        if which == MODEL:
            state.output_model_trace[addr] = val
        else:
            state.output_aux_trace[addr] = val

#def copy_model_to_model(state, addr1, addr2):
#def copy_model_to_aux(state, addr1, addr2):
#def copy_aux_to_model(state, addr1, addr2):
#def copy_aux_to_aux(state, addr1, addr2):

def involution_with_jacobian_det(f, input_model_trace, input_auxiliary_trace):
    state = InvolutionRunnerState(input_model_trace, input_auxiliary_trace, {}, {}, [], [])
    f(state)
    grads = []
    for output_cont_tensor in state.output_cont_tensors:
        output_cont_tensor.backward(retain_graph=True)
        grad = []
        for input_cont_tensor in state.input_cont_tensors:
            grad.append(input_cont_tensor.grad.clone())
            input_cont_tensor.grad.zero_()
        grads.append(grad)
    (_, logabsdet) = torch.tensor(grads).slogdet()
    return (state.output_model_trace, state.output_aux_trace, logabsdet)


###################
# involutive MCMC #
###################

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
        return (output_model_trace, True)
    else:
        return (input_model_trace, False)


###########
# example #
###########

Bernoulli = torch.distributions.bernoulli.Bernoulli
Gamma = torch.distributions.gamma.Gamma
Normal = torch.distributions.normal.Normal
Uniform = torch.distributions.uniform.Uniform

pi = 3.1415927410125732

def p(trace):
    if trace(Bernoulli(0.5), "polar"):
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

def f(state):
    polar = read(state, (MODEL, "polar"), DISCRETE)
    if polar:
        r = read(state, (MODEL, "r"), CONTINUOUS)
        theta = read(state, (MODEL, "theta"), CONTINUOUS)
        (x, y) = polar_to_cartesian(r, theta)
        write(state, (MODEL, "x"), x, CONTINUOUS)
        write(state, (MODEL, "y"), y, CONTINUOUS)
    else:
        x = read(state, (MODEL, "x"), CONTINUOUS)
        y = read(state, (MODEL, "y"), CONTINUOUS)
        (r, theta) = cartesian_to_polar(x, y)
        write(state, (MODEL, "r"), r, CONTINUOUS)
        write(state, (MODEL, "theta"), theta, CONTINUOUS)
    write(state, (MODEL, "polar"), not polar, DISCRETE)

trace = {
        "polar" : True,
        "r": 1.2,
        "theta" : 0.12
}

for it in range(1, 100):
    (trace, acc) = involution_mcmc_step(p, q, f, trace)
    print(trace, acc)
