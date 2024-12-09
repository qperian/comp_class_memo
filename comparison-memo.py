from memo import memo
import jax
import jax.numpy as np
from scipy.stats import norm
from functools import partial


N = [0, 1, 2, 3]  # number of nice people
U = [0, 1, 2]     # utterance: {none, some, all} of the people are nice

Utterances = {
    "positive": [0,1,2],#["positive_adjective", "positive_sub", "positive_super"],
    "negative": [3,4,5]#["negative_adjective", "negative_sub", "negative_super"]
}

@partial(jax.jit, static_argnums=(0,1))
def meaning(utterance, state, threshold):
    '''
    utterance_form = "positive" or "negative" - can't pass something that's not an array into JAX ahhh
    states = array of possible states
    threshold = threshold value to apply scalar adjective

    returns an array of boolean meaning values for all states
    '''
    
    utterance_form = "positive"
    # utterance_form = get_form(utterance)
    if utterance_form == "positive":
        return state > threshold
    elif utterance_form == "negative":
        return state < threshold
    else:
        raise ValueError("incorrect utterance format -- form is not positive or negative")


# states = np.array([13, 14, 15, 16, 17])
# print(meaning("positive", states, 16))


expt1DataFile = "../data/class-elicitation-full-trials.csv"
expt2DataFile = "../data/vague-prior-elicitation-1-trials.csv"


# @jax.jit
# def meaning(n, u):  # (none)  (some)  (all)
#     return np.array([ n == 0, n > 0, n == 3 ])[u]

# define space of possible states


def state_gaussian_params(comp_class, sub_mu, sub_sigma):
    '''
    determine which gaussian parameters to use for the states depending on whether it's the 
    subordinate or superordinate comparison class
    '''
    if comp_class == 0:
        return (sub_mu, sub_sigma)
    elif comp_class == 1:
        return (0, 1)
    else:
        raise ValueError("comparison class is not sub or super")



# @jax.jit
@partial(jax.jit, static_argnums=(0, 2))
def threshold_bins(utterance_form, States, bin_param):
    '''
    determine the set of threshold possibilities based on binned states  
    binning based on the form of the utterance ("positive" or "negative")
    returns an array of Thresholds
    '''
    if utterance_form == "positive":
        threshold_func = lambda x: x - (1/(bin_param*2))
    elif utterance_form == "negative":
        threshold_func = lambda x: x + (1/(bin_param*2))
    else:
        raise ValueError("incorrect utterance format -- form is not positive or negative")
    print("Hiya there")
    mapping_func = jax.vmap(threshold_func, 0, 0)       # assumes that the inputted array for States is one-directional
    print("Seeeya later pal")
    return mapping_func(States)

# States = np.array([1, 2, 3, 4])
# Thresholds = threshold_bins("negative", States, 3)
# print(type(Thresholds))
# print(Thresholds)

Comp_classes = [0,1]
bin_param = 3
utterance_form = "positive"
# define thresholds globally -- nope
#Thresholds = ...
stateParams = {'mu': 0, 'sigma': 1}
States = np.array(list(map(round, range((stateParams['mu'] - 3*stateParams['sigma'])*bin_param, 
                          (stateParams['mu'] + 3*stateParams['sigma'])*bin_param, 
                          stateParams['sigma']))))/bin_param
print(States.shape)
# can't have continuous distribution for memo, need to discretize states/normal distribution for states
# bubbles = Utterances[utterance_form]
Thresholds = threshold_bins(utterance_form, States, bin_param)
# print(Thresholds.shape)
print("threshole", Thresholds)
Utterances = np.array(Utterances["positive"])
print("utterances", Utterances)

print(norm.pdf(0, 0, 1))

# @partial(memo, debug_trace=True, debug_print_compiled=True)
@memo
def comparison[utterance: Utterances, comp_class: Comp_classes](sub_mu, sub_sigma):
    cast: [speaker, listener]
    # listener: knows(comp_class)
    listener: thinks[
        speaker: chooses(comp_class in Comp_classes, wpp = 1),
        #mu, sigma = state_gaussian_params(comp_class, sub_mu, sub_sigma)            # not okay, change so there's no variable assignment
        speaker: given(true_state in States, wpp =  norm.pdf(true_state, 0, 1)),
                                            # loc = state_gaussian_params(comp_class, sub_mu, sub_sigma)[0], 
                                            # scale = state_gaussian_params(comp_class, sub_mu, sub_sigma)[1])),
        # Thresholds = threshold_bins(utterance_form, States, bin_param)        # can utterance/form be passed in here if speaker has not chosen it yet?
        speaker: chooses(threshold in Thresholds, wpp = 1),
        speaker: chooses(utterance in Utterances, wpp = exp(imagine[
            listener: knows(utterance),
            listener: knows(threshold),    # is it true that the naive listener knows the threshold?
            listener: chooses(state in States, wpp = meaning(utterance, state, threshold)),     # should this be based on one threshold or an array of threshold values?
            Pr[listener.state == true_state]
        ]))
    ]
    # return listener[E[speaker.comp_class == comp_class]]
    listener: observes[speaker.utterance] is utterance
    #listener: chooses(state in States, wpp = E[speaker.state == state])         # is this needed?
    listener: chooses(comp_class in Comp_classes, wpp = E[speaker.comp_class == comp_class])
    return E[listener.comp_class == comp_class]#"sub"]

# print(comparison(0,0.5))


# np.array([norm.pdf(state, 
#                                             loc = state_gaussian_params(comp_class, sub_mu, sub_sigma)[0], 
#                                             scale = state_gaussian_params(comp_class, sub_mu, sub_sigma)[1])
#                                             for state in States])