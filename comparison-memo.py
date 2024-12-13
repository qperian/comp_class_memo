from memo import memo
import jax
import jax.numpy as np
from jax.scipy.stats import norm
from functools import partial
from jax import lax


expt1DataFile = "../data/class-elicitation-full-trials.csv"
expt2DataFile = "../data/vague-prior-elicitation-1-trials.csv"

Utterances = {
    "positive": [1,2,3],#["positive_silence", "positive_sub", "positive_super"],
    "negative": [-1,-2,-3]#["negative_silence", "negative_sub", "negative_super"]
}

@partial(jax.jit, static_argnums=(0,2))
def meaning(utterance, state, threshold):
    '''
    utterance_form = "positive" or "negative" - can't pass something that's not an array into JAX ahhh
    states = array of possible states
    threshold = threshold value to apply scalar adjective

    returns an array of boolean meaning values for all states
    '''
    # utterance_form = get_form(utterance)
    return lax.cond(utterance > 0, lambda _ : state > threshold, 
                    lambda _ : state < threshold, 0)
    # else:
    #     raise ValueError("incorrect utterance format -- form is not positive or negative")

@partial(jax.jit, static_argnums=(1)) ## 0 is sub, 1 super
def mu_finder(comp_class, sub_mu):
    return (1-comp_class)*sub_mu
@partial(jax.jit, static_argnums=(1))
def sigma_finder(comp_class, sub_sigma):
    return (1-comp_class)*sub_sigma+comp_class

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
    mapping_func = jax.vmap(threshold_func, 0, 0)       # assumes that the inputted array for States is one-directional
    return mapping_func(States)

@jax.jit
def normal(x, mean, stdev):
    return norm.pdf(x, loc=mean, scale=stdev)

# States = np.array([1, 2, 3, 4])
# Thresholds = threshold_bins("negative", States, 3)
# print(type(Thresholds))
# print(Thresholds)
@jax.jit
def utterance_to_cc(utterance, default):
    utterance = np.abs(utterance)
    return (utterance-2)*(utterance-3)*default/2+(utterance-1)*(utterance-2)/2

Comp_classes = [0,1]        # 0 = sub, 1 = super
bin_param = 3
utterance_form = "positive"
# define thresholds globally -- nope
#Thresholds = ...
stateParams = {'mu': 0, 'sigma': 1}
States = np.array(list(map(round, range((stateParams['mu'] - 3*stateParams['sigma'])*bin_param, 
                          (stateParams['mu'] + 3*stateParams['sigma'])*bin_param, 
                          stateParams['sigma']))))/bin_param

Thresholds = threshold_bins(utterance_form, States, bin_param)
#print("threshold", Thresholds)
Utterances = Utterances["positive"]
# Utterances = np.array(Utterances["positive"]+Utterances["negative"])
#print("utterances", Utterances)

# @partial(memo, debug_trace=True, debug_print_compiled=True)
@memo
def comparison[real_utterance: Utterances, guess_comp_class: Comp_classes](sub_mu, sub_sigma):
    cast: [speaker, listener]
    # listener: knows(guess_comp_class)
    # listener: knows(guess_utterance)
    # listener: knows(guess_state)

    listener: thinks[
        speaker: chooses(default_comp_class in Comp_classes, wpp = 1),
        speaker: given(state in States, wpp =  normal(state,sub_mu, sub_sigma)),
        speaker: given(threshold in Thresholds, wpp = 1),
        speaker: chooses(utterance in Utterances, wpp = imagine[
            listener: knows(utterance),
            listener: knows(threshold),    # is it true that the naive listener knows the threshold?
            # listener: chooses(default_comp_class in Comp_classes, wpp = 1),
            listener: knows(default_comp_class),
            listener: chooses(state in States, 
                        wpp = meaning(utterance, state, threshold)*normal(
                            state, mu_finder(utterance_to_cc(utterance,default_comp_class), sub_mu), 
                            sigma_finder(utterance_to_cc(utterance,default_comp_class), sub_sigma))),
            Pr[listener.state == state]
        ])
    ]
    # return listener[E[speaker.comp_class == comp_class]]
    listener: observes[speaker.utterance] is real_utterance
    #listener: chooses(state in States, wpp = E[speaker.state == state])         # is this needed?
    listener: chooses(comp_class in Comp_classes, wpp = E[speaker.default_comp_class == comp_class])
    return E[listener.comp_class == guess_comp_class]

print(comparison(3,0.2))