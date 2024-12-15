from memo import memo
import jax
import jax.numpy as np
from jax.scipy.stats import norm
from functools import partial
from jax import lax
import json


## UTIL FUNCTIONS  

# meaning func for expt 1
@partial(jax.jit, static_argnums=(0,2))
def meaning(utterance, state, threshold):
    '''
    utterance_form = "positive" or "negative" - can't pass something that's not an array into JAX ahhh
    states = array of possible states
    threshold = threshold value to apply scalar adjective

    returns an array of boolean meaning values for all states
    '''
    return lax.cond(utterance > 0, lambda _ : state > threshold, 
                    lambda _ : state < threshold, 0)

# meaning func for expt 2
@partial(jax.jit, static_argnums=(0,2))
def s2_meaning(utterance, state, threshold):
    '''
    utterance_form = 0 (silence) or 1/-1 (positive super, negative super)
    states = array of possible states
    threshold = threshold value to apply scalar adjective

    returns an array of boolean meaning values for all states
    '''
    return lax.cond(utterance == 0, lambda _ : True, lambda _ : 
                    lax.cond(utterance > 0, lambda _ : state > threshold, 
                             lambda _ : state < threshold, 0), 0)
    # return lax.cond(utterance > 0, lambda _ : state > threshold, 
    #                 lambda _ : state < threshold, 0)


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


@partial(jax.jit, static_argnums=(0, 1))
def catPrior(subprior, superprior, comp_class):
    return comp_class*superprior + (1-comp_class)*subprior


@jax.jit
def utterance_to_cc(utterance, default):
    utterance = np.abs(utterance)
    return (utterance-2)*(utterance-3)*default/2+(utterance-1)*(utterance-2)/2



## PARAMETERS AND DATA

expt1DataFile = "../data/class-elicitation-full-trials.csv"
expt2DataFile = "../data/vague-prior-elicitation-1-trials.csv"


# params/priors for both models
bin_param = 3
stateParams = {'mu': 0, 'sigma': 1}
# utterance_form = "positive"
States = np.array(list(map(round, range((stateParams['mu'] - 3*stateParams['sigma'])*bin_param, 
                          (stateParams['mu'] + 3*stateParams['sigma'])*bin_param, 
                          stateParams['sigma']))))/bin_param
# Thresholds = threshold_bins(utterance_form, States, bin_param)


# alpha and beta parameters (from paper)
exp1_alpha = 1.6        # speaker optimality param for expt 1 speaker
beta = 0.13             # frequency scale (only relevant for expt 1)
exp2_alpha1 = 3.5       # speaker optimality param for speaker 1     
exp2_alpha2 = 3.2       # speaker optimality param for speaker 2


# params/priors for expt 1 model
Comp_classes = [0,1]        # 0 = sub, 1 = super
Utterances_all = {
    "positive": [1,2,3],#["positive_silence", "positive_sub", "positive_super"],
    "negative": [-1,-2,-3]#["negative_silence", "negative_sub", "negative_super"]
}
Utterances = np.array(Utterances_all["positive"]+Utterances_all["negative"])


# params/priors for expt 2 model
s2_Utterances = {
    "positive": [0,1],#[silence_silence, positive_adjective],
    "negative": [0,-1]#[silence_silence, negative_adjective]
}
s2_Comp_classes = [1]   # just super
s2_Utterances = s2_Utterances[utterance_form]




## EXPT 1 MODEL
# @partial(memo, debug_trace=True, debug_print_compiled=True)
@memo
def comparison[real_utterance: Utterances, guess_comp_class: Comp_classes](sub_mu, sub_sigma, Thresholds, subprior, superprior, exp1_alpha, beta):
    cast: [speaker, listener]
    # listener: knows(guess_comp_class)
    # listener: knows(guess_utterance)
    # listener: knows(guess_state)

    listener: thinks[
        speaker: chooses(default_comp_class in Comp_classes, wpp = exp(beta*log(catPrior(subprior, superprior, default_comp_class)))),
        speaker: given(state in States, wpp =  normal(state,sub_mu, sub_sigma)),
        speaker: given(threshold in Thresholds, wpp = 1),
        speaker: chooses(utterance in Utterances, wpp = exp(exp1_alpha*log(imagine[
            listener: knows(utterance),
            listener: knows(threshold),    
            listener: knows(default_comp_class),
            listener: chooses(state in States, 
                        wpp = meaning(utterance, state, threshold)*normal(
                            state, mu_finder(utterance_to_cc(utterance,default_comp_class), sub_mu), 
                            sigma_finder(utterance_to_cc(utterance,default_comp_class), sub_sigma))),
            Pr[listener.state == state]
        ])))
    ]
    listener: observes[speaker.utterance] is real_utterance
    listener: chooses(comp_class in Comp_classes, wpp = E[speaker.default_comp_class == comp_class])
    return E[listener.comp_class == guess_comp_class]





## EXPT 2 MODEL
@memo 
def exp2_speaker[real_utterance: s2_Utterances](sub_mu, sub_sigma, exp2_alpha1, exp2_alpha2):
    # cast: [speaker, listener]

    speaker2: given(state_belief in States, wpp = normal(state_belief, sub_mu, sub_sigma))
    speaker2: chooses(real_utterance in s2_Utterances, wpp = exp(exp2_alpha2*log(imagine[
        listener2: knows(real_utterance),
        listener2: thinks[
            speaker1: chooses(default_comp_class in s2_Comp_classes, wpp = 1),       # comp class is always 1 (super)
            speaker1: given(state in States, wpp = normal(state, 0, 1)),    # state sampled from super distribution
            speaker1: given(threshold in Thresholds, wpp = 1),
            speaker1: chooses(utterance in s2_Utterances, wpp = exp(exp2_alpha1*log(imagine[
                listener1: knows(utterance),
                listener1: knows(threshold),    
                listener1: knows(default_comp_class),
                listener1: chooses(state in States, 
                            wpp = s2_meaning(utterance, state, threshold)*normal(
                                state, mu_finder(default_comp_class, sub_mu),   # always used passed in comp class (super)
                                sigma_finder(default_comp_class, sub_sigma))),
                Pr[listener1.state == state]
            ])))
        ],
        listener2: observes[speaker1.utterance] is real_utterance,
        listener2: chooses(state in States, wpp = E[speaker1.state == state]),
        Pr[listener2.state == state_belief]
        ])))
    return E[speaker2.real_utterance == real_utterance]
    # need to incorporate the expected value of the state based on S2 state belief, plus alpha 

# print(exp2_speaker(-2,0.5, exp2_alpha1, exp2_alpha2))





## RUNNING EXPT 1 MODEL
with open('webgram.json') as json_file:
    priors = json.load(json_file)
for type, dists in priors.items():
    print(type)
    supPrior = sum(dists['super'])
    for subcat, freqs in dists['sub'].items():
        subPrior = sum(freqs)/(sum(freqs)+supPrior)
        supPrior = supPrior/(subPrior+supPrior)

        print(f"subcat: {subcat}, supercat: {type}")
        print(comparison(1.2,0.4,threshold_bins("positive", States, bin_param), 
                         subPrior, supPrior, exp1_alpha, beta)[:3])
        print(comparison(1.2,0.4,threshold_bins("negative", States, bin_param), 
                         subPrior, supPrior, exp1_alpha, beta)[3:])