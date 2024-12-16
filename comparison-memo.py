from memo import memo
import jax
import jax.numpy as np
from jax.scipy.stats import norm
from functools import partial
from jax import lax
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as normalpy
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

expt1DataFile = "./data/class-elicitation-full-trials.csv"
expt2DataFile = "./data/vague-prior-elicitation-1-trials.csv"


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
s2_Utterances_all = {
    "positive": [0,1],#[silence_silence, positive_adjective],
    "negative": [0,-1]#[silence_silence, negative_adjective]
}
s2_Utterances = np.array(s2_Utterances_all["positive"]+s2_Utterances_all["negative"])
s2_Comp_classes = [1]   # just super




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
def exp2_speaker[real_utterance: s2_Utterances](sub_mu, sub_sigma, Thresholds, exp2_alpha1, exp2_alpha2):
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
# print(exp2_speaker(3, 0.5, threshold_bins("positive", States, bin_param), exp2_alpha1, exp2_alpha2)[:2])
# print(exp2_speaker(3, 0.5, threshold_bins("negative", States, bin_param), exp2_alpha1, exp2_alpha2)[2:])


# expt2DataFile

## EXPT 2 MODEL FITTING

# stupid function to get the index needed from each model output
@partial(jax.jit, static_argnums=(0))
def get_output_idx_model_expt2(form):   
    return lax.cond(form == "positive", lambda _: 0, lambda _: 2, 0)
    # if form == "positive":
    #     return 0
    # if form == "negative":
    #     return 2

# TO FIX -- check the output format
def get_output_idx_model_expt1(form):
    if form == "positive":
        return "hi"
    if form == "negative":
        return "blah"

# fit-evaluation function (MSE)
@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def mse(data, form, sub_mu, sub_sigma, exp2_alpha1, exp2_alpha2):
    # model_prob = exp2_speaker(sub_mu, sub_sigma, 
    #        
    #                    threshold_bins(form, States, bin_param), exp2_alpha1, exp2_alpha2)[get_output_idx_model_expt2(form)]
    # return np.mean((data - exp2_speaker(sub_mu, sub_sigma, 
    #                           threshold_bins(form, States, bin_param), exp2_alpha1, exp2_alpha2)[get_output_idx_model_expt2(form)]) ** 2)
    thresholded_bins = threshold_bins(form, States, bin_param)
    model_output = exp2_speaker(sub_mu, sub_sigma, thresholded_bins, exp2_alpha1, exp2_alpha2)

    model_prob = model_output[get_output_idx_model_expt2(form)]
    return np.mean((data - model_prob) ** 2)
grad = jax.value_and_grad(mse)

# print(mse(0, "positive", 3, 0.5, exp2_alpha1, exp2_alpha2))


# extract data 
expt_2_data = pd.read_csv(expt2DataFile)

with open('webgram.json') as json_file:
    priors = json.load(json_file)
# create dict with all empirical probabilities from expt 2
empirical_category_priors = {}
optimal_sigmas = {}
optimal_mus = {}
optimal_expt2_model_outputs = {}

for form in ["positive", "negative"]:
    form_df = expt_2_data[expt_2_data["form"] == form]
    
    empirical_category_priors[form] = {}
    optimal_sigmas[form] = {}
    optimal_mus[form] = {}
    optimal_expt2_model_outputs[form] = {}

    for type, dists in priors.items():
        type_df = form_df[form_df["super_category"] == type]

        empirical_category_priors[form][type] = {}
        optimal_sigmas[form][type] = {}
        optimal_mus[form][type] = {}
        optimal_expt2_model_outputs[form][type] = {}

        for subcat, freqs in dists['sub'].items():
            subcat_df = type_df[type_df["sub_category"] == subcat]
            empirical_prob = subcat_df["response"].mean()
            # print(form, subcat, empirical_prob)
            empirical_category_priors[form][type][subcat] = empirical_prob

            sub_mu = 0       # initial value, might need to change
            sub_sigma = 0.5        # initial value, might need to change
            sigmas = []
            mus = []
            mses = []
            mu_grid = np.linspace(-5, 5, 100)
            sig_grid = np.linspace(-5, 5, 100)
            mus, sigs = np.meshgrid(mu_grid,sig_grid)
            grid=np.array([mus.flatten(),sigs.flatten()]).T
            print(empirical_prob)
            @jax.jit 
            def mse_specific(tup):
                return mse(empirical_prob, form, tup[0], tup[1], exp2_alpha1, exp2_alpha2)
            print(len(grid))
            
            result = jax.vmap(mse_specific)(grid)
            minarg = normalpy.argmin(ma.masked_where(result==0, result))
            thresholded_bins = threshold_bins(form, States, bin_param)
            print(exp2_speaker(*grid[minarg], thresholded_bins, exp2_alpha1, exp2_alpha2))
            print(grid[minarg], result[minarg])
            print(mus.shape, sigs.shape, result.shape)
            sub_mu, sub_sigma = grid[minarg]
            plt.ioff()
            plt.contourf(mus.flatten().reshape(mus.shape), sigs.flatten().reshape(mus.shape), result.reshape(mus.shape), 30)
            plt.colorbar()
            plt.savefig(f'./mse_plots/{form}_{subcat}.png')
            plt.clf()
            # for t in range(10 + 1):
            #     print("starting a round of descent")
            #     mse_value, mse_grad = grad(empirical_prob, form, sub_mu, sub_sigma, exp2_alpha1, exp2_alpha2)
            #     print("graduated!")
            #     sub_mu = sub_mu - 0.01 * mse_grad
            #     sub_sigma = sub_sigma - 0.01 * mse_grad
            #     if t % 10 == 0:
            #         sigmas.append(sub_sigma)
            #         mus.append(sub_mu)
            #         mses.append(mse_value)
            #     print(sub_mu, sub_sigma)

            print("optimal values found for", subcat)
            optimal_sigmas[form][type][subcat] = sub_sigma
            optimal_mus[form][type][subcat] = sub_mu
            optimal_expt2_model_outputs[form][type][subcat] = exp2_speaker(sub_mu, sub_sigma, 
                                                                           threshold_bins(form, States, bin_param), exp2_alpha1, exp2_alpha2)[get_output_idx_model_expt2(form)]

# print(empirical_category_priors)
# print(optimal_mus)
# print(optimal_sigmas)
# print(optimal_expt2_model_outputs)

# print(empirical_category_priors)




## RUNNING EXPT 1 MODEL
# with open('webgram.json') as json_file:
#     priors = json.load(json_file)
# for type, dists in priors.items():
#     print(type)
#     supPrior = sum(dists['super'])
#     for subcat, freqs in dists['sub'].items():
#         subPrior = sum(freqs)/(sum(freqs)+supPrior)
#         supPrior = supPrior/(subPrior+supPrior)

#         print(f"subcat: {subcat}, supercat: {type}")
#         print(comparison(1.2,0.4,threshold_bins("positive", States, bin_param), 
#                          subPrior, supPrior, exp1_alpha, beta)[:3])
#         print(comparison(1.2,0.4,threshold_bins("negative", States, bin_param), 
#                          subPrior, supPrior, exp1_alpha, beta)[3:])