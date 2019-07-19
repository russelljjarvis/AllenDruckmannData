import base
from types import MethodType
from base import VmTest
from get_three_features_from_allen_data import inject_square_current, get_data_sets_from_cache, get_data_sets_from_remote
import sciunit
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
import allensdk
import pickle
import glob
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
import pdb
import numpy as np
import copy
from neuronunit.tests.fi import RheobaseTest as generic
from sciunit.scores.complete import ZScore

def allen_format(volts,times,key):
    '''
    Synposis:
        At its most fundamental level, AllenSDK still calls a single trace a sweep.
        In otherwords there are no single traces, but there are sweeps of size 1.
        This is a bit like wrapping unitary objects in iterable containers like [times].

    inputs:
        np.arrays of time series: Specifically a time recording vector, and a membrane potential recording.
        in floats probably with units striped away
    outputs:
        a data frame of Allen features, a very big dump of features as they pertain to each spike in a train.

        to get a managable data digest
        we out put features from the middle spike of a spike train.
    '''
    ext = EphysSweepSetFeatureExtractor([times],[volts])
    ext.process_spikes()
    swp = ext.sweeps()[0]
    spikes = swp.spikes()
    if len(spikes)==0:
        return (None,None)
    meaned_features_1 = {}
    skeys = [ skey for skey in spikes[0].keys() ]
    for sk in skeys:
        if str('isi_type') not in sk:
            meaned_features_1[sk] = np.mean([ i[sk] for i in spikes if type(i) is not type(str(''))] )
    allen_features = {}
    meaned_features_overspikes = {}
    for s in swp.sweep_feature_keys():# print(swp.sweep_feature(s))
        if str('isi_type') not in s:
            allen_features[s] = swp.sweep_feature(s)

    allen_features.update(meaned_features_1)
    return allen_features[key], allen_features

data_sets = get_data_sets_from_cache(do_features=False)

numbers = data_sets[0][0].get_sweep_numbers()
sweeps = []
for n in numbers:
    sweeps.append(data_sets[0][0].get_sweep(n))

stim_types = [ data_sets[0][0].get_sweep_metadata(n)['aibs_stimulus_name'] for n in numbers ]
responses = [ data_sets[0][0].get_sweep(n)['response'] for n in numbers ]

try:
    with open('allen_test.p','rb') as f:
        pre_obs = pickle.load(f)

except:
    pre_obs = allensdk.ephys.extract_cell_features.extract_feature_wave_russell(responses[0], stim_types[0], data_sets[0][0],numbers)
    observation = {}
    observation['value'] = pre_obs
    with open('allen_test.p','wb') as f:
        pickle.dump(pre_obs,f)
#['isi_cv'], 'spikes', 'mean_isi', 'id', 'adapt', 'latency', 'median_isi', 'avg_rate', 'first_isi'])
cv = {}
cv['mean'] = pre_obs[4]['isi_cv']
latency = {}
latency['mean'] = pre_obs[4]['latency']
avg_rate = {}
avg_rate['mean'] = pre_obs[4]['avg_rate']
median_isi = {}
median_isi['mean'] = pre_obs[4]['median_isi']
upstroke = {}
upstroke['mean'] = pre_obs[4]['spikes'][0]['upstroke_v']
upstroke['std'] = 1

width = {}
width['mean'] = pre_obs[4]['spikes'][0]['width']
height = {}
height['mean'] = pre_obs[4]['spikes'][0]['peak_v']
###
#
cv_test = sciunit.Test(cv)
cv_test.name = 'isi_cv'
latency_test = sciunit.Test(latency)
latency_test.name = 'latency'
avg_rate_test = sciunit.Test(avg_rate)
avg_rate_test.name = 'avg_rate'
median_isi_test = sciunit.Test(median_isi)
median_isi_test.name = 'median_isi'
width_test = sciunit.Test(width)
width_test.name = 'width'
upstroke_test = sciunit.Test(upstroke)
upstroke_test.name = 'upstroke_v'
peak_test = sciunit.Test(height)
peak_test.name = 'peak_v'

test_collection = [cv_test,latency_test,avg_rate_test,median_isi_test,width_test,upstroke_test,peak_test]
##
# Ingredients: models, tests to hijack and steal methods from
##

##
# Load NeuroML static models
##
viable_models = []
size = len(glob.glob("models/*.p"))
#while len(viable_models)<10:
temp_paths = list(glob.glob("models/*.p"))[0:10]

models = [ pickle.load(open(f,'rb')) for f in temp_paths ]
for m in models: m.static = None
for m in models: m.static = True


model = models[0]


def generate_prediction(self, model):
    key = copy.copy(self.name)
    if model.static = True:
        model.inject_square_current(model.druckmann2013_strong_current)
        volts = np.array([float(v) for v in model.vm30.magnitude])
        times = np.array([float(t) for t in model.vm30.times])
    else:
        dtc = DataTransportContainer()
        dtc.backend = 'BADEXP'
        rtest = get_rtest(dtc)
        rheobase = rtest.generate_prediction(model)
        keyed = {}
        keyed['injected_square_current'] = {}
        DURATION = 1000.0*pq.ms
        DELAY = 100.0*pq.ms
        if type(rheobase) is type({str('k'):str('v')}):
            keyed['injected_square_current']['amplitude'] = float(rheobase['value'])*pq.pA
        else:
            keyed['injected_square_current']['amplitude'] = rheobase
        keyed['injected_square_current']['delay']= DELAY
        keyed['injected_square_current']['duration'] = DURATION
        model.injected_square_current
    prediction = {}

    if not hasattr(model,'lookup'):# is None:
        pred,model.lookup = allen_format(volts,times,key)

    else:
        pred = model.lookup[key]
    prediction['model'] = model
    prediction['value'] = pred
    prediction['mean'] = pred
    return prediction

def judge(model,test):
    prediction = test.generate_prediction(model)
    model = prediction['model']
    score = VmTest.compute_score(test,test.observation,prediction)
    return score, model


##
# missing generate prediction
##
#cv_test.compute_score = MethodType(compute_score,cv_test)
def mutate_test(test,generic):
    generic = generic()
    generic.key = None
    generic.key = test.name
    generic.observation = test.observation
    generic.name = test.name
    generic.generate_prediction = MethodType(generate_prediction,generic)
    generic.judge = MethodType(judge,generic)
    return generic

generics = [ copy.copy(generic) for t in test_collection ]


for i,(t,g) in enumerate(zip(test_collection,generics)): test_collection[i] = mutate_test(t,g)
names = [ t.name for t in test_collection ]
keys = [ t.key for t in test_collection ]

test_collection[-2].score_type = ZScore
scores = [ judge(model,t) for t in test_collection ]
'''
#allensdk.ephys.extract_cell_features.mean_features_spike_zero(sweeps)

allensdk.ephys.extract_cell_features.extract_sweep_features(data_sets[0][0], sweeps_by_type)#[source]

allensdk.ephys.extract_cell_features.get_ramp_stim_characteristics(i, t)#[source]

#    Identify the start time and start index of a ramp sweep.

allensdk.ephys.extract_cell_features.get_square_stim_characteristics(i, t, no_test_pulse=False)#[source]

#    Identify the start time, duration, amplitude, start index, and end index of a square stimulus. This assumes that there is a test pulse followed by the stimulus square.

allensdk.ephys.extract_cell_features.get_stim_characteristics(i, t, no_test_pulse=False)#[source]

#    Identify the start time, duration, amplitude, start index, and end index of a general stimulus. This assumes that there is a test pulse followed by the stimulus square.
'''
