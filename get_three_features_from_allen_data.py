
##
#  The main method that does the aligned feature extraction is down the bottom.
# Two thirds of this file, it is called
# def three_feature_sets_on_static_models
##

##
# docker pull russelljarvis/efel_allen_dm
# I build it with the name russelljarvis/efel_allen_dm.
# This uses the docker file in this directory.
# I build it with the name efl.
# and launch it with this alias.
# alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata efel /bin/bash'
##

##
# This is how my travis script builds and runs:
# before_install:
# - docker pull russelljarvis/efel_allen_dm
# - git clone -b barcelona https://github.com/russelljjarvis/neuronunit.git
#
# Run the unit test
# script:
# show that running the docker container at least works.
#  - docker run -v neuronunit:/home/jovyan/neuronunit russelljarvis/efel_allen_dm python /home/jovyan/work/allendata/small_travis_run.py
#
##

from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
try:
    import cPickle
except:
    import _pickle as cPickle
import csv
import json
import os
import re
import shutil
import string
import urllib

import inspect

import numpy as np
from matplotlib import pyplot as plt

import dask.bag as dbag # a pip installable module, usually installs without complication
import dask
import urllib.request, json
import os
import requests
from neo.core import AnalogSignal
from quantities import mV, ms, nA
from neuronunit import models
import pickle
import efel
from types import MethodType
import quantities as pq
import pdb

from collections import Iterable, OrderedDict

import numpy as np
import efel
import pickle
from allensdk.ephys.extract_cell_features import extract_cell_features
import pandas as pd
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from neuronunit.neuromldb import NeuroMLDBStaticModel

import dm_test_interoperable #import Interoperabe
from dask import bag as db
import glob


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('agg')
#import logging
#logging.info("test")
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.ephys.extract_cell_features import extract_cell_features
from collections import defaultdict
#from neuronunit.optimisation.optimisation_management import inject_rh_and_dont_plot, add_dm_properties_to_cells

from allensdk.core.nwb_data_set import NwbDataSet
import pickle

import aibs

#dm_tests = init_dm_tests(value,1.5*value)

#try:
#    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
#except:#
#    pass#
#    all_features = ctc.get_all_features()
#    pickle.dump(all_features,open('all_features.p','wb'))
#





'''
import pdb; pdb.set_trace()
prefix = str('/dedicated_folder')
try:
    os.mkdir(prefix)
except:
    pass
pickle.dump(everything,open(prefix+str(specimen_id)+'.p','wb'))
'''

def generate_prediction(self,model):
    prediction = {}
    prediction['n'] = 1
    prediction['std'] = 1.0
    prediction['mean'] = model.rheobase['mean']
    return prediction

def find_nearest(array, value):
    #value = float(value)
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)

def get_m_p(model,current):
    '''
    synopsis:
        get_m_p belongs to a 3 method stack (2 below)

    replace get_membrane_potential in a NU model class with a statically defined lookup table.


    '''
    try:
        consolted = model.lookup[float(current['amplitude'])]
    except:
        consolted = model.lookup[float(current['injected_square_current']['amplitude'])]
    return consolted

def update_static_model_methods(model,lookup):
    '''
    Overwrite/ride. a NU models inject_square_current,generate_prediction methods
    with methods for querying a lookup table, such that given a current injection,
    a V_{m} is returned.
    '''
    model.lookup = lookup
    model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential

    return model#, tests

#Depreciated
def map_to_sms(tt):

    # given a list of static models, update the static models methods
    #for model in sms:
    #model.inject_square_current = MethodType(get_m_p,model)#get_membrane_potential
    for t in tt:
        if 'RheobaseTest' in t.name:
            t.generate_prediction = MethodType(generate_prediction,t)
    return sms

def standard_nu_tests_two(model):
    '''
    Do standard NU predictions, to do this may need to overwrite generate_prediction
    Overwrite/ride. a NU models inject_square_current,generate_prediction methods
    with methods for querying a lookup table, such that given a current injection,
    a V_{m} is returned.
    '''
    rts,complete_map = pickle.load(open('russell_tests.p','rb'))
    local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]
    #model = update_static_model_methods(model,lookup)
    nu_preds = []
    for t in local_tests:
        if str('Rheobase') not in t.name:
            #import pdb; pdb.set_trace()
            try:
                pred = t.generate_prediction(model)
            except:
                pred = None
        nu_preds.append(pred)
    return nu_preds


def standard_nu_tests(model,lookup):
    '''
    Do standard NU predictions, to do this may need to overwrite generate_prediction
    Overwrite/ride. a NU models inject_square_current,generate_prediction methods
    with methods for querying a lookup table, such that given a current injection,
    a V_{m} is returned.
    '''
    rts,complete_map = pickle.load(open('russell_tests.p','rb'))
    local_tests = [value for value in rts['Hippocampus CA1 pyramidal cell'].values() ]
    model = update_static_model_methods(model,lookup)
    nu_preds = []
    for t in local_tests:
        #import pdb; pdb.set_trace()
        try:
            pred = t.generate_prediction(model)
        except:
            pred = None
        nu_preds.append(pred)
    return nu_preds


def crawl_ids(url):
    ''' move to aibs '''
    all_data = requests.get(url)
    all_data = json.loads(all_data.text)
    Model_IDs = []
    for d in all_data:
        Model_ID = str(d['Model_ID'])
        Model_IDs.append(Model_ID)
    return Model_IDs

list_to_get =[ str('https://www.neuroml-db.org/api/search?q=traub'),
    str('https://www.neuroml-db.org/api/search?q=markram'),
    str('https://www.neuroml-db.org/api/search?q=Gouwens') ]

def get_all_cortical_cells(list_to_get):
    model_ids = {}
    for url in list_to_get:
        Model_IDs = crawl_ids(url)
        parts = url.split('?q=')
        try:
            model_ids[parts[1]].append(Model_IDs)
        except:
            model_ids[parts[1]] = []
            model_ids[parts[1]].append(Model_IDs)
    with open('cortical_cells_list.p','wb') as f:
        pickle.dump(model_ids,f)

    return model_ids



def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return (array[idx], idx)


def get_waveform_current_amplitude(waveform):
    return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA

def inject_square_current(model,current,data_set=None):
    '''
    model, data_set is basically like a lookup table.
    '''
    if type(current) is type({}):
        current = float(current['amplitude'])
    if data_set == None:
        data_set = model.data_set
        sweeps = model.sweeps
    numbers = data_set.get_sweep_numbers()
    injections = [ np.max(data_set.get_sweep(sn)['stimulus']) for sn in numbers ]
    sns = [ sn for sn in numbers]
    print([s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')])
    #sm.sweeps
    import pdb; pdb.set_trace()
    (nearest,idx) = find_nearest(injections,current)
    index = np.asarray(numbers)[idx]
    sweep_data = data_set.get_sweep(index)
    temp_vm = sweep_data['response']
    injection = sweep_data['stimulus']
    sampling_rate = sweep_data['sampling_rate']
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    model._vm = vm
    return model._vm

def get_membrane_potential(model):
    return model._vm

"""Auxiliary helper functions for analysis of spiking."""

def get_spike_train(vm, threshold=0.0*mV):
    """
    Inputs:
     vm: a neo.core.AnalogSignal corresponding to a membrane potential trace.
     threshold: the value (in mV) above which vm has to cross for there
                to be a spike.  Scalar float.

    Returns:
     a neo.core.SpikeTrain containing the times of spikes.
    """
    from elephant.spike_train_generation import threshold_detection

    spike_train = threshold_detection(vm, threshold=threshold)
    return spike_train

def get_spike_count(model):
    vm = model.get_membrane_potential()
    train = get_spike_train(vm)
    return len(train)



def get_data_sets(number_d_sets=2):
    try:
        with open('../all_allen_cells.p','rb') as f:
            cells = pickle.load(f)
    except:
        ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
        cells = ctc.get_cells()
        with open('all_allen_cells.p','wb') as f:
            pickle.dump(cells,f)
    data = []
    data_sets = []
    path_name = 'data_nwbs'

    try:
        os.mkdir(path_name)
    except:
        print('directory already made.')

    ids = [ c['id'] for c in cells ]

    for specimen_id in ids[0:number_d_sets]:
        temp_path = str(path_name)+str('/')+str(specimen_id)+'.p'
        if os.path.exists(temp_path):
            with open(temp_path,'rb') as f:
                (data_set_nwb,sweeps,specimen_id) = pickle.load(f)
            data_sets.append((data_set_nwb,sweeps,specimen_id))
        else:
            data_set = ctc.get_ephys_data(specimen_id)
            sweeps = ctc.get_ephys_sweeps(specimen_id)

            file_name = 'cell_types/specimen_'+str(specimen_id)+'/ephys.nwb'
            data_set_nwb = NwbDataSet(file_name)
            data_sets.append((data_set_nwb,sweeps,specimen_id))

            with open(temp_path,'wb') as f:
                pickle.dump((data_set_nwb,sweeps,specimen_id),f)


    return data_sets
from get_three_feature_sets_from_nml_db import three_feature_sets_on_static_models
import quantities as qt



def allen_to_model_and_features(content):
    data_set,sweeps,specimen_id = content
    sweep_numbers_ = defaultdict(list)
    for sweep in sweeps:
        sweep_numbers_[sweep['stimulus_name']].append(sweep['sweep_number'])

    sweep_numbers = data_set.get_sweep_numbers()
    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)


    cell_features = extract_cell_features(data_set, sweep_numbers_['Ramp'],sweep_numbers_['Short Square'],sweep_numbers_['Long Square'])

    sweep_numbers = data_set.get_sweep_numbers()
    smallest_multi = 1000
    all_currents = []
    for sn in sweep_numbers:
        spike_times = data_set.get_spike_times(sn)
        sweep_data = data_set.get_sweep(sn)

        if len(spike_times) == 1:
            inj_rheobase = np.max(sweep_data['stimulus'])

        if len(spike_times) < smallest_multi and len(spike_times) > 1:
            smallest_multi = len(spike_times)
            inj_multi_spike = np.max(sweep_data['stimulus'])
            temp_vm = sweep_data['response']
        val = np.max(sweep_data['stimulus'])
        all_currents.append(val)


    supras = [s for s in sweeps if s['stimulus_name'] == str('Square - 2s Suprathreshold')]
    supra_currents = [s['stimulus_absolute_amplitude']*1E-11 for s in supras]
    dmrheobase15 = supra_currents[0]#(1.5*inj_rheobase)
    dmrheobase30 = supra_currents[1]#(3.0*inj_rheobase)


    spike_times = data_set.get_spike_times(supras[-1]['sweep_number'])
    sweep_data = data_set.get_sweep(supras[-1]['sweep_number'])
    temp_vm = sweep_data['response']

    #(nearest_allen15,idx_nearest_allen) = find_nearest(supra_currents,dmrheobase15)
    #(nearest_allen30,idx_nearest_allen) = find_nearest(supra_currents,dmrheobase30)

    #print(nearest_allen15,nearest_allen30,inj_multi_spike)
    #if inj_multi_spike < nearest_allen15 and inj_rheobase!=nearest_allen15:# != inj_rheobase:
    #    pass
    #else:
    #    pass

    injection = sweep_data['stimulus']
    # sampling rate is in Hz
    sampling_rate = sweep_data['sampling_rate']
    vm = AnalogSignal(temp_vm,sampling_rate=sampling_rate*qt.Hz,units=qt.V)
    #model = NeuroMLDBStaticModel(cell_id)
    #from neuronunit.models.static import StaticModel

    sm = models.StaticModel(vm)
    sm.druckmann2013_standard_current = dmrheobase15
    sm.druckmann2013_strong_current = dmrheobase30
    sm.name = specimen_id
    sm.data_set = data_set
    sm.sweeps = sweeps
    sm.inject_square_current = MethodType(inject_square_current,sm)
    sm.get_membrane_potential = MethodType(get_membrane_potential,sm)


    sm.rheobase_current = inj_rheobase
    #sm.druckmann2013_standard_current = dmrheobase15
    #sm.druckmann2013_strong_current = dmrheobase30
    current = {}
    current['amplitude'] = sm.rheobase_current
    sm.vm_rheobase = sm.inject_square_current(current)
    current['amplitude'] = sm.druckmann2013_standard_current
    sm.vm15 = sm.inject_square_current(current)
    current['amplitude'] = sm.druckmann2013_strong_current
    sm.vm30 = sm.inject_square_current(current)

    #try:
    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([float(t) for t in sm.vm30.times],[float(v) for v in sm.vm30], label="data", width=100, height=80)
    fig.show()

    import asciiplotlib as apl
    fig = apl.figure()
    fig.plot([float(t) for t in sm.vm15.times],[float(v) for v in sm.vm15], label="data", width=100, height=80)
    fig.show()

    sm.get_spike_count = MethodType(get_spike_count,sm)
    print(sm.get_spike_count())
    #import pdb; pdb.set_trace()
    #except:
    #    pass

    # these lines are functional

    subs = [s for s in sweeps if s['stimulus_name'] == str('Square - 0.5ms Subthreshold')]
    import pdb; pdb.set_trace()
    sm.druckmann2013_input_resistance_currents = [s['stimulus_absolute_amplitude'] for s in subs]
    sm.inject_square_current(subs[0]['stimulus_absolute_amplitude'])



    spiking_sweeps = cell_features['long_squares']['spiking_sweeps'][0]
    multi_spike_features = cell_features['long_squares']['hero_sweep']
    biophysics = cell_features['long_squares']
    shapes =  cell_features['long_squares']['spiking_sweeps'][0]['spikes'][0]

    everything = (sm,sweep_data,cell_features,vm)
    return everything


def run_on_allen(number_d_sets=2):
    data_sets = get_data_sets(number_d_sets=number_d_sets)
    models = []
    for data_set in data_sets:
        models.append(allen_to_model_and_features(data_set))
    models = [mod[0] for mod in models]
    three_feature_sets = []
    for mod in models:
        import pdb; pdb.set_trace()

        three_feature_sets.append(three_feature_sets_on_static_models(mod))



def mid_to_model(mid_):
    model = get_static_models(mid_)
    if type(model) is not type(None):
        model.name = None
        model.name = str(mid_)
        with open(str('models')+str('/')+str(mid_)+'.p','wb') as f:
            pickle.dump(model,f)
    return

def faster_make_model_and_cache():
    '''
    Synposis:

        Mass download all the NML model waveforms for all cortical regions
        And perform three types of feature extraction on resulting waveforms.

    Inputs: None
    Outputs: None in namespace, yet, lots of data written to pickle.
    '''
    all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))

    mid = [] # mid is a list of model identifiers.
    for k,v in all_the_NML_IDs.items():
        mid.extend(v[0])

    path_name = str('models')
    try:
        os.mkdir(path_name)
    except:
        print('model directory already made :)')

    ##
    # Do the batch model download.
    ##
    mid_bag = db.from_sequence(mid,npartitions=8)
    list(mid_bag.map(mid_to_model).compute())

def model_analysis(model):
    if type(model) is not type(None):
        three_feature_sets = three_feature_sets_on_static_models(model)
        try:
            assert type(model.name) is not None
            with open(str('three_feature_folder')+str('/')+str(model.name)+'.p','wb') as f:
                pickle.dump(three_feature_sets,f)
        except:
            print('big error')
            import pdb; pdb.set_trace()
    return

def analyze_models_from_cache(file_paths):
    models = []
    for f in file_paths:
        models.append(pickle.load(open(f,'rb')))
    models_bag = db.from_sequence(models,npartitions=8)
    list(models_bag.map(model_analysis).compute())

def faster_feature_extraction():
    all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))
    file_paths = glob.glob("models/*.p")
    if file_paths:
        if len(file_paths)==len(all_the_NML_IDs):
            _ = analyze_models_from_cache(file_paths)
        else:
            _ = faster_make_model_and_cache()
    else:
        _ = faster_make_model_and_cache()
    file_paths = glob.glob("models/*.p")
    _ = analyze_models_from_cache(file_paths)



def more_challenging(model):
    '''
    Isolate harder code, still wrangling data types.
    When this is done, EFEL might be able to report back about input resistance.
    '''
    single_spike = {}
    single_spike['APWaveForm'] = [ float(v) for v in model.vm_rheobase]#temp_vm
    single_spike['T'] = [ float(t) for t in model.vm_rheobase.times.rescale('ms') ]
    single_spike['V'] = [ float(v) for v in model.vm_rheobase ]#temp_vm
    single_spike['stim_start'] = [ float(model.protocol['Time_Start']) ]
    single_spike['stimulus_current'] = [ model.model.rheobase_current ]
    single_spike['stim_end'] = [ trace15['T'][-1] ]

    single_spike = [single_spike]


    ##
    # How EFEL could learn about input resistance of model
    ##
    trace_ephys_prop = {}
    trace_ephys_prop['stimulus_current'] = model.druckmann2013_input_resistance_currents[0]# = druckmann2013_input_resistance_currents[0]
    trace_ephys_prop['V'] = [ float(v) for v in model.vminh ]
    trace_ephys_prop['T'] = [ float(t) for t in model.vminh.times.rescale('ms') ]
    trace_ephys_prop['stim_end'] = [ trace15['T'][-1] ]
    trace_ephys_prop['stim_start'] = [ float(model.inh_protocol['Time_Start']) ]# = in_current_filter[0]['Time_End']
    trace_ephys_props = [trace_ephys_prop]

    efel_results_inh = efel.getFeatureValues(trace_ephys_props,list(efel.getFeatureNames()))#
    efel_results_ephys = efel.getFeatureValues(trace_ephys_prop,list(efel.getFeatureNames()))#
    return efel_results_inh

def not_necessary_for_program_completion(DMTNMLO):
    '''
    Synopsis:
       Not necessary for feature extraction pipe line.
       More of a unit test.
    '''
    standard_current = DMTNMLO.model.nmldb_model.get_druckmann2013_standard_current()
    strong_current = DMTNMLO.model.nmldb_model.get_druckmann2013_strong_current()
    volt15 = DMTNMLO.model.nmldb_model.get_waveform_by_current(standard_current)
    volt30 = DMTNMLO.model.nmldb_model.get_waveform_by_current(strong_current)
    temp0 = np.mean(volt15)
    temp1 = np.mean(volt30)
    assert temp0 != temp1
    return
