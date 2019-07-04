
##
#  The main method that does the aligned feature extraction is down the bottom.
# Two thirds of this file, it is called
# def three_feature_sets_on_static_models
##

##
# I build the docker image with the name russelljarvis/efel_allen_dm.
# meaning that the command
# docker pull russelljarvis/efel_allen_dm should work
# This uses the docker file in this directory.
# I build it with the name russelljarvis/efel_allen_dm.
# and launch it with this alias.
# alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata russelljarvis/efel_allen_dm /bin/bash'
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

def standard_nu_tests(model):
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



def get_waveform_current_amplitude(waveform):
    return float(waveform["Waveform_Label"].replace(" nA", "")) * pq.nA


def get_static_models(cell_id):
    """
    Inputs: NML-DB cell ids, a method designed to be called inside an iteration loop.

    Synpopsis: given a NML-DB id, query nml-db, create a NMLDB static model based on wave forms
        obtained for that NML-ID.
        get mainly just waveforms, and current injection values relevant to performing druckman tests
        as well as a rheobase value for good measure.
        Update the NML-DB model objects attributes, with all the waveform data/injection values obtained for the appropriate cell IDself.
    """


    url = str("https://www.neuroml-db.org/api/model?id=")+cell_id
    model_contents = requests.get(url)
    model_contents = json.loads(model_contents.text)
    model = NeuroMLDBStaticModel(cell_id)

    wlist = model_contents['waveform_list']
    long_squares = [ w for w in wlist if w['Protocol_ID'] == 'LONG_SQUARE' ]
    applied_current_injections = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Current" ]
    currents = [ w for w in wlist if w["Protocol_ID"] == "LONG_SQUARE" and w["Variable_Name"] == "Voltage" ]
    in_current_filter = [ w for w in wlist if w["Protocol_ID"] == "SQUARE" and w["Variable_Name"] == "Voltage" ]
    rheobases = []
    for wl in long_squares:
        wid = wl['ID']
        url = str("https://neuroml-db.org/api/waveform?id=")+str(wid)
        waves = requests.get(url)
        temp = json.loads(waves.text)
        if temp['Spikes'] >= 1:
            rheobases.append(temp)
    if len(rheobases) == 0:
        return None

    in_current = []
    for check in in_current_filter:
        amp = get_waveform_current_amplitude(check)
        if amp < 0 * pq.nA:
            in_current.append(amp)
    rheobase_current = get_waveform_current_amplitude(rheobases[0])
    druckmann2013_standard_current = get_waveform_current_amplitude(currents[-2])
    druckmann2013_strong_current = get_waveform_current_amplitude(currents[-1])
    druckmann2013_input_resistance_currents = in_current
    model.waveforms = wlist
    model.protocol = {}
    model.protocol['Time_Start'] = currents[-2]['Time_Start']
    model.protocol['Time_End'] = currents[-2]['Time_End']
    model.inh_protocol = {}
    model.inh_protocol['Time_End'] = in_current_filter[0]['Time_End']
    model.inh_protocol['Time_Start'] = in_current_filter[0]['Time_Start']
    model.druckmann2013_input_resistance_currents = druckmann2013_input_resistance_currents

    model.rheobase_current = rheobase_current
    model.druckmann2013_standard_current = druckmann2013_standard_current
    model.druckmann2013_strong_current = druckmann2013_strong_current
    current = {}
    current['amplitude'] = rheobase_current
    model.vm_rheobase = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_standard_current
    model.vm15 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_strong_current
    model.vm30 = model.inject_square_current(current)
    current['amplitude'] = druckmann2013_input_resistance_currents[0]
    model.vminh =  model.inject_square_current(current)
    return model

def allen_format(volts,times):
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

    meaned_features_1 = {}
    skeys = [ skey for skey in spikes[0].keys() ]
    for sk in skeys:
        if str('isi_type') not in sk:
            meaned_features_1[sk] = np.mean([ i[sk] for i in spikes if type(i) is not type(str(''))] )
#

    allen_features = {}
    meaned_features_overspikes = {}
    for s in swp.sweep_feature_keys():# print(swp.sweep_feature(s))

        if str('isi_type') not in s:
            allen_features[s] = swp.sweep_feature(s)
            try:
                feature = swp.sweep_feature(s)
                if isinstance(feature, Iterable):
                    meaned_features_overspikes[s] = np.mean([i for i in feature if type(i) is not type(str(''))])
                else:
                    meaned_features_overspikes[s] = feature

            except:
                meaned_features_overspikes[s] = None #np.mean([i for i in swp.spike_feature(s) if type(i) is not type(str(''))])
                print(meaned_features_overspikes)
    for s in swp.sweep_feature_keys():
        print(swp.sweep_feature(s))

    #import pdb; pdb.set_trace()
    frame_shape = pd.DataFrame(meaned_features_1, index=[0])
    frame_dynamics = pd.DataFrame(meaned_features_overspikes, index=[0])
    meaned_features_1.update(meaned_features_overspikes)
    final_frame = pd.DataFrame(meaned_features_1, index=[0])

    return final_frame, frame_dynamics, allen_features


def three_feature_sets_on_static_models(model,debug = True, challenging=False):
    '''
    Conventions:
        variables ending with 15 refer to 1.5 current injection protocols.
        variables ending with 30 refer to 3.0 current injection protocols.
    Inputs:
        NML-DB models, a method designed to be called inside an iteration loop, where a list of
        models is iterated over, and on each iteration a new model is supplied to this method.

    Outputs:
        A dictionary of dataframes, for features sought according to: Druckman, EFEL, AllenSDK

    '''

    ##
    # wrangle data in preperation for computing
    # Allen Features
    ##


    times = np.array([float(t) for t in model.vm30.times])
    volts = np.array([float(v) for v in model.vm30])


    ##
    # Allen Features
    ##
    #frame_shape,frame_dynamics,per_spike_info, meaned_features_overspikes
    frame30, frame_dynamics, allen_features = allen_format(volts,times)
    #import pdb; pdb.set_trace()
    frame30['protocol'] = 3.0

    ##
    # wrangle data in preperation for computing
    # Allen Features
    ##

    times = np.array([float(t) for t in model.vm15.times])
    volts = np.array([float(v) for v in model.vm15])

    ##
    # Allen Features
    ##

    frame15, frame_dynamics, allen_features = allen_format(volts,times)


    frame15['protocol'] = 1.5
    allen_frame = frame30.append(frame15)
    #allen_frame.set_index('protocol')


    ##
    # Wrangle data to prepare for EFEL feature calculation.
    ##
    trace3 = {}
    trace3['T'] = [ float(t) for t in model.vm30.times.rescale('ms') ]
    trace3['V'] = [ float(v) for v in model.vm30]#temp_vm
    #trace3['peak_voltage'] = [ np.max(model.vm30) ]

    trace3['stim_start'] = [ float(model.protocol['Time_Start']) ]
    trace3['stimulus_current'] = [ model.druckmann2013_strong_current ]
    trace3['stim_end'] = [ trace3['T'][-1] ]
    traces3 = [trace3]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
    trace15 = {}
    trace15['T'] = [ float(t) for t in model.vm15.times.rescale('ms') ]
    trace15['V'] = [ float(v) for v in model.vm15]#temp_vm
    #trace15['peak_voltage'] = [ np.max(model.vm15) ]

    trace15['stim_start'] = [ float(model.protocol['Time_Start']) ]
    trace15['stimulus_current'] = [ model.druckmann2013_standard_current ]
    trace15['stim_end'] = [ trace15['T'][-1] ]
    traces15 = [trace15]# Now we pass 'traces' to the efel and ask it to calculate the feature# values
    ##
    # Compute
    # EFEL features (HBP)
    ##

    efel_results15 = efel.getFeatureValues(traces15,list(efel.getFeatureNames()))#
    efel_results30 = efel.getFeatureValues(traces3,list(efel.getFeatureNames()))#

    if challenging:
        efel_results_inh = more_challenging(model)


    df15 = pd.DataFrame(efel_results15)
    #import pdb; pdb.set_trace()
    df15['protocol'] = 1.5

    df30 = pd.DataFrame(efel_results30)
    df30['protocol'] = 3.0

    efel_frame = df15.append(df30)
    #efel_frame.set_index('protocol')


    ##
    # Get Druckman features, this is mainly handled in external files.
    ##
    DMTNMLO = dm_test_interoperable.DMTNMLO()
    DMTNMLO.test_setup(None,None,model= model)
    dm_test_features = DMTNMLO.runTest()

    dm_frame = pd.DataFrame(dm_test_features)
    if challenging:
        nu_preds = standard_nu_tests_two(DMTNMLO.model.nmldb_model)
    #import pdb; pdb.set_trace()

    if debug == True:
        ##
        # sort of a bit like unit testing, but causes a dowload which slows everything down:
        ##
        assert DMTNMLO.model.druckmann2013_standard_current != DMTNMLO.model.druckmann2013_strong_current
        from neuronunit.capabilities import spike_functions as sf
         _ = not_necessary_for_program_completion(DMTNMLO)
        print('note: False in evidence of spiking is not completely damning \n')
        print('a threshold of 0mV is used to detect spikes, many models dont have a peak amp')
        print('above 0mV, so 0 spikes using the threshold technique is not final')
        print('druckman tests use derivative approach')

        print(len(DMTNMLO.model.nmldb_model.get_APs()))

        print(len(sf.get_spike_train(model.vm30))>1)
        print(len(sf.get_spike_train(model.vm15))>1)


    return {'model_id':model.name,'efel':efel_frame,'dm':dm_frame,'allen':allen_frame}


def recoverable_interuptable_batch_process():
    '''
    Synposis:
        slower serial mode but debug friendly and simple
        Mass download all the NML model waveforms for all cortical regions
        And perform three types of feature extraction on resulting waveforms.

    Inputs: None
    Outputs: None in namespace, yet, lots of data written to pickle.
    '''
    all_the_NML_IDs =  pickle.load(open('cortical_NML_IDs/cortical_cells_list.p','rb'))

    mid = [] # mid is a list of model identifiers.
    for v in all_the_NML_IDs.values():
        mid.extend(v[0])
    path_name = str('three_feature_folder')
    try:
        os.mkdir(path_name)
    except:
        print('directory already made :)')
    try:
        ##
        # This is the index in the list to the last NML-DB model that was analyzed
        # this index is stored to facilitate recovery from interruption
        ##
        with open('last_index.p','rb') as f:
            index = pickle.load(f)
    except:
        index = 0
    until_done = len(mid[index:-1])
    cnt = 0
    ##
    # Do the batch job, with the background assumption that some models may
    # have already been run and cached.
    ##
    while cnt <until_done-1:
        for i,mid_ in enumerate(mid[index:-1]):
            until_done = len(mid[index:-1])
            model = get_static_models(mid_)
            if type(model) is not type(None):
            #if type(model) is not type(None):
                model.name = None
                model.name = str(mid_)
                three_feature_sets = three_feature_sets_on_static_models(model)
                with open(str(path_name)+str('/')+str(mid_)+'.p','wb') as f:
                    pickle.dump(three_feature_sets,f)
            with open('last_index.p','wb') as f:
                pickle.dump(i,f)
            cnt+=1

#import numpy as np

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
