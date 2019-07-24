import pickle
import numpy as np
from allensdk.model.biophysical import runner
import json
from allensdk.ephys.ephys_extractor import EphysSweepSetFeatureExtractor
from scipy.interpolate import interp1d

typical_wave = pickle.load(open('typical_wave.p','rb'))
xnew = np.linspace(0, float(np.max(typical_wave.times)), num=len(typical_wave.times), endpoint=True)

with open('manifest.json', 'r') as content:
  cell_d = json.load(content)
print(cell_d)
description = runner.load_description('manifest.json')
(vm,time) = runner.run(description)
#runner.extract_cell_features.mean_features_spike_zero(vm)

transform_function = interp1d([float(t) for t in vm],[float(v) for v in time])
print(len(xnew))
print(len(vm))
import pdb
pdb.set_trace()
vm_new = transform_function(xnew) #% generate the y values for all x values in xnew
print(vm_new)
#print(len(vm_new))
#self.vM = AnalogSignal(vm_new,units = mV,sampling_period = float(xnew[1]-xnew[0]) * pq.s)



def allen_format(volts,times,optional_vm=None):
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
    if optional_vm is not None:


        spike_train = threshold_detection(optional_vm, threshold=0)

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
    return meaned_features_overspikes, allen_features
try:
  (a,b) =allen_format(vm_new,time)
except:
  import pdb
  pdb.set_trace()
print(a,b)
