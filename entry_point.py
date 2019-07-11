import glob
import sys
import get_three_feature_sets_from_nml_db as runnable_nml
import get_three_features_from_allen_data as runnable_allen

import glob
import pickle
import pandas as pd
import numpy as np
import glob


##
# The slow old way
# better for debugging
# uncomment
#runnable_nml.recoverable_interuptable_batch_process()

##
# The faster way to complete everything when confident
##

#runnable_nml.faster_make_model_and_cache()
#file_paths = glob.glob("models/*.p")
#_ = runnable_nml.analyze_models_from_cache(file_paths)

#runnable_allen.faster_run_on_allen(150)
#_ = runnable_nml.analyze_models_from_cache(file_paths[0:2])
runnable_allen.faster_run_on_allen_revised()
#runnable_allen.faster_run_on_allen_revised()

file_paths = glob.glob("three_feature_folder/*.p")
nml_data = []

for f in file_paths:
    nml_data.append(pickle.load(open(f,'rb')))


file_paths = glob.glob("allen_three_feature_folder/*.p")
allen_analysis = []
for f in file_paths:
    allen_analysis.append(pickle.load(open(f,'rb')))
path=str('new_dir')

try:
    import os
    os.mkdir(path)
except:
    pass
merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=True,other_dir=path)
merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=False,other_dir=path)
print('so it finished')

import os
merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=True,other_dir=os.getcwd())
merged = runnable_nml.giant_frame(allen_analysis,nml_data,onefive=False,other_dir=os.getcwd())
print('exists cleanly')
sys.exit(0)
