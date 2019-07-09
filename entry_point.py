import glob
import sys
import get_three_feature_sets_from_nml_db as runnable_nml
import get_three_features_from_allen_data as runnable_allen
#sys.exit(0)

import glob
import pickle
import pandas as pd
#from sklearn.cross_decomposition import CCA
import numpy as np
import glob


#import aligned_feature_extraction as runnable_nml_db
##
# The slow old way
# better for debugging
# uncomment
#runnable_nml.recoverable_interuptable_batch_process()

##
# The faster way to complete everything when confident
##

runnable_nml.faster_make_model_and_cache()
file_paths = glob.glob("models/*.p")
_ = runnable_nml.analyze_models_from_cache(file_paths)
runnable_allen.faster_run_on_allen(200)
#import pdb
#pdb.set_trace()
#runnable_allen.run_on_allen(150)
#file_paths = glob.glob("models/*.p")
#_ = runnable_nml.analyze_models_from_cache(file_paths[0:1])
#runnable_allen.faster_run_on_allen(20)
#import pdb
#pdb.set_trace()
#runnable_allen.run_on_allen(1)



file_paths = glob.glob("three_feature_folder/*.p")
nml_data = []

for f in file_paths:
    nml_data.append(pickle.load(open(f,'rb')))
print(nml_data)

#print(nml_data[0]['dm'].columns)
#print(nml_data[0]['efel'].columns)
#print(nml_data[0]['allen'].columns)


file_paths = glob.glob("allen_three_feature_folder/*.p")
allen_analysis = []

for f in file_paths:
    allen_analysis.append(pickle.load(open(f,'rb')))


merged = runnable_nml.giant_frame(allen_analysis,nml_data)
merged.to_csv('index_by_id.csv', sep='\t')
