import glob
import sys
import get_three_feature_sets_from_nml_db as runnable_nml
import get_three_features_from_allen_data as runnable_allen

import glob
import pickle
import pandas as pd
import numpy as np
import glob
import os


##
# The slow old way
# better for debugging
# uncomment
#runnable_nml.recoverable_interuptable_batch_process()

##
# The faster way to complete everything when confident
##
#write_data()
#runnable_nml.faster_make_model_and_cache()
def cnt_check(cnt):
    if (10+cnt)>size:
        cnt+=1
    else:
        cnt += 10
    if cnt>= size:
        cnt = -1
    return cnt

def path_manipulation():
    cnt = 0
    viable_models = []
    size = len(glob.glob("models/*.p"))
    while len(viable_models)<10:
        temp_paths = list(glob.glob("models/*.p"))[cnt:10+cnt]

        models = [ pickle.load(open(f,'rb')) for f in temp_paths ]
        for p,m in zip(temp_paths,models):
            m.path = None
            m.path = p
        temp = [ m for m in models if not os.path.exists(str('three_feature_folder')+str('/')+str(m.name)+str('.p')) ]

        viable_models.extend(temp)
        cnt = cnt_check(cnt)
        if cnt == -1:
            return
    _ = runnable_nml.analyze_models_from_cache(viable_models)
    try:
        os.mkdir('completed_model')
    except:
        print('directory exists')
    for vm in viable_models:
        os.system(str('mv ')+str(vm.path)+str(' completed_model'))
    #print('exists cleanly')
    #exit(0)
    #return True


#additional_paths = glob.glob("data_nwbs/*.p")
other_paths = path_manipulation()
'''
runnable_allen.faster_run_on_allen_cached()

runnable_allen.faster_run_on_allen()
runnable_allen.faster_run_on_allen_cached()
'''
write_data()

print('exists cleanly')
sys.exit(0)
