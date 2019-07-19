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

from get_three_features_from_allen_data import get_static_models_allen
import dask.bag as db # a pip installable module, usually installs without complication

def cnt_check(cnt,size):
    if (8+cnt)>size:
        cnt+=1
    else:
        cnt += 8
    if cnt>= size:
        cnt = -1
    return cnt

def path_manipulation_orig():
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
        cnt = cnt_check(cnt,size)
        if cnt == -1:
            return
    _ = runnable_nml.analyze_models_from_cache(viable_models)
    try:
        os.mkdir('completed_model')
    except:
        print('directory exists')
    for vm in viable_models:
        os.system(str('mv ')+str(vm.path)+str(' completed_model'))
    return

#additional_paths = glob.glob("data_nwbs/*.p")
#_ = path_manipulation()
def build_iterator():
    cnt = 0
    viable_models = []
    size = len(glob.glob("models/*.p"))
    while len(viable_models)<9:
        temp_paths = list(glob.glob("models/*.p"))[cnt:10+cnt]
        models = [ pickle.load(open(f,'rb')) for f in temp_paths ]
        for p,m in zip(temp_paths,models):
            m.path = None
            m.path = p
        temp = [ m for m in models if not os.path.exists(str('three_feature_folder')+str('/')+str(m.name)+str('.p')) ]
        viable_models.extend(temp)
        cnt = cnt_check(cnt,size)
        if cnt == -1:
            return None
    return viable_models
    
def new():
    viable_models = 1.0
    while viable_models is not None:
        viable_models = build_iterator()
        _ = runnable_nml.analyze_models_from_cache(viable_models)
        try:
            os.mkdir('completed_model')
        except:
            print('directory exists')
        for vm in viable_models:
            print(str('mv ')+str(vm.path)+str(' completed_model'))
            os.system(str('mv ')+str(vm.path)+str(' completed_model'))
    return

def build_allen_models():
    '''

    '''
    cnt = 0
    viable_models = []
    size = len(glob.glob("data_nwbs/*.p"))
    while cnt!=-1:
        temp_paths = list(glob.glob("data_nwbs/*.p"))[cnt:8+cnt]

        pre_models = [ pickle.load(open(f,'rb')) for f in temp_paths ]
        pre_models = [ (m[0],m[1],m[2]) for m in pre_models if not os.path.exists(str('models')+str('/')+str(m[2])+str('.p')) ]
        try:
            pre_models_bag = db.from_sequence(pre_models,npartitions=8)
            list_of_nones = list(pre_models_bag.map(get_static_models_allen).compute())#(pre_models)
        except:

        list_of_nones = [l for l in list_of_nones if l is not None ]
        file_paths_to_broken = [l for l in list_of_nones if l[1] is False ]
        try:
            os.mkdir('broken_nwbs')
        except:
            print('directory exists')
        for l in file_paths_to_broken:
            print(str('mv ')+str('data_nwbs/')+str(l[0])+str('.p')+str(' broken_nwbs'))
            os.system(str('mv ')+str('data_nwbs/')+str(l[0])+str('.p')+str(' broken_nwbs'))



        cnt = cnt_check(cnt,size)
        if cnt == -1:
            return

build_allen_models()
new()
runnable_nml.write_data()
