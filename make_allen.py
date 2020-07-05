from make_allen_tests import AllenTest
nml_data = []
import pickle
import glob
from sciunit import TestSuite
from neuronunit.optimisation.optimization_management import TSD

#file_paths = glob.glob("three_feature_folder/*.p")
#for f in file_paths:
#ickle   nml_data.append(pickle.load(open(f,'rb')))
file_paths = glob.glob("allen_three_feature_folder/*.p")
allen_analysis = []
allen_test_suites = []
for f in file_paths:
    contents = pickle.load(open(f,'rb'))
    allen_tests = []
    #print(contents['model_information'],contents.keys())
    if contents['allen_15'] is not None:
        for k,v in contents['allen_15'].items():
            at = AllenTest(name='15'+str(k))
            at.set_observation(v)
            allen_tests.append(at)
    if contents['allen_30'] is not None:
        for k,v in contents['allen_30'].items():
            at = AllenTest(name='30'+str(k))
            at.set_observation(v)
            allen_tests.append(at)

    if contents['efel_15'] is not None:
        for k,v in contents['efel_15'][0].items():
            at = AllenTest(name='15'+str(k))
            at.set_observation(v)
            allen_tests.append(at)
    if contents['efel_30'] is not None:
        for k,v in contents['efel_30'][0].items():
            at = AllenTest(name='30'+str(k))
            at.set_observation(v)
            allen_tests.append(at)
    if contents['dm'] is not None:
        for k,v in contents['dm'].items():
            at = AllenTest(name='dm'+str(k))
            at.set_observation(v)
            allen_tests.append(at)
        
    #, 'efel_30', 'allen_30', 'model_id', 'model_information', 'dm', 'allen_15'])   
    #print(contents.keys())
    suite = TestSuite(allen_tests,name=str(contents['model_id']))
    allen_test_suites.append(suite)
    
    print(TSD(suite))
    pickle.dump(allen_test_suites,open('allen_NU_tests.p','wb'))
    


    
