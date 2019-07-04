
# coding: utf-8

# # Extracting features from the NeuroML-DB models

# In[1]:


import os
import glob
import pickle
import numpy as np
import sciunit


# ### Load model IDs

# In[2]:


# Get file path to pickled files containing NeuroML-DB IDs
model_id_file_paths = glob.glob("cortical_NML_IDs/*.p")

# Get that model list (done in `get_allen_features_from_nml_db.analyze_models_from_cache`)
models = []
for f in model_id_file_paths:
    models.append(pickle.load(open(f,'rb')))
    
# For example, here are 5 of the loaded model IDs
models[0]['markram'][0][:5]


# ### Load models from NeuroML-DB and cache them on disk

# The function `get_allen_features_from_nml_db.faster_make_model_and_cache()` can load the models from the database, extract the appropriate waveforms, and then cache the results.  It does so by first using `get_alen_features_from_nml_db.mid_to_model`, which extract and caches one whole model instance.

# In[3]:


# Use one model as an example
model_id = models[0]['markram'][0][:5][0]


# In[4]:

from get_three_feature_sets_from_nml_db import get_static_models, mid_to_model

# Here is one model instance
model = get_static_models(model_id)

standard_current = model.nmldb_model.get_druckmann2013_standard_current()
strong_current = model.nmldb_model.get_druckmann2013_strong_current()
volt15 = model.nmldb_model.get_waveform_by_current(standard_current)
volt30 = model.nmldb_model.get_waveform_by_current(strong_current)

current = {}
current['amplitude'] = standard_current
model.vm15 = model.inject_square_current(current)
assert np.mean(model.vm15) != 0
assert np.mean(model.vm15) == np.mean(volt15)

# Or just load the model and pickle it
os.makedirs('models', exist_ok=True)
mid_to_model(model_id)

# Now there is a model pickle file in the `models` directory
assert os.path.exists('models/%s.p' % model_id)


# ### Extract features from cached models

# The function `get_allen_features_from_nml_db.analyze_models_from_cache` does all this in parallel so it should be used to speed things up.  Ultimately it calls `get_allen_features_from_nml_db.model_analysis` on each model, which uses `get_allen_features_from_nml_db.three_feature_sets_on_static_models` and then saves the results.

# In[5]:


# Get file paths to pickled files containing the models that we loaded above
model_file_paths = glob.glob("models/*.p")


# In[6]:


# Unpickle one model from that list of file paths
with open(model_file_paths[0], 'rb') as f:
    model = pickle.load(f)

# It should be a SciUnit model
assert isinstance(model, sciunit.Model)

