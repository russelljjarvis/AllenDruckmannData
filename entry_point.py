import glob
import sys
import get_three_feature_sets_from_nml_db as runnable_nml
import get_three_features_from_allen_data as runnable_allen

#import aligned_feature_extraction as runnable_nml_db
##
# The slow old way
# better for debugging
# uncomment
# runnable.recoverable_interuptable_batch_process()

##
# The faster way to complete everything when confident
##

#runnable_nml.faster_make_model_and_cache()
file_paths = glob.glob("models/*.p")
_ = runnable_nml.analyze_models_from_cache(file_paths)
#runnable_allen.faster_run_on_allen(20)
#runnable_allen.run_on_allen(22)

sys.exit(0)
