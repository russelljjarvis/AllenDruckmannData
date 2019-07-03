import get_allen_features_from_nml_db as runnable

##
# The slow old way
# better for debuggin
##
runnable.recoverable_interuptable_batch_process()

##
# The faster way to complete everything when confident
##
runnable.faster_make_model_and_cache()
file_paths = glob.glob("models/*.p")
_ = runnable.analyze_models_from_cache(file_paths)
