[![Build Status](https://travis-ci.org/russelljjarvis/AllenEFELDruckmanData.png)](https://travis-ci.org/russelljjarvis/AllenEFELDruckmanData)


##
# Apply 3 feature extraction protocols to the same NeuroML-DB static waveforms.
##

A notebook is defined that acts on loaded per static modle pickle files.
Each file name is determined by an allen cell id.

The contents of each file is combined Allen and Druckmann features, on Allen Data waveforms (from real cells).
Each pickle file consists of Druckmann features, and Allen Features on Allen Data.

There is another single pickle file that specifies cell model ids to be used.

The folder three_feature_folder contains outputs.
