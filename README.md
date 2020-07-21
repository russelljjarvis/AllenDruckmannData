[![Build Status](https://travis-ci.org/russelljjarvis/AllenEFELDruckmanData.png)](https://travis-ci.org/russelljjarvis/AllenEFELDruckmanData)

https://russelljjarvis.github.io/AllenEFELDruckmanFeatures/index.html
##
# Unify three different feature extraction protocols.
# Apply the knitted features to NeuroML-DB model waveform and also Allen and BBP experimental Recordings.
##

This repository contains:
a script that downloads NML-DB waveform data, and uses this data to instantiate appropriate NML-static models to do feature extraction with using three of the biggest feature extraction suites.

A notebook is defined that acts on loaded per static modle pickle files.
Each file name is determined by an allen cell id.

The contents of each file is combined Allen and Druckmann features, on Allen Data waveforms (from real cells).
Each pickle file consists of Druckmann features, and Allen Features on Allen Data.

There is another single pickle file that specifies cell model ids to be used.

The folder three_feature_folder contains outputs.


The main method that does the aligned feature extraction is called
```def three_feature_sets_on_static_models```

 I build the docker image with the name russelljarvis/efel_allen_dm, meaning that the command
 ```
 docker pull russelljarvis/efel_allen_dm 
 ```
 should work. This uses the docker file in this directory.
I build it with the name russelljarvis/efel_allen_dm.
 and launch it with this alias.
```
alias efel='cd /home/russell/outside/neuronunit; sudo docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/russell/outside/neuronunit:/home/jovyan/neuronunit -v /home/russell/Dropbox\ \(ASU\)/AllenDruckmanData:/home/jovyan/work/allendata russelljarvis/efel_allen_dm /bin/bash'
```


* This is how my travis script builds and runs:
* before_install:
 - docker pull russelljarvis/efel_allen_dm
 - git clone -b barcelona https://github.com/russelljjarvis/neuronunit.git

* script:
* show that running the docker container at least works.
  - docker run -v neuronunit:/home/jovyan/neuronunit russelljarvis/efel_allen_dm python /home/jovyan/work/allendata/small_travis_run.py
