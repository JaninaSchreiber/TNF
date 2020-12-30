EMEWS project template
-----------------------

You have just created an EMEWS project.

This project is compatible with swift-t v. 1.3+. Earlier
versions will NOT work.

The project consists of the following directories:

```
  data/
  ext/
  etc/
  scripts/
  swift/
  model/
  README.md
```
The directories are intended to contain the following:

 * `data` - model input etc. data
 * `etc` - additional code used by EMEWS
 * `experiments` - results of the runs
 * `scripts` - any necessary scripts (e.g. scripts to launch a model), excluding
    scripts used to run the workflow.
 * `swift` - swift code
 * `results` - scripts to analyse the output of the model exploration process
 * `model` - should contain the model itself

Note: In contrast to the cancer-immune model, this project structures 
a bit different. The PhysiCell model is in the "data/" folder
 * `TNF/data/PhysiBoSSv2`

Before running the algorithms make sure EMEWS is installed:
	https://emews.github.io/
	
It is recommended to use the Spack package manager
	https://spack.readthedocs.io/en/latest/
	
The TNF repository can be downloaded from
	https://github.com/xarakas/spheroid-tnf-v2-emews/
	
 
To run the model exploration methods go to:

	cd TNF/swift/

Open the file
	
	swift_run_bas.sh  # for HNS and IHR
	swift_run_rs.sh   # for PRS
	
and change it according to your needs. You can set up the optimization strategy, 
cooling schedule, number of iterations. Moreover, the maximum Walltime, 
nodes, etc can be chosen. The script is commented and thus self explaining.

Finally, run

	./swift_run_bas.sh <experimentname> ../data/tnf_params.json

or 

	./swift_run_rs.sh <experimentname> ../data/tnf_params.json

The "swift_run_bas.sh" starts the swift code in "swift_run_bas.swift". 
It communicates via Queues with the model exploration method in Python 
and runs the simulations. 

The bounds of the parameter values can be chosen in the folder
	
	data/ 

The model exploration methods are implemented in the folder 
	
	python/            (here: bas_box.py and randomsearch.py)
	
	
The results are stored in the folder

	experiments/
