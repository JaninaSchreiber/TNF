testcase
--------

In this directory are the scripts and data of the testcases 


The project consists of the following directories:

```
 scripts/             # scripts
 sixhumpcamelback/    # data 
 griewank/            # data
 sixhumpcamelbacktime # data produced without 'print' statements to compare time
 
```

The scripts directory is structured as follows:

```
 scripts/        
	algorithm/
	data/
	evaluation/
	plots/ 
```

In 'algorithm' is the code for running the parameter space exploration 
optimization located.

HNS and IHR can be run with the following command

	 cd scripts/algorithm/
	 python bas.py >problem< >m_max< >strategy< >cooling_schedule<
	 
with:
	problem : 'shcb' , 'griewank'
	m_max   : any integer
	strategy: 'improving-hit-and-run', 'hide-and-seek'
	cooling_schedule: 'adaptive', 'geometric'
	    

To run Random Search: 

	 cd scripts/algorithm/
	 python random_search.py >problem< >m_max< 
	 
	 
The scripts for preprocessing and plotting the data
are in the directories

```
	scripts/
		preprocessing/
		plots/
```
	
To plot the data, execute the code 'preprocessing'. But these results
are also saved in the 
```
	scripts/
	    data/
```
directory.



