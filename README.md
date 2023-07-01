# pg-gan

This branch is for local investigation and analysis. To operate the most active, up-to-date version of pg-gan, please go to https://github.com/mathiesonlab/pg-gan

## Mutation rate estimation
The branch `mut_estimation` in this repository (`local-pg-gan`) has a feature that allows the use of linear regression to estimate mutation rate, substituting the default constant value of 1.25e-8. This feature can be used to generate a function that predicts mutation rate from the number of SNPs (per bp) in a given region of real data, for a population, and can be validated with trials on simulated data, generated under known mutation rate values. To enable this feature, use the flag `--mut_est` when running the application, for example:
~~~
python3 pg_gan.py -m exp -p N1,N2,growth,T1,T2 -n 198 -d /data/mydata.h5 -b /data/mymask.bed -r /data/myrecos/ --mut_est
~~~
The note "using mutation rate estimation" will appear in your output, and the regression values will be printed in the initial regression generation, every time the simulation params are update, and after "bonus training" at the end of the trial. The printing will provide the R^2 value, the x value of the regression (the coefficient) and the y value (the constant term), for the function of SNPs per bp to mutation rate.
~~~
REGRESSION SCORE: 0.981966418233882     x: 7.235236174267772e-06        y: 1.0000495316311384e-10
~~~
Enabling this feature does result in an average 2% increase in runtime. It is likely that this feature increases the accuracy of the final parameter estimations, and statistics verifying this are in progress. The `summary_stats.py` program will detect if mutation rate estimation was in use in your trial and apply the final predicted regression to the simulated data.

Not necessarially for merging.