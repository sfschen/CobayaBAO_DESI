# CobayaBAO_DESI
Repository with Fitting Code for BAO as part of DESI KP4.

The basic call is something like:

srun -n 8 -c 8 cobaya-run martin_recsym.yaml

The details of what likelihood/model (i.e. recsym vs. reciso, what degree and size of polynomial terms used, etc.) are specified in the yaml files. Currently a "martin_recsym.yaml" and "martin_reciso.yaml" are provided as examples. To make the equivalent for other data sets, go into "lss_likelihodd/mock_likelihoods" and add a class for a particular dataset inheriting the basic BAO class, e.g.

class Martin_RecSym(RecSymLikelihood):
    pass

and add a yaml file with the requisite data paths in "lss_likelihood". Then, for a specific run setup given a data set (i.e. scale cuts, priors etc.) make a yaml file wherever and run it using "cobaya-run" as above. On one core at nersc these runs take about 15 minutes without optimizing for the best-fit before the run.

In principle all one needs to do to run the code is (after installing cobaya) adding custom paths in the various yaml files. More details to come...
