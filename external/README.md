# Existing baseline methods and environments
To avoid dependency issues, each of the baseline methods has a separate environment (Docker, conda, pipenv).  
In this repository, each of the baseline methods is independent from the evaluation process.  
To run the existing baselines we chose, follow the instruction in each directory.

- [gplearn](./gplearn)
- [AFP](./ellyn)
- [AFP-FE (same dir as AFP)](./ellyn)
- [AI Feynman](./ai_feynman2)
- [DSR](./dso)

Once you finish training and obtaining the estimated equations, you will follow the evaluation instruction 
[here](https://github.com/omron-sinicx/srsd-benchmark#compute-edit-distance-between-estimated-and-ground-truth-equations)
to evaluate your model.

## Your own model

You will write a script to 
1. load tabular datasets such as [our SRSD datasets](https://github.com/omron-sinicx/srsd-benchmark#download-or-re-generate-srsd-datasets)
2. train your model on each of the datasets
3. choose the best model per dataset (e.g., based on regression errors on validation dataset after hyperparameter tuning)
4. dump the estimated symbolic expression (equation)
  
As a starting point, we suggest that you make a copy of `gplearn/` folder and edit the project to work with your own model.  
`gplearn/gp_runner.py` is a minimal script to execute Steps 1 - 4 above for gplearn baseline.  
  
Once you estimate equations from the datasets, you will follow the evaluation instruction [here](https://github.com/omron-sinicx/srsd-benchmark#compute-edit-distance-between-estimated-and-ground-truth-equations) 
and evaluate your model on the true equation and/or test dataset, using the estimated equation obtained at Step 4.
