# atm233-midterm-winter2020

### For a single run
Run "soil_heat_main.py" after changing the "constants.py" file to the parameters you want to use. Data will be output to the "/data/single_runs" folder, with file names indicating the main parameters used to create the run. Preliminary figures are also output to the "/figures/output/single_runs" directory following the same naming convention. Both of these folders will be created in the program if they don't already exist.

### For multiple runs
Run "soil_heat_main_loop.py" after changing the "constants.py" file to the parameters you want to initially have sent to the program (namely  "kap" for the soil heat conductivity). In the main program itself, you'll want to change the list in each for loop to the values you want to loop through. Naming conventions follow the output from "soil_heat_main.py", but subfolders will be named based on dz values first, and dt values second.

### TODO
- Update "soil_testbed.ipynb" to include troubleshooting we want to do.
- Look at different "kap" values to see if model blows up when we change the parameter.
- See if slope of analytical solution matches what we get in the code.
