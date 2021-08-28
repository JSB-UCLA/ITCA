#  Information-theoretic Classification Accuracy: A  Data-driven Criterion to Combining Ambiguous Outcome Labels in Multi-class Classification

# Author Contributions Checklist Form

## Data

The data  four applications as described in the manusrcipt Section 4 and the Supplementary Material. The processed data is located in the folder `../data/`. The size of the feature matrix `user_age_X.pkl` of application 3 exceeds the GitHub limit. The data can be downloaded via [the dropbox link here](https://www.dropbox.com/s/edhk8j4dmkxurz7/user_age_X.pkl?dl=0') or run `python download_files_for_application3.py`.

- **Application1**: `cov_casacolina.rds` contains the covariates matrix of the patients. `response_tot_casacolina.rds` contains the discharge FIM of the patients (class labels).
- **Application2**:  The csv file `gbm_tcga.csv` contains the covariates and the survival time. The data is also publicly avaiable at [cBioPortal](https://www.cbioportal.org/).   
- **Application3**:  `user_age_X.pkl` and `user_age_Y.pkl` contain the covariates and the class labels, respectively. The data is available at the [Kaggle challenge page ](https://www.kaggle.com/c/talkingdata-mobile-user-demographics). 
- **Application4**:  `Hydra_40pcs.rds` contains the 40 principal components and `Hydra_y.rds` contains the class labels.

The data are given and explained in the text of Section 4 of the manuscript. The Supplementary Material desribes the preprocessing of the data. 

## Code
The ipython notebooks for "Information-theoretic Classification Accuracy: A  Data-driven Criterion to Combining Ambiguous Outcome Labels in Multi-class Classification", to reproduce the results in the manuscript, is available from  https://github.com/JSB-UCLA/ITCA/tree/main/notebooks. 
- `simulation_studies.ipynb`: generate the tables and figures in Section 3. 
- `application1_prognosis_of_TBI.ipynb`: generate the figures of Section 4.1.
- `application2_GBM_survival.ipynb` : generate figures of Section 4.2.
-  `application3_prediction_of_user_demographics.ipynb`: generate Figure 7 and Table 7.
-  `application4_hydra_single-cell.ipynb` : generate  Table 8, Figure 9 and Figure 10.
-  `theoretic_remarks.ipynb`: generate Figure 11 and Figure 12 of the Appendix.

The analysis and presentation used Python version 3.8.8 and the following Python package versions:
numpy==1.16.5, lifelines==0.26.0, xgboost==1.4.2, pandas==1.1.3, matplotlib==3.1.1, seaborn == 0.9.0, itca==0.1.0, scikit-learn==0.24.1, pyreadr==0.4.2, torch==1.9.0

Please check the `requirements.txt` and ensure that all the packagtes are installed properly.  Please refer to [ITCA documentation](https://messcode.github.io/ITCA/) for detailed guides of the use ITCA.

## Instruction for use
To reproduce the reusults, please run each ipython notebooks cell by cell.