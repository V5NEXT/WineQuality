# AMLS_SS22

Group members: Johannes Honeder, Samir Kumar

## Data Preparation

To explore and prepare the dataset, the first step was to check if the data is clean and complete. Since there were no missing values present, we did not need to impute any missing values. 
We checked out the nature of the features by making different plots, including distribution plots, scatterplots and boxplots for every feature (see AMLS_SS22/plots/data_exploration). 
Next, we plotted a correlation heatmap to get the relations between the different features, and also the relation to the target features (wine type, quality). 
We also computed summary statistics for every feature, including mean, median, min/max values. Since the dataset looked to be very clean, we did not do any data cleaning at all.

The last part in our exploration was to check the balance between the different output classes. The first thing to mention here is that the ratio of instances of red wine and white wine are not very balanced. Also, the quality classes are not balanced at all (for example, there are no instances of red wine with quality 9). We handled this in later stages (see Tuning). 

We also transformed the data to a lower dimensional space (with TSNE), but the generated plots were not really useful (see AMLS_SS22/plots/data_exploration/tsne*)


## Modelling

We defined two models, a classifier for the wine type prediction as well as a regressor for the wine quality prediction.
For the classifier, we went with a simiple DecisionTreeClassifier, which gave very good results without tuning any parameters or tuning the data, that's why we didn't consider any further hyperparameter tuning for this subtask. The final results (including accuracy, AUC, confusion matrix and classification report) can be found in AMLS_SS22/evaluation/classification_simple_evaluation.png.

For the regressor, we tried out different models, including RandomForest, LassoRegressor and LinearRegressor (see AMLS_SS22/plots/evaluation/regression_simple_comparison.png). We also tried to find the best model with the AutoSklearn library. Since the RandomForestRegressor performed best (also considering the complexity of the resulting model), we implemented a Grid search to fine-tune several parameters (including number of trees, the number of features to consider for the best split and whether to use bootstrap sampling for building the trees). The final model parameters for the RandomForestRegressor can be found in AMLS_SS22/grid_search_evaluation.png.


## Tuning

In the Tuning step, we tried different data transformations to boost the performance of our model. We considered dropping highly-correlated featuers and binning several features (including "handcrafted" binning functions and also general binning with qcut).

Another step for solving the balance issue with the data, we generated new data from the given data via SMOTE (Synthetic Minority Oversampling Technique). Other data augmentation techniques (e.g. VAE) were not feasible due to the lack of data samples. We both tried out to balance the red and white wine samples as well as the different quality classes. In the end, we only balanced the quality data classes, since balancing red and white wine samples did not improve our score. In the end, the dataset was still not perfectly balanced, but the performance on the minority classes (quality 3, 8, 9...) improved significantly (see Model Debugging). 
In the end, the best performance was reached when binning 'residual sugar' and 'chloride' with the handcrafted binning functions (to see all results from our data tuning evaluation, have a look at AMLS_SS22/plots/tuning_evaluation).

## Model Debugging

In this section, we had a closer look at the obtained results and how the model performed on different quality classes. For example, we compared how the model performs to predict different quality types, once for the normal data and once for the balanced data. We found that balancing the dataset gave much better results at the underrepresented classes, even though the error increased a little for the majority classes (see AMLS_SS22/plots/evaluation/normal_balanced_evaluation.png). 

We also provided explanations for the model with the SHAP library. We generated plots to explain the impact of different features on the Classifier and the Regressor. We also generated plots to explain the prediction of single data samples (see AMLS_SS22/plots/explanation). 


## Parallelization

The last part was to build the whole ML pipeline with the found data tuning steps and models. We built the pipeline to be able to run it both in sequential and parallel mode. The pipeline consists of generating and transforming the data, building the models, find the best model parameters for the regression, and evaluating the results. Since plotting multiple explanation plots did not work in our pipeline, we excluded it from our pipeline (but you can see the explanation plots in AMLS_SS22/plots/explanation).

In the end, the sequential mode had a runtime of of ~204 seconds. The parallel mode with 5 cores had a runtime of ~27.5 seconds (see AMLS_SS22/plots/final_run). Increasing th e number of cores to 10 just improved the runtime by 2s.


## Running the code

There are basically two interesting modes to run the source code. To run the data exploration, navigate to AMLS_SS22/src and execute "python3 05_Paralell.py explore", which will generate plots and the summary statistics. 

To run the ML pipeline, navigate to AMLS_SS22/src and execute either "python3 05_Paralell.py seq" (for sequential mode) or "python3 05_Paralell.py par 5" (for paralell mode, where 5 indicates the number of cores that should be used). 


## Scoring and evaluation

The final scores can be found in AMLS_SS22/plots/final_run/. We reached a classification accuracy of 0.991 for predicting the wine type and a MSE of 0.345 for predicting the wine quality.



