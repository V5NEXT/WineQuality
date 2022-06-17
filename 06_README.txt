# AMLS_SS22

Vishnu Viswambharan (Had to do project alone as I couldnt find any partners to start with)

## 01_DataPrep

To explore and prepare the dataset, the first step was to check if the data is clean and complete. Since there were no missing values present, I did not need to impute any missing values. 
Then checked out the nature of the features by making different plots, including skwness measure for each feature (see Project/plots/01_DataPrep). 
Next, I plotted a correlation heatmap to get the relations between the different features, and also the relation to the target features (wine type, quality). 
I also computed summary statistics for every feature, including mean, median, min/max values. Since the dataset looked to be very clean, I did some basic preprocessing only

The last part in my exploration was to check the balance between the different output classes. The first thing to mention here is that the ratio of instances of red wine and white wine are not very balanced. Also, the quality classes are not balanced.
The categotical specific plots where also measured to see the difference and is saved in (Project/plots/01_DataPrep)

I further divided the dataset into train and test set. Then further divided the tain set into validation set.



## 02_Model

There are defined two models used, a classifier for the wine type prediction and a regressor for the wine quality prediction.
For both regression and classification I used neural networks( there where many tutorials for using a predefined models but opted for neural networks for exploring the capabilites)

Classfication Model: I constructed a base model with 1 Hidden Layer and a input and output layer with activation as relu and sigmoid for the output.
I passed all the params required for tuning in a sequential pipeline (except for additional Layers)

Regression Model : Regression Base model I started with linear model with a single layer and calculated the Val_MSE for getting an idea to which direction to tune.


## 03_Tuning

In the Tuning step, I tried different data transformations to boost the performance of the model. 

#Classfication : I changed hyperparameters EPOCHS, Layers, Optimizers and number of nodes and measured the validation loss among them to determine
them. In the 03_Tuning.py you can see 5 relavant model architectures build with Final_Model providing the best result. Below are the validation losses for the models.
(In the code, the plot is commented out so as to measure the pipeline, Please find the val_loss plot at (Project/plots/03_Tuning))
Validation Losses for different Approaches :
Changing Number of Layers (4): 0.17088524997234344
Changing Number Of Nodes(on Base Model): 0.1869492381811142
Changing Batch Sizes : 0.9273497462272644
Changing Optimizers : 0.2500722110271454
Final Model:0.1527889221906662

#Regression : I started from a base model and build on it comparing each approaches with the validation loss. Since I dealt with Deep Learning Models, it meant substantially more runtime.
I used differnt representaion models (1,2,3), changed optimizers, added callbacks and dropouts and generated the best model Final_Model_regression after batch normalizing .
(In the code, the plot is commented out so as to measure the pipeline, Please find the val_loss plot at (Project/plots/03_Tuning))

Validation Losses of Different Approaches :

One Representation Model : 0.5211682915687561
Two Representation Model : 0.521885335445404
Three Representation Model : 0.5249411463737488
Optimized Model : 0.46790364384651184
Adding Callbacks with Validation Set : 0.4751804769039154 (not a seperate Model but a method to achieve the final model)
After Adding Dropout : 0.47221213579177856
Batch Normalization : 0.4742
Final Model : 0.26413536071777344 (Loss , Final Model is streamlined with al above best features)

## 04_Debug

Debugging was done for both Classfication and Regression Model and the results are Below:
The Final Classification Model Results :
Accuracy : 95.15384615384616
Balanced Accuracy :  91.25577909071441
Confusion Matrix [[965   9]
 [ 54 272]]
Classification Report               precision    recall  f1-score   support

           0       0.95      0.99      0.97       974
           1       0.97      0.83      0.90       326

    accuracy                           0.95      1300
   macro avg       0.96      0.91      0.93      1300
weighted avg       0.95      0.95      0.95      1300

Average Precision 0.8491711493536701

The Final Regression Model Results :

Final Regression model score (R2): 0.3213893003869255
Final Regression model score (MAE): 0.5229203370901254
Final Regression model score (MSE): 0.4990976907809902
Residual Error 3.002124309539795







## Parallelization

The last part was to build the whole ML pipeline with the found data tuning steps and models. I built the pipeline to be able to run it both in sequential and parallel mode.
But the paralell preprocessing was not successful, so omitted it.

The Time for Sequential Execution : 

Sequential Execution time 897.229100227356s
Sequential Execution time Classification 34.427135944366455s
Sequential Execution time Regression 862.8019642829895s



## Running the code

To run the code(through sequential) head on to Project Folder and run python .\05_Parallel.py
(the plots are commented) to check the run time.

To run the visualization part run python .\01_DataPrep.py (after uncommenting function evaluating_dataset())

## Scoring and evaluation

We reached a classification accuracy of 0.9515384615384616 for predicting the wine type and a MSE of 0.4726 for predicting the wine quality.



