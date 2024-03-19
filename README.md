# deep-learning-challenge


Alphabet Soup, a nonprofit foundation, seeks a tool to aid in selecting applicants for funding with the highest potential for success. Leveraging machine learning and neural networks, the task involves developing a binary classifier based on features within a provided dataset. This classifier aims to predict the likelihood of an applicant's success if funded by Alphabet Soup.

Tools to help Alphabet Soup Foundation:

* Create a new repository for this project called deep-learning-challenge. Do not add this Challenge to an existing repository.

* Clone the new repository to your computer.

* Inside your local git repository, create a directory for the Deep Learning Challenge.
Push the above changes to GitHub.

* Module 21 files

# Step 1: Preprocess the Data

* Upload the starter file to Google Colab and follow provided instructions for preprocessing steps.

* Read charity_data.csv into a Pandas DataFrame.

* Identify target variable(s) and feature(s) for the model, dropping EIN and NAME columns.

* Determine unique values for each column.

* For columns with more than 10 unique values, determine data points for each unique value.

* Bin "rare" categorical variables into a new value, "Other," and verify success.

* Encode categorical variables using pd.get_dummies().

* Split preprocessed data into features array (X) and target array (y) using train_test_split.

* Scale training and testing features datasets using StandardScaler instance, fitting it to the training data, then transforming the data.

# Step 2: Compile, Train, and Evaluate the Model

 * Design a neural network model using TensorFlow for binary classification based on dataset features.
    
 * Determine the number of input features and nodes for each layer.
    
 * Create the first hidden layer with an appropriate activation function.
   
 * Optionally add a second hidden layer with an appropriate activation function.
 
 * Create the output layer with an appropriate activation function.
    
 * Verify the structure of the model.
   
 * Compile and train the model, saving weights every five epochs using a callback.
    
 * Evaluate the model using test data to calculate loss and accuracy.
   
 * Save results to an HDF5 file named AlphabetSoupCharity.h5.

# Step 3: Optimize the Model

  * Adjust input data by dropping or creating more bins for rare occurrences in columns to reduce confusion.
    
  * Experiment with adding more neurons or hidden layers, using different activation functions for hidden layers.
    
  * Modify the number of epochs during training.
    
  * Create a new Google Colab file named AlphabetSoupCharity_Optimization.ipynb.
    
  * Import dependencies and read charity_data.csv into a Pandas DataFrame.
    
  * Preprocess the dataset as in Step 1, considering modifications for optimization.
    
  * Design a neural network model, adjusting for modifications to achieve over 75% accuracy.
    
  * Save and export results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.
  
# Step 4: Write a Report on the Neural Network Model

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis.

## Results

Using bulleted lists and images to support your answers, address the following questions:

## Data Preprocessing

What variable(s) are the target(s) for your model?

What variable(s) are the features for your model?

What variable(s) should be removed from the input data because they are neither targets nor features?

## Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

Were you able to achieve the target model performance?

What steps did you take in your attempts to increase model performance?

## Summary: 
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

# References 

IRS. Tax Exempt Organization Search Bulk Data Downloads. https://www.irs.gov/

*Thanks to the LA who helped in on the line about replacing the classification for binning. Mentioned I was using the wrong variable which is why I got the error. 

*I had to research the types of layers, activations, checkpoints/call backs and overall get the idea of Keras. Those links are in the Starter copy notebook.

*Created a copy since I forgot to create checkpoints/call backs on the first doc. I kept the original since I wanted to compare if creating checkpoints affects the accuracy of the model. 
