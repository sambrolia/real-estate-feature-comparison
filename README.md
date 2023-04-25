# Task 2: DC Real Property Assessment Data Analysis

This project analyzes the DC Real Property Assessment Data to find the most important feature with respect to the 2014 assessment, and visualize the feature

## NOTE ON IMPROVEMENTS: 
After finding and fixing some bugs in the code it seems that the first 4 digits of the ssl are a better feature than the combined street address I had created.  

I further validated this by running a gradiant boosting model to validate the findings of the random forest.  

I have made some further improvements to the code such as refactoring into prepare, model, and visualize files to aid readability.


## Approach - see note above

On first glance at the dataset it was clear that for some entries the address would provide quite accurate predictions 
where entries existed for similar addresses. However, the address field was not formatted in a regular way.

Taking the easy to parse categorical features first, I used one_hot encoding and quickly found the use_code to be best
of those features. However, once I correctly parsed the ssl into two seperate numbers I found the first part to
significantly out perform use_code. 

These still felt limited in how far they could improve with additional data so I decided to use the address field 
to create a new feature called combined_street_info. 
This feature combined a few details parsed from address to get decent result but still not quite as strong as use_code.  

It occured to me that multiple streets could have the same names so I added neighborhood to the feature to make it more unique.  
With this the new feature drastically outperformed the use_code feature as more data was trained on. 

Although neighborhood was needed to get this performance, I think its still fair to say that address is more important than use_code. 

As well as becoming stronger and stronger the more data collected, it also benefits from having sufficient detail to 
look up the property online and get additional information to further improve predictions.

Address also can be further trained and weighted to predict a series of flats or blocks in the same building where 
assessment would be highly correlated, alongside other patterns, whilst use_code is a quick moderately accurate result, 
but has little ability to scale.

It does not yet outperform the first 4 digits of the SSL though. 



## Steps
1. Load the dataset
2. Preprocess the data
   - Remove the rows with missing values
   - Split SSL into 2 parts as first gives some grouping information
   - Parse the address field to get some standardised details e.g. street name
   - Create a combined_street_info field with some address details + neighborhood
   - Target encode the combined_street_info and first part of ssl to give average values as input because one_hot would add too many features
   - one_hot encoding of categorical features
7. Train models
   - Random Forest
   - Gradient Boosting
8. Output the most important features for each with respect to the 2014 assessment
9. Plot most important feature and display them in webpage

## Setup
1. Install the required packages using the following command:  
`pip install -r requirements.txt`

2. Download the dataset from https://data.world/codefordc/dc-real-property-assessment-data and save it as data/dc-real-property-assessment-data.csv.

## Running the Analysis
To run the analysis, simply execute the following command:

`python src/app.py`

This will load the dataset, preprocess the data, train a random forest model, find the most important feature, and display a plot of the most important feature with respect to the 2014 assessment.

- `Navigate to localhost:3001 to see the result.`

## Dockerizing the App
You can also run the analysis using Docker.  
1. Build the Docker image:  
`docker build -t task-2 .`  

2. Run the Docker container:  
`docker run -it --rm task-2`

## Configuration
You can change the plot title and some other details by modifying the config/config.yaml file. Update the plot_title value to change the plot title.