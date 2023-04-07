# Task 2: DC Real Property Assessment Data Analysis

This project analyzes the DC Real Property Assessment Data to find the most important feature with respect to the 2014 assessment, and visualize the feature

## Approach

On first glance at the dataset it was clear that for some entries the address would provide quite accurate predictions 
where entries existed for similar addresses. However, the address field was not formatted in a regular way.

Taking the easy to parse categorical features first, I used one_hot encoding and quickly found the use_code to be best
of those features. However, it was not something that could scale with accuracy as a dataset grew whilst address with 
sufficient data could provide much closer predictions given the right training data.

I decided to use the address field to create a new feature called combined_street_info. This feature 
combined a few details parsed from address to get decent result but still not quite as strong as use_code. 
It occured to me that multiple streets could have the same names so I added neighborhood to the feature to make it more unique.  
With this the new feature drastically outperformed the use_code feature. Although neighborhood was needed to get 
this performance, I think its still fair to say that address is the most important feature we have. 
As well as becoming stronger and stronger the more data collected, it also benefits from having sufficient detail to 
look up the property online and get additional information to further improve predictions.

Address also can be further trained and weighted to predict a series of flats or blocks in the same building where 
assessment would be highly correlated, alongside other patterns, whilst use_code is a quick moderately accurate result, 
but has little ability to scale.

## Steps
1. Load the dataset
2. Preprocess the data
   - Remove the rows with missing values
   - Split SSL into 2 parts as first gives some grouping information
   - Parse the address field to get some standardised details e.g. street name
5. Create a combined_street_info field with some address details + neighborhood
   - Target encode the combined_street_info to give average values as input because one_hot would add too many collumns
6. one_hot encoding of categorical features
7. Train a random forest model
8. Output the most important feature with respect to the 2014 assessment
9. Plot most important feature

## Setup
1. Install the required packages using the following command:  
`pip install -r requirements.txt`

2. Download the dataset from https://data.world/codefordc/dc-real-property-assessment-data and save it as data/dc-real-property-assessment-data.csv.

## Running the Analysis
To run the analysis, simply execute the following command:

`python src/app.py`

This will load the dataset, preprocess the data, train a random forest model, find the most important feature, and display a plot of the most important feature with respect to the 2014 assessment.

## Dockerizing the App
You can also run the analysis using Docker.  
1. Build the Docker image:  
`docker build -t task-2 .`  

2. Run the Docker container:  
`docker run -it --rm task-2`

## Configuration
You can change the plot title by modifying the config/config.yaml file. Update the plot_title value to change the plot title.

## Folder Structure
The project is organized as follows:

```
task-2/
|-- config/
|   |-- config.yaml
|-- data/
|   |-- assessment-data.csv
|-- src/
|   |-- config.py
|   |-- app.py
|-- Dockerfile
|-- requirements.txt
|-- README.md
```

- config/: Contains the configuration file config.yaml for data the plot details.
- data/: Contains the input dataset assessment-data.csv.
- src/: Contains the source code for data processing, modeling, and visualization.
- Dockerfile: Defines the Docker image for the application.
- requirements.txt: Lists the required packages for the application.
- README.md: Provides instructions on how to set up and run the analysis.
