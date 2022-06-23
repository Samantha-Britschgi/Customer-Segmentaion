# Customer-Segmentation
### Background
This project was part of the final project for "Programming for Data Analytics: Python & Machine Learning" as part of the MBA in Data Analytics program at Seattle Pacific University. 

Data is obtained from [kaggle](https://www.kaggle.com/imakash3011/customer-personality-analysis). The dataset is small containing 29 columns and 2.2k rows in a single csv file. 

* ***The presented problem:***
There is a lack of understanding on consumer habits which is crucial as their expedited growth has started to slow down. 
As the business had been growing a lot, they are hoping to figure out their consumer groups and the habits of these groups to best create better marketing campaigns targeted towards each group. 

* ***The goals:*** 
    See influential variables that determine habits of consumers
    Create consumer groups based on their shared habits 
    See similarities between these groups

### Instructions
The python [code](https://github.com/Samantha-Britschgi/Customer-Segmentaion/blob/dcd9b8d4241ef441d80f0c0abdf940da95557b18/EDA%20&%20Customer%20Segmentation.ipynb) was written inside of Jupyter Notebooks, but any IDE editor should work to run the code.
 
### Process
    1. Installing & Importing Libraries
    2. Data Wrangling 
    3. Exploratory Analysis
    4. Model Comparison (Kmeans vs GMM for Clustering)
    5. Kmeans Consumer Cluster Analysis 
    6. Cluster Segmentation & Recommendations
    
### Findings
The KMeans clustering successfully determined 4 customer clusters, which were then segmented by demographics and habits  
<img src="https://github.com/Samantha-Britschgi/Customer-Segmentaion/blob/dcd9b8d4241ef441d80f0c0abdf940da95557b18/Customer-Segmentation%20Images/Clusters.png" width="800" height="400" />

Based on these clusters only one marketing campaign had been succesful, so it was recommended that marketing campaigns are updated to target each segment.

A [deck](https://docs.google.com/presentation/d/1GGol9PqSIw4yfXw50EyIArNH6T-FTt5amRpb6rJwM_s/edit?usp=sharing) was made to easily display the process as well as present the findings in a comprehensible way to encourage marketing campaign action.
