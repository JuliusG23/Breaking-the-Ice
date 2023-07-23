# Breaking the Ice

All the files required to reproduce the research from the thesis ‘Breaking the Ice: Surpassing the Cold User Barrier in Model-Based Recommender Systems using Active Learning Approaches’ by Julius Geukers, are included in this repository. This research uses implicit user feedback from a luxury webshop. A user-item interaction is positive (coded as 1) if the user buys the item more than it returns. A user-item interaction is negative (coded as 0) if the user purchases and returns the item equally. Some interactions are missing (items that the user neither buys nor returns). The data set has over 500,000 distinct users and more than 200,000 distinct items. It also has about 2.5 million interactions, with an average of 4.55 interactions per user.

The data set can be reconstructed by downloading the seven zipped user-item-matrix files and unzipping and recombining them with 7-Zip File Manager (https://www.7-zip.org/). Note: the index (first column) provided in the dataset should be ignored.

In addition to the data set, this repository contains the Python file used to implement the matrix factorization recommender and the active learning approaches. Moreover, the R file used to analyze the results is also included. The research has been conducted with Python version 3.10.11 and R version 4.2.3.
