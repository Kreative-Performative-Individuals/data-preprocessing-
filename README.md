# data-preprocessing-

In the repository you will find several documents. 

smart_app_data.pkl is the dataset that was provided to us.

In pipeline_prototype you will find a prototype of the pipeline that is not functional yet, but it represents the workflow for the cleaning, data transformation and ML implementation. It also simulate the data ingestion, for testing with both data coming from the dataset or random generated. 

In information.py the fundamental information about the dataset itself (comprending the machines, kpis, values and operations), as well as information about how to preprocess each kpi are stored. This file is also the one that allow to store the dictionaries for the extraction of the batches and models that will be generated and used along the pipeline implementation.

In data_cleaning_functions the functions that are used in the data cleaning step of the pipeline are implemented, their functioning was tested on simulation data.

In feature_engineering_functions the functions that are used in the feature engineering step of the pipeline are implemented, their functioning was tested on simulation data. Note that how it is implemented right now is to perform the data transformation as a previous step to the machine learning, but the same functions can be used to generate new features.

In machine_learning_functions the functions that are used in the machine learning step are implemented, their functioning was tested on simulation data. They include the drift, the anomaly detection and the forecasting algorithm. Future developments will include the creation of new features.


In the folder exploration documents that are used for the test of the function or the exploration of data are stored.