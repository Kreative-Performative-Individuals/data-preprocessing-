# Data preprocessing block

In the repository you will find several documents. 

The main files used for running the pipeline are streaming_pipeline.py and on_request_pipeline.py. 

streaming_pipeline.py is presenting a while loop, so it's continuosly running when the application is started and performs the cleaning and storing of new streaming data.

on_request_pipeline.py presents the pipeline that starts whenever a request for forecasting or feature engineering is received and ends when the result is delivered to requester.

On connections_functions.py there are all the necessary connections to other engines of the application that are required to efficiently run the pipelines, there you can find get_datapoint(.) (for the moment is a mockup of the streaming), get_historical_data(.), send_alert(.) and store_datapoint(.).

dataprocessing_functions.py on other side contains the main functions and classes of the data processing block, like for example data_cleaning(.), ADWIN_drift(.), AnomalyDetector (), feature_engineering_pipeline(.), tdnn_forecasting_prediction(.).., and all their subfunctions.

smart_app_data.pkl is the dataset that was provided to us.

synthetic_data.json and store.json are instead mockup data that we use to simulate the streaming data and the historical data.

Other documents are related to the API connections, explainability and security controls.

In the folder exploration documents that are used for the test of the function or the exploration of data are stored.
