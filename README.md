# Cattle_Monitoring_Tracking
Develop a system to detect individual cows, count them, monitor behaviour (standing/lying/eating), and identify signs of distress. 

# How to run model
1. Extract files from Implementation folder onto your local device.
2. Create a venv and install requirements.txt
3. Open and terminal and type "python -m streamlit run app.py"
4. Upload video on streamlit to detect behavior

# About the files
1. Cattle_Detection_Counting contains YOLO and DeepSort Code for counting and detecting catlle
2. Pose_Estimation contains code to "TEST" pose (NOT TO TRAIN).
3. Detect+Pose is combination of above two files
4. Cow_Random_Forest contains code to train random forest 
    1. To train Cow_Random_Forest you need a csv file similar to master_cow_dataV2.csv file
    2. master_cow_dataV2.csv was created by :
        * Extract keypoints in JSON format
        * Create features based on JSON file
        * Integrate all features

# Dataset Used
1. Cattle Dataset (Pig , Sheep ,Cow , Horse) - Kaggle (for Cow detection and counting) : https://www.kaggle.com/datasets/amiteshpatra07/cattle-dataset-pig-sheep-cow-horse
2. Cow Pose Estimation Dataset - Kaggle (For pose estimation) : https://www.kaggle.com/datasets/zaidworks0508/cow-pose-estimation-dataset



Note :
* Pose estimation was trained on Roboflow. Therefore not provided with yaml file.





