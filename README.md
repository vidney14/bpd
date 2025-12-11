(https://youtu.be/WeJyDgC8ZJU) Team DataJustice
# Boston Police Department (BPD) Budget & Payroll Analysis  

## Project Overview  
This project analyzes the **Boston Police Department (BPD) budget and payroll data (2024)** to understand:   
- Shifts in funding across departments.  
- Officer pay trends (regular vs overtime pay).  
- Employees with highest regular pay with respect to other Boston city employees.  
- Prediction of overtime expenditure based on regular,retro,injure,other,detail payments and postal code given of the employees using regression.
- Classification of overtime expenditure based on regular,retro,injure,other,detail payments and postal code given of the employees
- Website Boston Police Department (BPD) Budget & Payroll Analysis explaining the workflow and predicting the overtime expenditure using machine learning models.
  
## Makefile
Google Colab Makefile (Python Only – No Website)
<br/>
PYTHON = python3
<br/>
PIP = pip
<br/>

Install Dependencies
<br/>
install:<br/>
	$(PIP) install numpy==2.0.2<br/>
	$(PIP) install pandas==2.2.2<br/>
	$(PIP) install scikit-learn==1.6.1<br/>
	$(PIP) install matplotlib==3.10.0<br/>
	$(PIP) install xgboost<br/>


 Run placeholder (edit if needed)

run:<br/>
	@echo "Run your notebook or script manually in Google Colab."


Makefile for Website (VS Code / Local Machine Terminal)<br/>
PIP = pip

install:<br/>
	$(PIP) install flask<br/>
	$(PIP) install flask-cors<br/>
	$(PIP) install pandas<br/>
	$(PIP) install numpy<br/>
	$(PIP) install scikit-learn<br/>
	$(PIP) install xgboost<br/>


run:<br/>
	python app2.py

## Workflow:
1. Import the Employee Earnings Data (2024).
2. Execute **Boston_Police_Overtime.ipynb** to produce the cleaned dataset **Cleaned_police_overtime_data.csv**.
3. Run **data_visualization.ipynb** to generate visualizations and analytical summaries based on the cleaned 2024 dataset.
4. Use **ML_Model_DS_Project.ipynb** to train and evaluate three machine learning models—**XGBoost**, **Random Forest Regressor**, and **Linear Regression**—using the prepared 2024 data.
5. Import the Employee Earnings Data (2023).
6. Execute **Boston_Police_Overtime_test.ipynb** to create the cleaned test dataset **Cleaned_data_test.csv**.
7. Run **ML_Model_DS_test.ipynb** to assess the performance of **XGBoost** and **Random Forest Regressor** on the 2023 test data.
8. Execute the project website featuring a dashboard, analytical insights, and an overtime cost prediction interface powered by the trained **XGBoost** model and **Cleaned_police_overtime_data.csv**.

## Dataset and Dataset preprocessing 
The analysis uses single dataset:  
- **Employee Earnings Data (2024)** – Payroll data for all Boston employees (filter for police). This is the original dataset link: https://data.boston.gov/dataset/employee-earnings-report/resource/579a4be3-9ca7-4183-bc95-7d67ee715b6d.
- **Cleaned_police_overtime_data**- The Employee Earnings Data (2024) data that is filtered and cleaned with zero missing values.
**Dataset preprocessing**
  - The original dataset  Employee Earnings Data (2024) had many missing values. So at the first we removed the columns which had the most missing values compared to the values present or the columns that were redundant. Column - QUINN_EDUCATION, TOTAL GROSS were dropped.
  - Data was the preprocessed with filling of missing values using data augmentation tools like SimpleImputer, IterativeImputer
  - Dataset was then ready with 14057 data values and with zero missing values.
-  **Employee Earnings Data (2023)** - Payroll data for all Boston employees https://data.boston.gov/dataset/employee-earnings-report/resource/6b3c5333-1dcb-4b3d-9cd7-6a03fb526da7
- **Cleaned_data_test**- The Employee Earnings Data (2023) data that is filtered and cleaned with zero missing values.
**Dataset preprocessing**
  - The dataset Employee Earnings Data (2023) had many missing values. So at the first we removed the columns which had the most missing values compared to the values present or the columns that were redundant. Column - QUINN_EDUCATION, TOTAL GROSS were dropped.
  - Data was the preprocessed with filling of missing values using data augmentation tools like SimpleImputer, IterativeImputer
  - Dataset was then ready with 14004 data values and with zero missing values.
  

---
## Data Visualization
- Data will be visualized using the pie charts and the bar plots.
- Data that was preprocessed was visualized by seeing:
   - Distribution of regular pay for different departments with pie and bar plots.
   - Top 10 police employees by total pay using the bar plot
   - Relationship between Regular pay and overtime pay using the scatter plot
   - Correlation Heatmap among the pay components
   - Distribution of pay components (regular+injury+other+detail+overtime) using bar and pie plots
   
## Train/Test Strategy:
- Data will be split into 80-20 split. 80 percent of the data will be used as a training data and the later 20 percent would be used as a test data to check the performances of the models.
  
## Models:
Data was split into X and Y dataset. X columns represent the input(The numerical columns other than OVERTIME) and Y column represent the output(OVERTIME). Then models were used:
- XGBoost will be the main model to be used with results mean absolute error: $2756.10 and $r^2$ score = 0.820. Additionally, there are more models that are going to be used those are:
    - Random Forest Regressor with $R^2$ score = 0.813 and mean absolute error:$2838.64
   - Linear Regression with $R^2$ score = 0.709 and mean absolute error:$4420.76   
- There are shape value explainers also added to visualize which inputs are important thereby affecting the output.
Additionally clustering was done among officers using KMeans based on pay patterns . We have also summarized the clustering analysis.

Testing with 2023 data
The Cleaned_data_test which is cleaned dataset of Employee Earnings Data (2023) and testing was done by ML_Model_DS_test.ipynb
The models used are Random Forest Regressor and Xgboost models.
Model results: Random Forest Model Mean Absolute Error (MAE): $9,578.76 R² Score: 0.123
Xg BOOST Model Mean Absolute Error (MAE): $10,010.79 R² Score: 0.045

## Codes:
- Boston_Police_Overtime.ipynb explains the data preprocessing and cleaning part. It also visualizes distribution of regular pay for different apartments using bar and pie plots.This code uses employee_earnings_report_2024.csv and creates cleaned_police_overtime_data.csv.
- data_visualization.ipynb explains the data visualization part of the project. This code uses cleaned_police_overtime_data.csv.
- ML_Model_DS_Project.ipynb explains the models part of the project. This code uses cleaned_police_overtime_data.csv.
- Classification_model.ipynb explains the classification part to predict the high overtime earners and low overtime earners.This code uses cleaned_police_overtime_data.csv.
- Boston_Police_Overtime_test.ipynb creates the cleaned dataset of Employee Earnings Data (2023) which is Cleaned_data_test.
- ML_Model_DS_test.ipynb this code tests the Cleaned_data_test dataset using the Random Forest Regressor and Xgboost models.xgb_model.json was created and saved.

# Data_Science Folder:
- This folder contains the complete codebase for the Boston Police Department (BPD) Budget & Payroll Analysis website. It includes the dashboard components, machine learning model scripts, and the main application file app2.py, which runs both the frontend and backend of the project. The file xgb_model.json contains the trained XGBoost model integrated into the system for overtime cost prediction.

## Goals achieved:  
- Shifts in funding across departments.  
- Officer pay trends (regular vs overtime pay).  
- Employees with highest regular pay with respect to other Boston city employees.  
- Prediction of overtime expenditure based on regular,retro,injure,other payments given to the employees by data preprocessing and using different models with results.
- Explored clustering of officers based on scatter plot of overtime pay vs regular pay.
- Predicted overtime costs for different dataset that is 2023 data.
- Build classification model to predict which officers are likely to become “high overtime earners” in 2024 data.
- Explainatory website which explains the project including the machine learning model xgboost predicting the overtime costs given the input parameters.
  

## Tech Stack  
- **Python** (data preprocessing, ML models)  
- **Google Colab** (development environment, notebooks)
- **Visual Studio Code** (development environment)
- **Flask**
- **Javascript** 
- **HTML/CSS Boostrap**
  
# About us:
- Kaushik Joshi
- Ashish Joshi
- Siddhanth Kalyanaraman
- Vidney Jadhav
- Adithya Nayak

