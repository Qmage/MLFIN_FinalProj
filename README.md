# COMS 4995 Machine Learning Applications in Finance \[Final Project\]
This is the code repository for the group final project for 
Columbia University 2020 Summer Semester COMS 4995

Joint work by members:
- Adrian Adduci (faa2160)
- Clement Chin (kc3168)
- Adam Kravitz (ajk2254)
- Yuhao Zhang (yz2700)

## System requirements:
1. Downloaded the Program and Data Files contained in this repository (you can download the entire repo as zip, then extract them)
2. have a Anaconda python distribution installed on your computer
3. make sure the installed Pandas library is lower than v1.1.0
4. run "pip install streamlit" to install the library streamlit

## To preprocess newly extracted data from Bloomberg terminal:
 
1.      Open the Economic Data Cleaning.ipynb using Jupyter notebook
2.      Modify the second cell of input data to be the Bloomberg extract that you wish to process. e.g. 'DATA_PULL_BB_080120_HC.xlsx'
3.      Modify the last cell of output filename as needed e.g. 'Economic_data_clean_20200801.xlsx'
4.      Run all the cells to generate the clean output file that can be served to the program.
-      The example raw dataset (from Bloomberg) and processed dataset (excel) are available in this repository. 
 
## To run the Web Application:
 
1.      Go to your terminal and move to the directory where the Application.py code is
2.      In the terminal type “streamlit run Application.py”
3.      Click on the top IP address link that is generated (should take you to the browser)
4.      On the left, select an excel file, e.g. the (Economic_data_clean_20200801.xlsx)
5.      On the left, select a start and end time for data to train on
6.      Press the “Train Model” button
 
## Additional Options For The Web Application:
 
1.      Press “Display Historical Data Table?” to see the data we’re using.
2.      Click in the “View Historical Indices” search field. You can select multiple other factors and see how those factors compares to the target variables.
3.      On the left, can press on “Feature Importance”. On the page can pick “which model?”, the number picked shows the graphs when the model forecasts is that number of days ahead
4.      On the left, can press on “Model Performance”. On the page can pick “which model?”, the number picked shows the graphs when the model forecasts is that number of days ahead
 
## Prediction and Forecast Interpretation:
 
1.      The written recommendation. The top written prediction is for the CDX HY, the bottom written prediction is for the CDX IG
2.      Each row represent different days.
3.      Green arrow up, means the model predicts the corresponding CDX will increase. Red arrow down, means the model predicts the corresponding CDX will decrease.
 
## Prediction and Forecast Column Label Interpretation:
 
1.      CDX_HY = CDX.HY
2.      CDX_IG = CDX.IG
3.      Pred = Models predictions
4.      “X”D = Models Prediction’s “X” days ahead.
