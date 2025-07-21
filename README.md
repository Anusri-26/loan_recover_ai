📊 Loan Recovery Risk Analysis and Segmentation
This project leverages data visualization, unsupervised clustering, and machine learning to analyze a loan dataset and recommend targeted recovery strategies based on borrower risk profiles.

📁 Dataset
File: loan_recovery.csv

Features: Includes borrower demographics, loan terms, payment history, and recovery status.
Target (for classification): Whether a borrower is High Risk or not, derived using clustering.

🧠 Project Workflow
1. 📥 Load and Explore Data
Load the dataset using pandas
Display first few rows and summary statistics

2. 📈 Data Visualization
Loan Amount Distribution with overlaid density curve
Loan Amount vs Monthly Income using scatter plot
Payment History vs Recovery Status using grouped histogram

3. 👥 Borrower Segmentation using K-Means
Selected features are standardized
KMeans clustering (n_clusters=4) is used to segment borrowers into 4 groups
Segments are later used to define high-risk borrowers

4. 🔍 Risk Classification using Random Forest
Borrowers in certain clusters (0 and 3) are flagged as High Risk
A RandomForestClassifier is trained to predict High_Risk_Flag based on borrower features

5. 📊 Predict Risk Score & Recommend Strategy
Risk scores (probability of high risk) are predicted for test set
Recovery strategy is assigned based on risk score:
Risk Score Range	Recommended Strategy
> 0.75	Immediate legal notices & aggressive recovery
0.50 to 0.75	Settlement offers & repayment plans
< 0.50	Automated reminders & monitoring

🧪 Tech Stack
Python 3
Pandas, Scikit-learn
Plotly for interactive visualizations
Machine Learning: KMeans clustering, Random Forest classifier
Preprocessing: StandardScaler

📦 How to Run
Clone the repository
Install dependencies:
pip install pandas scikit-learn plotly
Place your loan-recovery.csv file in the root directory

Run the script:
Loan Recover AI.py

📌 Sample Output
Interactive visual plots in browser
Console output with test set predictions and assigned recovery strategies

📚 Future Enhancements
Include feature importance visualization
Evaluate model using metrics like ROC-AUC, F1-Score
Add support for automated report generation
