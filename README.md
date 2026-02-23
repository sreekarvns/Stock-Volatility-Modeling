# Stock-Volatility-Modeling


<img width="2780" height="977" alt="image" src="https://github.com/user-attachments/assets/ed96f9af-9161-4426-aab8-69efcfd0d4b2" />


<img width="2780" height="1177" alt="image" src="https://github.com/user-attachments/assets/8de4b763-892c-4616-b1f2-83cb1918c392" />


<img width="2775" height="1177" alt="image" src="https://github.com/user-attachments/assets/113242e7-fcfa-4596-992f-74f1e6a1326a" />


<img width="2400" height="1500" alt="image" src="https://github.com/user-attachments/assets/5701c6b6-9f95-4fe4-bb78-72132e8bd508" />



<img width="3000" height="1800" alt="image" src="https://github.com/user-attachments/assets/ea7db395-6c5d-4f2b-a2b3-d6710fa8eb2d" />





📈 Hybrid Volatility Forecasting using GARCH, LSTM & Market Regime Detection

This project builds a complete end-to-end financial time-series volatility forecasting system using traditional econometric models and deep learning.

It combines ARCH/GARCH family models, LSTM neural networks, and Hidden Markov Models (HMM) to improve prediction of stock market volatility and capture changing market regimes.

This project was developed as a deep exploration into real-world financial modeling and required extensive experimentation, model tuning, and evaluation.

🎯 Objective

Financial markets exhibit volatility clustering, regime shifts, and nonlinear behavior that cannot be captured by a single model.

This project aims to:

Model volatility using econometric methods

Enhance predictions using deep learning

Detect market regimes (calm vs volatile)

Build a hybrid ensemble for better accuracy

Evaluate performance across multiple real stocks

📊 Dataset Used

Source: Kaggle
NIFTY-50 India Stock Data (10 Years Historical)
https://www.kaggle.com/datasets/ankitpalcode/nifty-50-india-stock-data-for-last-10-years/data

The dataset contains daily:

Open

High

Low

Close

Volume

for major NIFTY-50 companies.

Stocks used in this project include:

RELIANCE

TCS

TATASTEEL

TATAMOTORS

TITAN

ULTRACEMCO

WIPRO

UPL

SBIN

SBILIFE

SUNPHARMA

TECHM

TATACONSUM

Using multiple stocks ensured the model was tested on different volatility behaviors and not overfitted to a single asset.

🧠 What This Project Actually Does

This is not just a model.

It is a complete volatility research pipeline:

1️⃣ Download and preprocess historical stock data
2️⃣ Compute log-returns
3️⃣ Analyze volatility clustering using ACF
4️⃣ Fit ARCH/GARCH family models
5️⃣ Train LSTM on volatility patterns
6️⃣ Detect market regimes using HMM
7️⃣ Build Hybrid GARCH-LSTM ensemble
8️⃣ Compare predictions with realized volatility
9️⃣ Evaluate using multiple statistical metrics
🔟 Generate detailed visual analytics

🏗 Models Implemented
Econometric Models

ARCH(1)

GARCH(1,1)

EGARCH(1,1)

GJR-GARCH(1,1)

Deep Learning

LSTM-based volatility forecasting

Hybrid Model

Ensemble of GARCH + LSTM predictions

Regime Detection

Hidden Markov Model (HMM)

Identifies:

Calm markets

High-volatility regimes

📂 Project Architecture
Hybrid-Volatility-Model/
│
├── download_and_prepare_data.py
├── plot_acf_analysis.py
├── fit_volatility_models.py
├── hybrid_garch_lstm.py
├── detect_market_regimes.py
├── evaluate_and_plot_volatility.py
├── create_hybrid_visualizations.py
├── plot_volatility_with_ci.py
├── main.py
└── outputs/
📈 Analysis Performed
Volatility Behavior

ACF of returns

ACF of squared returns

Volatility clustering detection

Model Evaluation Metrics

RMSE

MAE

Correlation

Directional Accuracy

Accuracy %

Visualization Suite

Volatility comparison plots

Hybrid vs realized volatility

RMSE comparison charts

Correlation heatmaps

Box plots

Regime detection timelines

🏆 Key Results

Hybrid GARCH-LSTM consistently outperformed standalone models

Significant improvement in RMSE and MAE

High directional accuracy in volatility prediction

Strong correlation with realized market volatility

HMM successfully captured regime shifts during volatile periods

🔬 What I Implemented

This project involved:

End-to-end financial data pipeline development

Multiple econometric model implementations

Deep learning model design and training

Ensemble modeling

Regime detection using probabilistic models

Model comparison and statistical evaluation

Advanced visualization for interpretability

This required extensive experimentation, debugging, parameter tuning, and validation across multiple datasets.

💻 How to Run
git clone https://github.com/your-username/hybrid-volatility-model.git
cd hybrid-volatility-model

pip install -r requirements.txt
python main.py
🎯 Applications

Risk management

Algorithmic trading

Portfolio volatility estimation

Market behavior analysis

Quantitative finance research

🧠 Skills Demonstrated

Time-series analysis

Financial econometrics

Deep learning for finance

Hybrid modeling

Ensemble techniques

Statistical evaluation

Data visualization

Research-level ML pipeline design

❤️ Author

Made with ❤️ by sreekarvns

This project represents hours of experimentation, learning, and implementation in financial machine learning and hybrid modeling.

🔹 GitHub Repo Description

Hybrid financial volatility forecasting system using GARCH models, LSTM neural networks, and HMM regime detection on NIFTY-50 stock data.
