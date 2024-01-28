Certainly! I've incorporated the additional information about the dataset, including its data types, non-null counts, and descriptive statistics. Here's the updated report:

# Graduate Admissions Prediction - Neural Networks

## 1. Introduction:
This report focuses on developing a neural network model to predict the likelihood of admission based on various features. The dataset used contains information such as GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research, and the Chance of Admit.

## 2. Data Overview:
The dataset comprises 500 samples with 8 columns. The Serial No. column is dropped as it does not contribute to the prediction. The dataset is explored to understand its structure, data types, non-null counts, and descriptive statistics.

## 3. Sample Dataset:
| GRE Score | TOEFL Score | University Rating | SOP | LOR | CGPA | Research | Chance of Admit |
|-----------|-------------|---------------------|-----|-----|------|----------|------------------|
| 337       | 118         | 4                   | 4.5 | 4.5 | 9.65 | 1        | 0.92             |
| 324       | 107         | 4                   | 4.0 | 4.5 | 8.87 | 1        | 0.76             |
| 316       | 104         | 3                   | 3.0 | 3.5 | 8.00 | 1        | 0.72             |
| 322       | 110         | 3                   | 3.5 | 2.5 | 8.67 | 1        | 0.80             |
| 314       | 103         | 2                   | 2.0 | 3.0 | 8.21 | 0        | 0.65             |

## 4. Dataset Information:
- **Data Types:**
  - GRE Score: int64
  - TOEFL Score: int64
  - University Rating: int64
  - SOP: float64
  - LOR: float64
  - CGPA: float64
  - Research: int64
  - Chance of Admit: float64

- **Non-Null Counts:**
  - All columns have 500 non-null entries.

- **Descriptive Statistics:**
  | | GRE Score | TOEFL Score | University Rating | SOP | LOR | CGPA | Research | Chance of Admit |
  |---|-----------|-------------|---------------------|-----|-----|------|----------|------------------|
  | count | 500.000000 | 500.000000 | 500.000000 | 500.000000 | 500.00000 | 500.000000 | 500.000000 | 500.00000 |
  | mean | 316.472000 | 107.192000 | 3.114000 | 3.374000 | 3.48400 | 8.576440 | 0.560000 | 0.72174 |
  | std | 11.295148 | 6.081868 | 1.143512 | 0.991004 | 0.92545 | 0.604813 | 0.496884 | 0.14114 |
  | min | 290.000000 | 92.000000 | 1.000000 | 1.000000 | 1.00000 | 6.800000 | 0.000000 | 0.34000 |
  | 25% | 308.000000 | 103.000000 | 2.000000 | 2.500000 | 3.00000 | 8.127500 | 0.000000 | 0.63000 |
  | 50% | 317.000000 | 107.000000 | 3.000000 | 3.500000 | 3.50000 | 8.560000 | 1.000000 | 0.72000 |
  | 75% | 325.000000 | 112.000000 | 4.000000 | 4.000000 | 4.00000 | 9.040000 | 1.000000 | 0.82000 |
  | max | 340.000000 | 120.000000 | 5.000000 | 5.000000 | 5.00000 | 9.920000 | 1.000000 | 0.97000 |

## 5. Exploratory Data Analysis (EDA):
Descriptive statistics are utilized to analyze the range and distribution of features. A pair plot is generated to visualize relationships between different features, providing insights into potential correlations.

![eda](https://github.com/Kunal3012/NeuralNetworks_GraduateAdmissionPrediction/blob/main/pearplot.png)

## 6. Data Preprocessing:
The data is split into features (X) and the target variable (y). A train-test split is performed, with 80% of the data used for training and 20% for testing. Standardization is applied to ensure consistent scaling of the features.
Certainly! I've added the data preprocessing section to the report:

```python
X = df.drop('Chance of Admit ', axis=1)
y = df['Chance of Admit ']

# Train Test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

This preprocessing step ensures that the features are on a similar scale, contributing to the stability and convergence of the neural network during training.

## 7. Model Architecture:
A neural network model is constructed using the Keras Sequential API. It consists of multiple dense layers with varying activation functions, aiming to capture the complex relationships within the data. The model is compiled with the Adam optimizer and mean squared error loss.

## 8. Model Training:
The model is trained on the training set for 205 epochs. During training, the loss is continuously minimized, and the model learns to make predictions on the validation set.

## 9. Model Summary:
The architecture of the trained model is summarized, providing insights into the number of layers, output shapes, and trainable parameters below :

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 17)                136       
                                                                 
 dense_1 (Dense)             (None, 15)                270       
                                                                 
 dense_2 (Dense)             (None, 10)                160       
                                                                 
 dense_3 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 577 (2.25 KB)
Trainable params: 577 (2.25 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

This summary provides insights into the number of layers, output shapes, and trainable parameters of the neural network model.

## 10. Model Evaluation:



| **Metric**              | **Value**                     |
|-------------------------|-------------------------------|
| Loss                    | 0.0037                        |
| Mean Squared Error (MSE)| 0.0037                        |
| R2 Score                | 0.8187                        |

This table summarizes the evaluation metrics obtained after testing the model on the validation set. The loss, mean squared error (MSE), and R2 score provide insights into the performance of the neural network.

## 11. Loss Plot:

![model training/epochs](https://github.com/Kunal3012/NeuralNetworks_GraduateAdmissionPrediction/blob/main/download.png)
    
A plot is generated to visualize the training and validation loss over the 205 epochs. This plot helps to understand the convergence and potential overfitting of the model.

For a more detailed overview, including the code implementation, you can refer to the [Kaggle Notebook](https://www.kaggle.com/kunal30122002/neuralnetworks-mnistdigitclassification/edit).