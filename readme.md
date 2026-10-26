# Violent Increase in Points! | Parallel Prediction Model Based on Informer+BiGRU-GlobalAttention

This issue proposes an Informer-+BiGRU-GlobalAttention parallel prediction model based on a wind power power data set, which significantly improves the prediction effect in single step prediction tasks!
● Dataset: Wind power dataset, electricity dataset, wind speed dataset, etc. of a wind farm
● Environment Framework: Python 3.9, PyTorch 1.8 and above can be run.
● Step-by-step prediction model score: 0.98 on test set
● Usage Object: Paper requirements, graduation design requirements.
● Code Guarantee: The code comments are detailed and ready to run.
We also provide experiments and comparisons based on multiple datasets in multiple prediction tasks, along with detailed documentation and tutorials on how to replace your own dataset, parameter tuning guides, and replacement of prediction tasks. The code is commented line by line, and the introduction of parameters is detailed:
1. Innovation Model Effect:
1.1 Model Evaluation:
\t  
1.2 Wind Power Power Forecast Visualization:
\t1.3 Power Load Forecast Visualization:
2. Introduction to the Model Innovation Points
2.1 Combine the advantages of Informer and BiGRU-GATT
Informer: Expert in handling long sequences, capable of parallel computation, enhancing computational efficiency and predictive performance. Informer has been improved on the basis of Transformer to better suit time series data, especially those with long-term dependencies.
\tBiGRU-GlobalAttention: By utilizing the Bidirectional Gated Recurrent Unit (BiGRU) to model sequences from both forward and backward directions simultaneously, this approach better captures dependencies within the sequence. Additionally, it applies a Global Attention mechanism to weight the output of the BiGRU, focusing the model on the most critical parts of the sequence, thereby enhancing predictive performance.
\t  
By using both models in parallel, it is possible to better capture patterns across different time scales, thereby enhancing the accuracy and robustness of predictions.
\t2.2 Parallel Model Architecture\tParallel use of Informer and BiGRU-GATT. Through two branches of parallel learning. The model can extract and process information at different time scales:
(1) Informer section:
\tPrimarily used for capturing global temporal patterns, suitable for processing longitudinal data.
\tEnhance the ability to capture temporal dependencies over long distances through sparse self-attention mechanisms.
Global feature extraction for input data.
\t  
(2) BiGRU-GATT Part:
Multi-feature sequence data is processed simultaneously through a BiGRU network optimized with GlobalAttention. GlobalAttention is a mechanism designed to enhance the model's focus on different parts of the input sequence. In the BiGRU model, the global attention mechanism helps the model concentrate more on the most relevant parts of the input sequence, thereby enhancing the model's performance and generalization capabilities. At each time step, the global attention mechanism calculates a weight vector that represents the model's focus on various parts of the input sequence. These weights are then applied to the feature representations output by the BiGRU, allowing for weighted consideration of all features at each position, enabling the model to pay more attention to important temporal features. This approach improves the model's ability to perceive temporal features in multi-feature sequences.
\t  
This architecture can more comprehensively capture the characteristics of sequential data, enhancing the model's predictive performance.
\t2.3 Model Fusion \tConcatenate the outputs of Informer and BiGRU-GATT, and fuse the features of different models through a fully connected layer. This fusion method enables the model to simultaneously utilize the global information extraction capability of Informer and the local temporal relationship modeling capability of BiGRU-GATT.
2.4 Efficient Computation
The use of Informer significantly enhances the computational efficiency for long-term sequential data, while the implementation of BiGRU-GATT ensures the full utilization of local temporal information. This combination not only guarantees efficient computation but also improves prediction accuracy and reliability.
3 Informer in-depth explanation with three innovative points
3.1 Probability - Sparse Attention Mechanism (ProbSparse Self-attention)
ProbSparse Self-attention is a sparse attention mechanism introduced in the Informer model. The core idea is to select the most important part of attention weights through probability methods, ignoring those that have a smaller impact on the results. This method can significantly reduce computational complexity while maintaining high model performance.
Sparse Attention: Unlike the dense self-attention mechanism of standard Transformers, Informer introduces ProbSparse Self-attention, which selects important Q-K pairs through a probability sampling mechanism, reducing computational complexity.
Efficiency Improvement: The sparse attention mechanism significantly reduces computational complexity from O(L2⋅d) to O(L⋅log(L)⋅d), where L is the sequence length and d is the feature dimension per time step.
3.2 Multi-scale Feature Extraction - Information Distillation
Informer's architecture diagram does not mark N Encoder stacks on the left side as in Transformers, but rather two large and small trapezoids. From a horizontal perspective, look at the individual Encoder (the big trapezoid on the left side of the architecture diagram, representing the main stack of the entire input sequence).
The Encoder's role is Self-attention Distilling. Due to the ProbSparse self-correlation mechanism often using V's mean for filling, there naturally exists redundant attention source. Therefore, convolution and pooling are applied between adjacent Attention Blocks to downsample features. Thus, in designing the Encoder, distillation operations are continuously extracted to focus on key features, thereby obtaining feature maps worth paying attention to.

Multi-scale Time Series Feature Extraction: Informer models different time granularities through multi-scale methods, which can better capture the multi-scale dependencies in time series.

Information Distillation: The introduction of information distillation mechanism gradually reduces the temporal step size by hierarchical time convolutional pooling layers, extracting features at different scales, achieving efficient modeling of long-term dependencies.

Convolutional Downsampling: A 1D convolutional pooling layer is used in the encoder for downsampling, with a stride of 2, reducing the sequence length by half, further reducing computational complexity.

Information Compression: Through convolutional pooling layers, information is compressed, condensing long sequence information into shorter temporal steps, thus more efficiently performing temporal modeling.

3.3 Time Coding
Informer not only adds PositionEmbedding (position encoding) essential for the Transformer architecture but also adds various encodings related to time:

Day cycle encoding: Represents time points within a day.
Week cycle encoding: Represents time points within a week.
Month cycle encoding: Represents time points within a month.
In the LSTF problem, the ability to capture remote independence requires global information, such as hierarchical timestamps (week, month, and year) and unknowable timestamps (holidays, events).
The specific addition of what kind of GlobalTimeStamp is needed to be added here still needs to be confirmed according to the actual problem. If calculating the passenger flow at high-speed train stations, it is evident that the "holiday" time difference is very important. If calculating the passenger flow of public transportation such as buses and subways, it is clear that "weekday" can reveal whether it is a working day.
4. Data Introductions on Wind Power Power and Other Datasets
4.1 Import Data
\t  
The wind power dataset consists of 35,040 samples with 15 features. The first 6,000 samples are selected for visualization.
4.2 Data Set Creation and Preprocessing
For detailed information, please refer to the provided documents!
\t  
5 Parallel Prediction Model Based on Informer-BiGRU-GATT
\t5.1 Define the Informer-BiGRU-GATT parallel prediction network model
5.2 Set parameters and train the model
50 epochs, MSE is 0.01284. The Informer-BiGRU-GATT parallel prediction effect is significant. The model can fully utilize the long-term dependency modeling ability of the Informer and the short-term dependency capturing ability of the BiGRU-GATT. The convergence speed is fast, the performance is excellent, the prediction accuracy is high. If the model parameters are appropriately adjusted, the model prediction performance can be further improved.
6 Results Visualization and Model Evaluation
6.1 Visualization of Prediction Results
6.2 Model Evaluation
The forecast results show that the Informer-BiGRU-GATT parallel prediction model has a good fitting effect. Through this design, we can fully utilize the advantages of Informer and BiGRU-GATT to achieve efficient and accurate sequential prediction, with significant combined prediction effects!
Click to download: Complete original data, Python code
Previously Featured Content:
Time Series Forecasting: Analysis and comparison of LSTM, ARIMA, Holt-Winters, SARIMA models
Wind Speed Prediction (I) Data Set Introduction and Preprocessing
Wind Speed Prediction (II) EMD-LSTM Model Based on PyTorch
Wind Speed Prediction (III) EMD-LSTM-Attention Model
Wind Speed Prediction (IV) EMD-Transformer Model Based on PyTorch
Wind Speed Prediction (V) EMD-CNN-LSTM Model Based on PyTorch
Wind Speed Prediction (VI) EMD-CNN-GRU Parallel Model Based on PyTorch
CEEMDAN + Combination Prediction Model (BiLSTM-Attention + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-LSTM + ARIMA)
CEEMDAN + Combination Prediction Model (Transformer - BiLSTM + ARIMA)
CEEMDAN + Combination Prediction Model (CNN-Transformer + ARIMA)
Multi-Feature Variable Sequence Prediction (I) - CNN-LSTM Wind Speed Prediction Model
Multi-Feature Variable Sequence Prediction (II) - CNN-LSTM-Attention Wind Speed Prediction Model
Multi-Feature Variable Sequence Prediction (III) - CNN-Transformer Wind Speed Prediction Model
Multi-Feature Variable Sequence Prediction (IV) Transformer-BiLSTM Wind Speed Prediction Model
Multi-Feature Variable Sequence Prediction (V) CEEMDAN+CNN-LSTM Wind Speed Prediction Model
Multi-Feature Variable Sequence Prediction (VI) CEEMDAN+CNN-Transformer Wind Speed Prediction Model
Prediction Model Based on SSA Optimized by Sparrow: CEEMDAN-BiLSTM-Attention
Prediction Model Based on SSA Optimized by Sparrow: CEEMDAN-Transformer-BiGRU
Superior Prediction Model: Secondary Decomposition-Combination Prediction
Multi-feature variable sequence prediction (7) CEEMDAN+Transformer-BiLSTM prediction model
Multi-feature variable sequence prediction (8) CEEMDAN-SSA-BiLSTM prediction model based on the Plumage optimization algorithm
Multi-feature variable sequence prediction (11) TCN-GRU prediction model based on Pytorch
VMD + CEEMDAN secondary decomposition, CNN-LSTM prediction model
VMD + CEEMDAN secondary decomposition, CNN-Transformer prediction model
VMD + CEEMDAN secondary decomposition, Transformer-BiGRU prediction model
Prediction model based on SSA of Plumage optimization algorithm - code family kit
| Superior combination prediction model!
High innovation | CEEMDAN + SSA-TCN-BiLSTM-Attention prediction model
| Superior combination prediction model!
Lowest price in the network | Family kit continuously updated!
| Parallel prediction model based on TCN-SENet + BiGRU-GlobalAttention
Secondary decomposition - innovative model prediction family kit
| BiTCN-BiGRU-CrossAttention integrated spatiotemporal features high innovation prediction model
Combination prediction update | Superior combination prediction model!
Spatiotemporal feature integration BiTCN-Transformer parallel prediction model
| Based on multi-level attention mechanism parallel prediction model
 | CEEMDAN-CNN-GRU-GlobalAttention + XGBoost combination prediction
Multi-step prediction series | LSTM, CNN, Transformer, TCN, serial, parallel model collection

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
