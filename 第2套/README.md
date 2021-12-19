# AWS MLS 66-130

1. 一位机器学习专家正在与一家大公司合作，在其产品中利用机器学习。该公司希望根据哪些客户在未来6个月内会或不会流失，将其客户分成几类。该公司已将数据标记给专家，专家应使用哪种机器学习模型类型来完成这项任务？ A Machine Learning Specialist is working with a large company to leverage machine learning within its products. The company wants to group its customers into categories based on which customers will and will not churn within the next six months. The company has labeled the data available to the Specialist Which machine learning model type should the Specialist use to accomplish this task?
   - [ ] A. 线性回归 Linear regression
   - [ ] B. 分类 Classification
   - [ ] C. 聚类 Clustering
   - [ ] D. 强化学习 Reinforcement learning

   <details>
      <summary>Answer</summary>

      答案B：分类的目标是确定一个数据点（在我们的例子中是客户）属于哪个类别或类别。对于分类问题，数据科学家将使用具有预定义目标变量的历史数据，即标签（流失者/非流失者）答案，需要对其进行预测以训练算法。通过分类，企业可以回答以下问题。这个客户会不会流失。客户是否会续订？一个用户会不会降低定价计划的等级？是否有任何不寻常的客户行为的迹象？

   </details>

2. 显示的图表来自一个测试时间序列的预测模型。仅仅考虑该图，机器学习专家应该对该模式的行为做出哪个结论？ The displayed graph is from a forecasting model for testing a time series. Considering the graph only, which conclusion should a Machine Learning Specialist make about the behavior of the mode?

   ![67](./img/67.png)

   - [ ] A. 该模型对趋势和季节性的预测都很好。 The model predicts both the trend and the seasonality well.
   - [ ] B. 该模型很好地预测了趋势，但没有预测季节性。 The model predicts the trend well, but not the seasonality.
   - [ ] C. 该模型很好地预测了季节性，但没有预测趋势。 The model predicts the seasonality well, but not the trend.
   - [ ] D. 该模型不能很好地预测趋势或季节性。 The model does not predict the trend or the seasonality well.

   <details>
      <summary>Answer</summary>

      答案A。

   </details>

3. 一家公司希望将用户行为分类为欺诈行为或正常行为。根据内部研究，一位机器学习专家希望建立一个基于两个特征的二进制分类器：账户年龄和交易月份。这些特征的类别分布如图所示。基于这些信息，哪个模型的准确率最高？ A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features is illustrated in the figure provided. Based on this information, which model would have the HIGHEST accuracy?

   ![68](./img/68.png)

   - [ ] A. 长短时记忆(LSTM)模型与缩放指数线性单元(SELU) Long short-term memory(LSTM)model with scaled exponential linear unit (SELU)
   - [ ] B. Logistic回归 Logistic regression
   - [ ] C. 支持向量机（SVM）与非线性核 Support vector machine(SVM) with non-linear kernel
   - [ ] D. 具有tanh激活函数的单感知器 Single perceptron with tanh activation function

   <details>
      <summary>Answer</summary>

      答案C。

   </details>

4. 一家公司使用决策树收集了客户对其产品的评论，将其评为安全或不安全。训练数据集有以下特征：ID、日期、完整评论、完整评论摘要和一个二进制安全/不安全标签。在训练过程中，任何缺少特征的数据样本都会被剔除。在少数情况下，测试集被发现缺少完整的评论文本字段。对于这个用例，哪一个是解决有缺失特征的测试数据样本的最有效行动方案？ A company has collected customer comments on its products, rating them as safe or unsafe, using decision trees. The training dataset has the following features id, date, full review, full review summary, and a binary safe/unsafe tag. During training, any data sample with missing features was dropped. In a few instances, the test set was found to be missing the full review text field. For this use case, which is the most effective course of action to address test data samples with missing features?
   - [ ] A. 丢掉缺少完整评论文本字段的测试样本，然后再运行测试集。 Drop the test samples with missing full review text fields, and then run through the test set.
   - [ ] B. 复制摘要文本字段，用它们来填补缺失的完整评论文本字段，然后通过测试集运行。 Copy the summary text fields and use them to fill in the missing full review text fields, and then run through the test set.
   - [ ] C. 使用一种比决策树更好的处理缺失数据的算法。 Use an algorithm that handles missing data better than decision trees.
   - [ ] D. 生成合成数据来填补缺失数据的字段，然后通过测试集运行。 Generate synthetic data to fill in the fields that are missing data, and then run through the test set.

   <details>
      <summary>Answer</summary>

      答案B：在这种情况下，完整的评论摘要通常包含整个评论中最具描述性的短语，是缺失的完整评论文本字段的有效替身。

   </details>

5. 一家保险公司需要将索赔合规性审查自动化，因为人工审查成本高且容易出错。该公司有一大批索赔和每个索赔的合规标签。每个索赔由几个英文句子组成，其中许多包含复杂的相关信息。管理层希望使用Amazon SageMaker的内置算法来设计一个机器学习的监督模型，该模型可以被训练来阅读每一个索赔，并预测索赔是否合规。应该使用哪种方法从索赔中提取特征，作为下游监督任务的输入？ An insurance company needs to automate claim compliance reviews because human reviews are expensive and error prone. The company has a large set of claims and a compliance label for each. Each claim consists of a few sentences in English, many of which contain complex related information. Management would like to use Amazon SageMaker built-in algorithms to design a machine learning supervised model that can be trained to read each claim and predict if the claim is compliant or not. Which approach should be used to extract features from the claims to be used as inputs for the downstream supervised task?
   - [ ] A. 从整个数据集中的目标中得出一个标记词典。对在训练集的每项要求中发现的标记进行一次编码。将导出的特征步伐作为输入发送到Amazon SageMaker内置的监督学习算法。 Derive a dictionary of tokens from aims in the entire dataset. Apply one-hot encoding to tokens found in each claim of the training set. Send the derived features pace as inputs to an Amazon SageMaker built in supervised learning algorithm.
   - [ ] B. 将Amazon SageMaker Blazing Text以Word2Vec模式应用于训练集的索赔。将导出的特征空间作为下游监督任务的输入。 Apply Amazon SageMaker Blazing Text in Word2Vec mode to claims in the training set. Send the derived features space as inputs for the downstream supervised task.
   - [ ] C. 在分类模式下将Amazon SageMaker Blazing Text应用于训练集中已标记的索赔，以得出索赔的特征，分别对应于合规和不合规的标签。 Apply Amazon SageMaker Blazing Text in classification mode to labeled claims in the training set to derive features for the claims that correspond to the compliant and non-compliant labels, respectively.
   - [ ] D. 将Amazon SageMaker Object2Vec应用于训练集中的索赔。将导出的特征空间作为下游监督任务的输入。 Apply Amazon SageMaker Object2Vec to claims in the training set. Send the derived features space as inputs for the downstream supervised task.

   <details>
      <summary>Answer</summary>

      答案D：Amazon SageMaker Object2Vec将单词的Word2Vec嵌入技术推广到更复杂的对象，如句子和段落。由于监督学习任务是在有标签的整个索赔水平上进行的，而在单词水平上没有标签，因此需要使用Object2Vec而不是Word2Vec。

   </details>

6. 一家公司正在运行一个机器学习预测服务，每天产生100TB的预测数据。一个机器学习专家必须从预测中生成一个每日精度-召回曲线的可视化，并将一个只读的版本转发给业务团队。哪个解决方案需要最少的编码工作？ A company is running a machine learning prediction service that generates 100 TB of predictions every day. A Machine learning Specialist must generate a visualization of the daily precision-recall curve from the predictions and forward a read-only version to the Business team. Which solution requires the LEAST coding effort?
   - [ ] A. 运行每天的亚马逊EMR工作流来生成精确召回数据并将结果保存在亚马逊S3中。给予业务团队对S3的只读访问权。 Run daily Amazon EMR workflow to generate precision-recall data and save the results in Amazon S3. Give the Business team read-only access to S3.
   - [ ] B. 在Amazon QuickSight中生成每天的精确呼叫数据，并在与业务团队共享的仪表板中发布结果。 Generate daily precision-recall data in Amazon QuickSight and publish the results in a dashboard shared with the Business team.
   - [ ] C. 运行每天的亚马逊EMR工作流来生成精确召回数据并将结果保存在亚马逊S3中。在Amazon QuickSight中对阵列进行可视化，并将其发布在与业务团队共享的仪表盘中。 Run a daily Amazon EMR workflow to generate precision-recall data and save the results in Amazon S3. Visualize the arrays in Amazon QuickSight and publish them in a dashboard shared with the Business team.
   - [ ] D. 在Amazon ES中生成每天的精确召回数据，并在与业务团队共享的仪表板中发布结果。 Generate daily precision-recall data in Amazon ES and publish the results in a dashboard shared with the Business team.

    <details>
      <summary>Answer</summary>

      答案C。

   </details>

7. 机器学习专家正在准备数据，以便在Amazon SageMaker上进行训练。专家正在使用Shoemaker的一个内置算法进行训练。数据集是以CSV格式存储的，并被转换为NumPy数组，这似乎对训练的速度产生了负面影响。专家应该做什么来优化数据，以便在SageMaker上进行训练？ A Machine Learning Specialist is preparing data for raining on Amazon SageMaker. The Specialist is using one of the Shoemaker built-in algorithms for the training. The dataset is stored in CSV format and is transformed into a NumPy array, which appears to be negatively affecting the speed of the training. What should the Specialist do to optimize the data for training on SageMaker?
   - [ ] A. 使用SageMaker的批量转换功能，将训练数据转换为一个数据框架。 Use the SageMaker batch transform feature to transform the training data into a Data frame.
   - [ ] B. 使用AWS Glue将数据压缩成Apache Parquet格式。 Use AWS Glue to compress the data into the Apache Parquet format.
   - [ ] C. 将数据集转换成RecordIO protobuf格式。 Transform the dataset into the RecordIO protobuf format
   - [ ] D. 使用SageMaker超参数优化功能，自动优化数据。Use the SageMaker hyperparameter optimization feature to automatically optimize the data.

    <details>
      <summary>Answer</summary>

      答案C：许多Amazon SageMaker内置算法对protobuf格式的RecordIO数据优化都非常好。

   </details>

8. 一位机器学习专家被要求建立一个有监督的图像识别模型来识别一只猫。ML专家进行了一些测试，并为一个基于神经网络的图像分类器记录了以下结果。可用图像总数=1,000；测试集图像=100（恒定测试集）。ML专家注意到，在超过75%的错误分类的图像中，猫被它们的主人倒提着。ML专家可以使用哪些技术来改善这个特定的测试错误？ A Machine Learning Specialist is required to build a supervised image-recognition model to identify a cat. The ML Specialist performs some tests and records the following results for a neural network-based image classifier: Total number of images available =1,000; Test set images=100 (constant test set). The ML Specialist notices that, in over 75% of the misclassified images, the cats were held upside down by their owners. Which techniques can be used by the ML Specialist to improve this specific test error?
   - [ ] A. 通过增加训练图像的旋转变化来增加训练数据。 Increase the training data by adding variation in rotation for training images.
   - [ ] B. 增加模型训练的epochs数量。Increase the number of epochs for model training.
   - [ ] C. 增加神经网络的层数。Increase the number of layers for the neural network.
   - [ ] D. 增加倒数第二层的辍学率。 Increase the dropout rate for the second-to-last layer.

   <details>
      <summary>Answer</summary>

      答案A。

   </details>

9. 机器学习专家需要能够摄取流媒体数据并将其存储在Apache Parquet文件中，以便进行探索和分析。以下哪项服务能够以正确的格式摄取和存储这些数据？ Machine Learning Specialist needs to be able to ingest streaming data and store it in Apache Parquet files for exploration and analysis. Which of the following services would both ingest and store this data in the correct format?
   - [ ] A. AWS DMS
   - [ ] B. Amazon Kinesis Data Streams
   - [ ] C. Amazon Kinesis Data Firehose
   - [ ] D. Amazon Kinesis Data Analytics

   <details>
      <summary>Answer</summary>

      答案C。

   </details>

10. 一位数据科学家正在开发一个机器学习模型，以对一项金融交易是否是欺诈性的进行分类。可供训练的标记数据包括100,000个非欺诈性的观察值和1,000个欺诈性的观察值。数据科学家将XGBboost算法应用于数据，当训练好的模型应用于之前未见过的验证数据集时，会产生以下混淆矩阵。该模型的准确率为99.1%，但数据科学家被要求减少假阴性的数量。数据科学家应该采取哪种步骤组合来减少该模型的假阳性预测的数量？(选择两个) A Data Scientist is developing a machine learning model to classify whether a financial transaction is fraudulent. The labeled data available for raining consists of 100,000 non-fraudulent observations and 1.000 fraudulent observations. The Data Scientist applies the XGBboost algorithm o the data, resulting in the following confusion matrix when the trained model is applied to a previously unseen validation dataset. The accuracy of the model is 99.1%, but the Data Scientist has been asked to reduce the number of false negatives. Which combination of steps should the Data Scientist take to reduce the number of false positive predictions by the model? (Choose two)

    ![75](./img/75.png)

    - [ ] A. 改变XGBboost`eval_metric`参数，根据RMSE而不是误差进行优化。 Change the XGBboost `eval_metric` parameter to optimize based on RMSE instead of error.
    - [ ] B. 增加XGBboost `scale_pos_weight`参数来调整正负权重的平衡。 Increase the XGBboost `scale_pos_weight` parameter to adjust the balance of positive and negative weights.
    - [ ] C. 增加XGBboost `max_depth`参数，因为模型目前对数据的拟合不足。 Increase the XGBboost `max_depth` parameter because the model is currently underfitting the data.
    - [ ] D. 改变XGBOOST `eval_metric`参数，根据AUC而不是误差来优化。 Change the XGBOOST `eval_metric` parameter to optimize based on AUC instead of error.
    - [ ] E. 减少XGBOOST `max_depth`参数，因为该模型目前过度拟合数据。 Decrease the XGBOOST `max_depth` parameter because the model is currently overfitting the data.

    <details>
       <summary>Answer</summary>

       答案BD。

    </details>

11. 一位机器学习专家被分配到一个TensorFlow项目唱Amazon SageMaker进行训练，并需要在没有WI-FI接入的情况下继续工作很长一段时间。该专家应该使用哪种方法来继续工作？ A Machine Learning Specialist is assigned a TensorFlow project sing Amazon SageMaker for training and needs to continue working for an extended period with no WI-FI access. Which approach should the Specialist use to continue working?
    - [ ] A. 在他们的笔记本电脑上安装Python 3和boto3，并使用该环境继续进行代码开发。 Install Python 3 and boto3 on their laptop and continue the code development using that environment
    - [ ] B. 从GitHub下载Amazon SageMaker中使用的TensorFlow Docker容器到他们的本地环境，并使用Amazon SageMaker Python SDK来测试代码。 Download the TensorFlow Docker container used in Amazon SageMaker from GitHub to their local environment and use the Amazon SageMaker Python SDK to test the code.
    - [ ] C. 从tensorflow.org下载TensorFlow，在SageMaker环境中模拟TensorFlow内核。 Download TensorFlow from tensorflow.org to emulate the TensorFlow kernel in the SageMaker environment.
    - [ ] D. 将SageMaker笔记本下载到他们的本地环境，然后在他们的笔记本电脑上安装Jupyter笔记本，在本地笔记本中继续开发。 Download the SageMaker notebook to their local environment, then install Jupyter Notebooks on their laptop and continue the development in a local notebook.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

12. 一家对安全敏感的公司的机器学习专家正在准备一个数据集进行模型训练。该数据集存储在Amazon S3中，包含个人身份信息（P）。该数据集必须只能从VPC中访问。-不能穿越公共互联网。怎样才能满足这些要求？ A Machine Learning Specialist at a company sensitive to security is preparing a dataset for model training. The dataset is stored in Amazon S3 and contains Personally Identifiable Information(P). The dataset -Must be accessible from a VPC only. -Must not traverse the public internet. How can these requirements be satisfied?
    - [ ] A. 创建一个VPC端点，应用一个桶访问策略，限制对给定的VPC端点和VPC的访问。 Create a VPC endpoint and apply a bucket access policy that restricts access to the given VPC endpoint and the VPC.
    - [ ] B. 创建一个VPC端点，并应用一个桶访问策略，允许从给定的VPC端点和Amazon EC2实例访问。 Create a VPC endpoint and apply a bucket access policy that allows access from the given VPC endpoint and an Amazon EC2 instance.
    - [ ] C. 创建一个VPC端点，并使用网络访问控制列表（NACLS），只允许指定的VPC端点和Amazon EC2实例之间的流量。 Create a VPC endpoint and use Network Access Control Lists (NACLS) to allow traffic between only the given VPC endpoint and an Amazon EC2 instance.
    - [ ] D. 创建一个VPC端点，并使用安全组来限制对指定的VPC端点和亚马逊EC2实例的访问。 Create a VPC endpoint and use security groups to restrict access to the given VPC endpoint and an Amazon EC2 instance.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

13. 在对一个分类问题的神经网络进行小批量训练时，一位数据科学家注意到训练的准确性出现了震荡。这个问题最可能的原因是什么？ During mini-batch training of a neural network for a classification problem, a Data Scientist notices that training accuracy oscillates. What is the MOST likely cause of this issue?
    - [ ] A. 数据集中的类别分布是不平衡的 The class distribution in the dataset is imbalanced
    - [ ] B. 数据集洗牌被禁用 Dataset shuffling is disabled
    - [ ] C. 批量大小太大 The batch size is too big
    - [ ] D. 学习率非常高 The learning rate is very high

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

14. 一名员工在一家公司的社交媒体上发现了一个带有音频的视频片段。视频中使用的语言是西班牙语。英语是该员工的第一语言，而他们不懂西班牙语。该员工想做一个情感分析。哪种服务的组合对完成这项任务最有效？ An employee found a video clip with audio on a company’s social media feed. The language used in the video is Spanish. English is the employee’s first language, and they do not understand Spanish. The employee wants to do a sentiment analysis. What combination of services is the MOST efficient to accomplish the task?
    - [ ] A. Amazon Transcribe, Amazon Translate, and Amazon Comprehend.
    - [ ] B. Amazon Transcribe, Amazon Comprehend, and Amazon SageMaker seq2seq.
    - [ ] C. Amazon Transcribe, Amazon Translate, and Amazon SageMaker Neural Topic Model (NTM).
    - [ ] D. Amazon Transcribe, Amazon Translate and Amazon SageMaker Blazing Text.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

15. 一位机器学习专家正在将一个定制的Resnet模型打包成一个Docker容器，以便公司能够利用Amazon SageMaker进行训练。该专家正在使用亚马逊EC2 P3实例来训练该模型，并需要正确配置Docker容器以利用英伟达GPUS。该专家需要做什么？ A Machine Learning Specialist is packaging a custom Resnet model into a Docker container so the company can leverage Amazon SageMaker for training. The Specialist is using Amazon EC2 P3 instances to train the model and needs to properly configure the Docker container to leverage the NVIDIA GPUS. What does the Specialist need to do?
    - [ ] A. 将NVIDIA驱动程序与Docker镜像捆绑在一起。 Bundle the NVIDIA drivers with the Docker image.
    - [ ] B. 构建Docker容器，使其与NVIDIA-Docker兼容。 Build the Docker container to be NVIDIA-Docker compatible.
    - [ ] C. 组织Docker容器的文件结构，以便在GPU实例上执行。 Organize the Docker containers file structure to execute on GPU instances.
    - [ ] D. 在Amazon SageMaker中设置GPU标志。创建TrainingJob请求体。 Set the GPU flag in the Amazon SageMaker. Create TrainingJob request body.

    <details>
       <summary>Answer</summary>

       答案B。如果你计划使用GPU设备，请确保我们的容器与nvidia-docker兼容，只有CUDA工具包应该被包含在容器上。不要在镜像中捆绑NVIDIA驱动程序。关于nvidia-docker的更多信息，请参见NVIDIA/nvidia-docker。

    </details>

16. 一位机器学习专家正在建立一个逻辑回归模型，预测一个人是否会订购比萨饼。该专家正试图建立一个具有理想分类阈值的最佳模型。该专家应该使用什么模型评估技术来了解不同的分类阈值将如何影响模型的性能？ A Machine Learning Specialist is building a logistic regression model that will predict whether or not a person will order a pizza. The Specialist is trying to build the optimal model with an ideal classification threshold. What model evaluation technique should the Specialist use to understand how different classification thresholds will impact the model’s performance?
    - [ ] A. 接收者操作特征（ROC）曲线 Receiver operating characteristic (ROC) curve
    - [ ] B. 错误分类率 Misclassification rate
    - [ ] C. 均方根误差(RMSE) Root Mean Square Error (RMSE)
    - [ ] D. L1准则 L1 norm

    <details>
       <summary>Answer</summary>

       答案A：[ref](https://docs.aws.amazon.com/zh_cn/machine-learning/latest/dg/binary-model-insights.html)

    </details>

17. 一位机器学习专家正在开发一个回归模型，以预测出租房源的租金。一个名为Wall_Color的变量代表了该房产最突出的外墙颜色。以下是样本数据。排除所有其他变量。Propery_ID Wall_Color: 1000 Red, 1001 White, 1002 Green. 专家选择了一个需要数字输入数据的模式。专家应该使用哪些特征工程师的方法来让回归模型从墙壁颜色数据中学习？(选择两个) A Machine Learning Specialist is developing a regression model to predict rental rates from rental listings. A variable named Wall_Color represents the most prominent exterior wall color of the property. The following is the sample data. excluding all other variables: Propery_ID Wall_Color: 1000 Red, 1001 White, 1002 Green. The specialist chose a mode that needs numerical input data. Which feature engineer approaches should the Specialist use to allow the regression model to learn from the Wall Color data? (Select TWO)
    - [ ] A. 应用整数转换，设置红=1，白=5，绿=10。 Apply integer transformation and set Red=1, White=5, and Green=10.
    - [ ] B. 增加新的列，存储颜色的一热表示。 Add new columns that store one-hot representation of colors.
    - [ ] C. 用颜色名称字符串的长度替换它。 Replace the color name string by its length.
    - [ ] D. 创建三列，以RGB格式对颜色进行编码。 Create three columns to encode the color in RGB format.
    - [ ] E. 用它的训练集频率替换每个颜色名称。 Replace each color name by its training set frequency.

    <details>
       <summary>Answer</summary>

       答案BD。

    </details>

18. 一家大公司开发了一个BI应用，利用从各种运营指标中收集的数据生成报告和仪表盘。该公司希望为高管提供增强的体验，使他们能够使用自然语言从报告中获得数据。该公司希望高管们能够通过书面和口语界面来提问。哪种服务组合可以用来建立这种对话式界面？(选择三个) A large company has developed a BI application that aerates reports and dashboards using data collected from various operational metrics. The company wants to provide executives with an enhanced experience so they can use natural language to get data from the reports. The company wants the executives to be able to ask questions sin written and spoken interfaces. Which combination of services can be used to build this conversational interface? (Select THREE)
    - [ ] A. Alexa for Business
    - [ ] B. Amazon Connect
    - [ ] C. Amazon Lex
    - [ ] D. Amazon Polly
    - [ ] E. Amazon Comprehend
    - [ ] F. Amazon Transcribe

    <details>
       <summary>Answer</summary>

       答案CEF。

    </details>

19. 一位数据科学家正在开发一个管道，以嵌套流式网络流量数据。该数据科学家需要实现一个过程，以识别通常的网络流量模式，作为管道的一部分。该模式将被用于下游的警报和事件响应。如果需要，数据科学家可以访问未标记的历史数据来使用。该解决方案需要做以下工作。为每个网络流量条目计算一个异常分数。使异常事件的识别适应随时间变化的网络模式。该数据科学家应该实施哪种方法来满足这些要求？ A data scientist is developing a pipeline to nest streaming web traffic data. The data scientist needs to implement a process to identify usual web traffic patterns as part of the pipeline. The pattern will be used downstream for alerting and incident response. The data scientist has access to unlabeled historic data to use, if needed. The solution needs to do the following: Calculate an anomaly score for each web traffic entry. Adapt unusual event identification to changing web patterns over time. Which approach should the data scientist implement to meet these requirements?
    - [ ] A. 使用历史网络流量数据，使用Amazon SageMaker随机切割森林（RCF）内置模型训练异常检测模型。使用Amazon Kinesis数据流来处理传入的网络流量数据。附加一个预处理的AWS Lambda函数，通过调用RCF模型计算每条记录的异常分数来执行数据丰富化。 Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker Random Cut Forest (RCF) Built-in model. Use an Amazon Kinesis Data stream to process the incoming web traffic data. Attach a preprocessing AWS Lambda function to perform data enrichment by calling the RCF model to calculate the anomaly score for each record.
    - [ ] B. 使用历史网络流量数据，使用Amazon SageMaker内置的XGBOOST模型训练异常检测模型。使用Amazon Kinesis数据流来处理传入的网络流量数据。附加一个预处理AWS Lambda函数，通过调用XGBOOST模型计算每条记录的异常分数来执行数据丰富化。 Use historic web traffic data to train an anomaly detection model using the Amazon SageMaker built-in XGBOOST model. Use an Amazon Kinesis Data Stream to process the incoming web traffic data. Attach a preprocessing AWS Lambda function to perform data enrichment by calling the XGBOOST model to calculate the anomaly score for each record.
    - [ ] C. 使用Amazon Kinesis Data Firehose收集流数据，将交付流映射为Amazon Kinesis Data Analytics的输入源。编写一个SQL查询，用k-nearest neighbors（KNN）SQL扩展对流媒体数据进行实时运行，使用翻转窗口计算每条记录的异常得分。 Collect the streaming data using Amazon Kinesis Data Firehose Map the delivery stream as an input source for Amazon Kinesis Data Analytics. Write a SQL query to run in real time against the streaming data with the k-nearest neighbors (KNN) SQL extension to calculate anomaly scores for each record using a tumbling window.
    - [ ] D. 使用Amazon Kinesis Data Firehose收集流数据。将交付流映射为亚马逊Kinesis数据分析的输入源。编写一个SQL查询，用亚马逊随机数据实时运行。Cut Forest(RCF)SQL扩展，使用滑动窗口计算每条记录的异常分数。 Collect the streaming data using Amazon Kinesis Data Firehose. Map the delivery stream as an input source for Amazon Kinesis Data Analytics. Write a SQL query to run in real time against the streaming data with the Amazon Random. Cut Forest(RCF)SQL extension to calculate anomaly scores for each record using a sliding window.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

20. 一个机器学习团队在Amazon SageMaker上运行自己的训练算法。训练算法需要外部资产。该团队需要向Amazon SageMaker提交自己的算法代码和算法特定参数。该团队应该使用哪种服务组合来在Amazon SageMaker中建立一个自定义算法？(选择两个) A Machine Learn team runs its own training algorithm on Amazon SageMaker. The training algorithm requires external assets. The team needs to submit both its own algorithm code and algorithm-specific parameters to Amazon SageMaker. Which combination of services should the team use to build a custom algorithm in Amazon SageMaker? (Select TWO)
    - [ ] A. AWS Secrets Manager
    - [ ] B. AWS CodeStar
    - [ ] C. Amazon ECR
    - [ ] D. Amazon ECS
    - [ ] E. Amazon S3

    <details>
       <summary>Answer</summary>

       答案CE。

    </details>

21. A company is setting up an Amazon SageMaker environment. The corporate data security policy does not allow communication over the internet. How can the company enable the Amazon SageMaker service without enabling direct internet access to Amazon SageMaker notebook instances?
    - [ ] A. 在公司VPC内创建一个NAT网关。 Create a NAT gateway within the corporate VPC.
    - [ ] B. 将Amazon SageMaker的流量通过企业内部的网络进行路由。 Route Amazon SageMaker traffic through an on-premises network.
    - [ ] C. 在企业内部创建Amazon SageMaker VPC接口端点。 Create Amazon SageMaker VPC interface endpoints within the corporate.
    - [ ] D. 与托管Amazon SageMaker的Amazon VPC建立VPC对等关系。 Create VPC peering with Amazon VPC hosting Amazon SageMaker.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

22. 机器学习专家正在训练一个模型来识别图像中的车辆品牌和型号。专家希望使用转移学习和一个在一般物体的图像上训练的现有模型。专家整理了一个包含不同车辆品牌和型号的大型自定义图片数据集。专家应该如何初始化模型，以便用自定义数据重新训练它？ Machine Learning Specialist is training a model to identify the make and model of vehicles in images. The Specialist wants to use transfer learning and an existing model trained on images of general objects. The Specialist collated a large custom dataset of pictures containing different vehicle makes and models. What should the Specialist do to initialize the model to re-train it with the custom data?
    - [ ] A. 在所有层中用随机权重初始化模型，包括最后的全连接层。 Initialize the model with random weights in all layers including the last fully connected layer
    - [ ] B. 用所有层的预训练权重初始化模型，并替换最后一个全连接层。 Initialize the model with pre-trained weights in all layers and replace the last fully connected layer.
    - [ ] C. 在所有层中用随机权重初始化模型，并替换最后一个全连接层。 Initialize the model with random weights in all layers and replace the last fully connected layer.
    - [ ] D. 用所有层的预训练权重初始化模型，包括最后一个全连接层。 Initialize the model with pre-trained weights in all layers including the last fully connected layer.
  
    <details>
       <summary>Answer</summary>

       答案B。

    </details>

23. 一家办公室安全机构使用安装在主要办公室内关键位置的100个摄像头进行了一次成功的试点。摄像机的图像被上传到Amazon S3，并使用Amazon Rekognition进行标记，结果被储存在Amazon ES中。该机构现在希望将试点扩大到一个完整的生产系统，在全球的办公地点使用数以千计的视频摄像头。其目标是实时识别非雇员的活动。该机构应该考虑哪种解决方案？ An office security agency conducted a successful pilot using 100 cameras installed at key locations within the main office. Images from the cameras were uploaded to Amazon S3 and tagged using Amazon Rekognition, and the results were stored in Amazon ES. The agency is now looking to expand the pilot into a full production system using thousands of video cameras in its office locations globally. The goal is to identify activities performed by non -employees in real time. Which solution should the agency consider?
    - [ ] A. 在每个地方办公室和每个摄像头使用一个代理服务器，将RTSP馈送流向一个独特的亚马逊Kinesis视频流视频流。在每个流中，使用Amazon Rekognition Video，并创建一个流处理器，从已知员工的集合中检测人脸，并在检测到非员工时发出警报。 Use a proxy server at each local office and for each camera and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection of known employees, and alert when non-employees are detected.
    - [ ] B. 在每个本地办公室和每个摄像头使用一个代理服务器，并将RTSP馈送流向一个独特的亚马逊Kinesis视频流视频流。在每个流中，使用Amazon Rekognition Image从已知员工的集合中检测人脸，并在检测到非员工时发出警报。 Use a proxy server at each local office and for each camera and stream the RTSP feed to a unique Amazon Kinesis Video Streams video stream. On each stream, use Amazon Rekognition Image to detect faces from a collection of known employees and alert when non-employees are detected.
    - [ ] C. 安装AWS DeepLens摄像头，并使用DeepLens Kinesis Video模块将每个摄像头的视频流传到Amazon Kinesis Video Streams。在每个流中，使用Amazon Rekognition Video并创建一个流处理器，从每个流的集合中检测人脸，并在检测到非雇员时发出警报。 Install AWS DeepLens cameras and use the DeepLens Kinesis Video module to stream video to Amazon Kinesis Video Streams for each camera. On each stream, use Amazon Rekognition Video and create a stream processor to detect faces from a collection on each stream, and alert when non-employees are detected.
    - [ ] D. 安装AWS DeepLens摄像机，并使用DeepLens Kinesis Video模块将每个摄像机的视频流转到Amazon Kinesis Video Streams。在每个流中，运行AWS Lambda函数来捕获图像片段，然后调用Amazon Rekognition Image来从已知员工的集合中检测人脸，并在检测到非员工时发出警报。 Install AWS DeepLens cameras and use the DeepLens Kinesis Video module to stream video to Amazon Kinesis Video Streams for each camera. On each stream, run an AWS Lambda function to capture image fragments and then call Amazon Rekognition Image to detect faces from a collection of known employees, and alert when non-employees are detected.

    <details>
       <summary>Answer</summary>

       简单题，答案C

    </details>

24. 一家宠物保险公司的营销经理计划在社交媒体上发起一个有针对性的营销活动，以获取新客户。目前，该公司在Amazon Aurora有以下数据： -所有过去和现有客户的档案。-所有过去和现有的被保险宠物的档案。-保单级别信息 -收到的保费 已支付的索赔。应该采取什么步骤来实现机器学习模型，以识别社交媒体上的潜在新客户？ A Marketing Manager at a pet insurance company plans to launch a targeted marketing campaign on social media to acquire new customers. Currently, the company has the following data in Amazon Aurora: -Profiles for all past and existing customers. -Profiles for all past and existing insured pets. -Policy level information -Premiums received Claims paid. What steps should be taken to implement a machine learning model to identify potential new customers on social media?
    - [ ] A. 对客户资料数据进行回归，了解消费者群体的关键特征。在社交媒体上寻找类似的资料。 Use regression on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.
    - [ ] B. 在客户资料数据上使用聚类，以了解消费者群体的关键特征。在社交媒体上寻找类似的资料。 Use clustering on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.
    - [ ] C. 在客户资料数据上使用推荐引擎来了解消费者群体的主要特征。在社交媒体上寻找类似的资料。 Use a recommendation engine on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.
    - [ ] D. 在客户资料数据上使用决策树分类器引擎来了解消费者群体的主要特征。在社交媒体上寻找类似的资料。 Use a decision tree classifier engine on customer profile data to understand key characteristics of consumer segments. Find similar profiles on social media.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

25. 一家制造公司有一大组带标签的历史销售数据。该制造商想预测每个季度应该生产多少个特定的零件。应该用哪种机器学习方法来解决这个问题？ A manufacturing company has a large set of labeled historical sales data. The manufacturer would like to predict how many units of a particular part should be produced each quarter. Which machine learning approach should be used to solve this problem?
    - [ ] A. Logistic回归 Logistic regression
    - [ ] B. 随机切割森林(RCF) Random Cut Forest (RCF)
    - [ ] C. 主成分分析（PCA） Principal component analysis (PCA)
    - [ ] D. 线性回归 Linear regression

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

26. 一家金融服务公司正在Amazon S3上构建一个强大的无服务器数据湖。该数据湖应该是灵活的，并满足以下要求。-支持通过Amazon Athena和Amazon Redshift Spectrum查询Amazon S3上的旧数据和新数据。-支持事件驱动的ETL管道。-提供快速和简单的方法来理解元数据。哪种方法符合这些要求？ A financial services company is building a robust serverless data lake on Amazon S3. The data lake should be flexible and meet the following requirements: -Support querying old and new data on Amazon S3 through Amazon Athena and Amazon Redshift Spectrum. -Support event-driven ETL pipelines. -Provide a quick and easy way to understand metadata. Which approach meets these requirements?
    - [ ] A. 使用AWS Glue爬虫来抓取S3数据，使用AWS Lambda函数来触发AWS Glue ETL作业，以及使用AWS Glue数据目录来搜索和发现元数据。 Use an AWS Glue crawler to crawl S3 data, an AWS Lambda function to trigger an AWS Glue ETL job, and an AWS Glue Data catalog to search and discover metadata.
    - [ ] B. 使用AWS Glue爬虫来抓取S3数据，使用AWS Lambda函数来触发AWS Batch作业，并使用外部Apache Hive元存储来搜索和发现元数据。 Use an AWS Glue crawler to crawl S3 data, an AWS Lambda function to trigger an AWS Batch job, and an external Apache Hive metastore to search and discover metadata.
    - [ ] C. 使用AWS Glue爬虫来抓取S3数据，使用Amazon Cloud Watch警报来触发AWS Batch作业，并使用AWS Glue数据目录来搜索和发现元数据。 Use an AWS Glue crawler to crawl S3 data, an Amazon Cloud Watch alarm to trigger an AWS Batch job, and an AWS Glue Data Catalog to search and discover metadata.
    - [ ] D. 使用AWS Glue爬虫来抓取S3数据，使用Amazon Cloud Watch报警来触发AWS Glue ETL作业，使用外部Apache Hive元存储来搜索和发现元数据。 Use an AWS Glue crawler to crawl S3 data, an Amazon Cloud watch alarm to trigger an AWS Glue ETL job, and an external Apache Hive metastore to search and discover metadata.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

27. A company’s Machine Learning Specialist needs to improve the training speed of a time-series forecasting model using TensorFlow. The training is currently implemented on a single-GPU machine and takes approximately 23 hours to complete. The training needs to be run daily. The model accuracy is acceptable, but the company anticipates a continuous increase in the size of the training data and a need to update the model on an hourly, rather than a daily, basis. The company also wants to minimize coding effort and infrastructure changes. What should the Machine Learning Specialist do to the training solution to allow it to scale for future demand?
    - [ ] A. Do not change the TensorFlow code. Change the machine to one with a more powerful GPU to speed up the training.
    - [ ] B. Change the TensorFlow code to implement a Horovod distributed framework supported by Amazon SageMaker. Parallelize the training to as many machines as needed to achieve the business goals.
    - [ ] C. Switch to using a built-in AWS SageMaker DEEPAR model. Parallelize the training to as many machines as needed to achieve the business goals.
    - [ ] D. Move the training to Amazon EMR and distribute the workload to as many machines as needed to achieve the business goals.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

28. 机器学习专家一般应使用以下哪种指标来对机器学习分类模型进行相互比较/评价？ Which of the following metrics should a Machine Learning Specialist generally use to compare/evaluate machine learning classification models against each other?
    - [ ] A. 召回率 Recall
    - [ ] B. 错误分类率 Misclassification rate
    - [ ] C. 平均绝对百分比误差（MAPE） Mean absolute percentage error (MAPE)
    - [ ] D. ROC曲线下的面积（AUC） Area Under the ROC Curve (AUC)

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

29. 一位机器学习专家正在与一家大型网络安全公司合作，为世界各地的公司实时管理安全事件。这家网络安全公司希望设计一个解决方案，使其能够在数据输入时，使用机器学习对恶意事件进行评分，作为数据的异常情况。该公司还希望能够将结果保存在其数据湖中，供以后处理和分析。完成这些任务的最有效方法是什么？ A Machine Learning Specialist is working with a large cybersecurity company that manages security events in real time for companies around the world. The cybersecurity company wants to design a solution that will allow It to use machine learning to score malicious events as anomalies on the data as it is being ingested. The company also wants to be able to save the results in its data lake for later processing and analysis. What is the MOST efficient way to accomplish these tasks?
    - [ ] A. 使用Amazon Kinesis Data Firehose摄取数据，并使用Amazon Kinesis Data Analytics Random Cut Forest（RCF）进行异常检测。然后使用Kinesis Data Firehose将结果流向Amazon S3。 Ingest the data using Amazon Kinesis Data Firehose, and use Amazon Kinesis Data Analytics Random Cut Forest (RCF) for anomaly detection. Then use Kinesis Data Firehose to stream the results to Amazon S3.
    - [ ] B. 使用Amazon EMR将数据输入Apache Spark Streaming，并使用Spark MLlib与k-means来进行异常检测。然后使用Amazon EMR将结果存储在Apache Hadoop分布式文件系统（HDFS）中，复制系数为3，作为数据湖。 Ingest the data into Apache Spark Streaming using Amazon EMR and use Spark MLlib with k-means to perform anomaly detection. Then store the results in an Apache Hadoop Distributed File System (HDFS) using Amazon EMR with a replication factor of three as the data lake.
    - [ ] C. 摄取数据并将其存储在Amazon S3中。使用AWS Batch和AWS深度学习AMI，在Amazon S3的数据上使用TensorFlow训练一个k-means模型。 Ingest the data and store it in Amazon S3. Use AWS Batch along with the AWS Deep Learning AMIs to train a k-means model using TensorFlow on the data in Amazon S3.
    - [ ] D. 摄取数据并将其存储在Amazon S3中。让AWS的Glue作业按需触发，转换新数据。然后在Amazon SageMaker中使用内置的随机切割森林（RCF）模型来检测数据中的异常情况。 Ingest the data and store it in Amazon S3. Have an AWS Glue job that is triggered on demand transform the new data. Then use the built-in Random Cut Forest (RCF) model within Amazon SageMaker to detect anomalies in the data.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

30. 一位数据科学家希望获得对GZIP文件数据流的实时洞察力。哪种解决方案允许使用SQL来查询延迟最小的数据流？ A Data Scientist wants to gain real-time insights into a data stream of GZIP files. Which solution would allow the use of SQL to query the stream with the LEAST latency?
    - [ ] A. 亚马逊Kinesis数据分析与AWS Lambda函数来转换数据。 Amazon Kinesis Data Analytics with an AWS Lambda function to transform the data.
    - [ ] B. AWS Glue和一个自定义ETL脚本来转换数据。 AWS Glue with a custom ETL script to transform the data.
    - [ ] C. 亚马逊Kinesis客户端库来转换数据并保存到亚马逊ES集群中。 An Amazon Kinesis Client Library to transform the data and save into an Amazon ES cluster.
    - [ ] D. Amazon Kinesis Data Firehose来转换数据，并将其放入Amazon S3桶中。 Amazon Kinesis Data Firehose to transform the data and put it into an Amazon S3 bucket.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

31. 一家零售公司打算使用机器学习来对新产品进行分类。向数据科学团队提供了一个当前产品的标记数据集。该数据集包括1,200种产品。标记的数据集对每个产品有15个特征，如标题尺寸、重量和价格。每个产品都被标记为属于六个类别中的一个，如书籍、游戏、电子产品和电影。使用所提供的数据集进行训练，哪种模型应该被用来对新产品进行分类？ A retail company intends to use machine learning to categorize new products. A labeled dataset of current products was provided to the Data Science team. The dataset includes 1,200 products. The labeled dataset has 15 features for each product such as title dimensions, weight, and price. Each product is labeled as belonging to one of six categories such as books, games, electronics, and movies. Which model should be used for categorizing new products using the provided dataset for training?
    - [ ] A. 一个XGBboost模型，目标参数被设置为多：softmax。 An XGBboost model where the objective parameter is set to multi: softmax.
    - [ ] B. 一个深度卷积神经网络（CNN），最后一层有一个softmax激活函数A deep convolutional neural network (CNN) with a softmax activation function for the last layer
    - [ ] C. 一个回归森林，树的数量被设置为等于产品类别的数量。 A regression forest where the number of trees is set equal to the number of product categories.
    - [ ] D. 一个基于循环神经网络（RNN）的DeepAR预测模型。 A DeepAR forecasting model based on a recurrent neural network (RNN).
  
    <details>
       <summary>Answer</summary>

       答案A。

    </details>

32. 一位数据科学家正在开发一个执行情感分析的应用程序。验证准确率很低，数据科学家认为原因可能是数据集中有丰富的词汇和低的平均词频。应该使用哪种工具来提高验证的准确性？ A Data Scientist is working on an application that performs sentiment analysis. The validation accuracy is poor, and the Data Scientist thinks that the cause may be a rich vocabulary and a low average frequency of words in the dataset. Which tool should be used to Improve the validation accuracy?
    - [ ] A. Amazon Comprehend 语法分析和实体检测。 Amazon Comprehend syntax analysis and entity detection.
    - [ ] B. Amazon SageMaker Blazing Text CBOW模式。 Amazon SageMaker Blazing Text CBOW mode.
    - [ ] C. 自然语言工具箱（NLTK）的词干和停止词的去除。 Natural Language Toolkit (NLTK) stemming and stop word removal.
    - [ ] D. Scikit-learn术语频率-反向文档频率（TF-IDF）向量器。 Scikit-learn term frequency-inverse document frequency (TF-IDF) vectorizer.
  
    <details>
       <summary>Answer</summary>

       答案D。

    </details>

33. 一位数据科学家需要将一个现有的企业内部ETL流程迁移到云端。目前的流程以固定的时间间隔运行，并使用PySpark将多个大型数据源合并和格式化为一个单一的综合输出，供下游处理。数据科学家对云解决方案有以下要求： -合并多个数据源。-重用现有的PySpark逻辑。-在现有的时间表上运行该解决方案。-最大限度地减少需要管理的服务器的数量。该数据科学家应该使用哪种架构来建立这个解决方案？ A Data Scientist needs to migrate an existing on-premises ETL process to the cloud. The current process runs at regular time intervals and uses PySpark to combine and format multiple large data sources into a single consolidated output for downstream processing. The Data Scientist has been given the following requirements to the cloud solution: -Combine multiple data sources. -Reuse existing PySpark logic. -Run the solution on the existing schedule. -Minimize the number of servers that will need to be managed. Which architecture should the Data Scientist use to build this solution?
    - [ ] A. 把原始数据写到Amazon S3。安排一个AWS Lambda函数，根据现有的时间表向一个持久的亚马逊EMR集群提交Spark步骤。使用现有的PySpark逻辑，在EMR集群上运行ETL工作。将结果输出到Amazon S3中的 "处理过的 "位置，供下游使用。 Write the raw data to Amazon S3. Schedule an AWS Lambda function to submit a Spark step to a persistent Amazon EMR cluster based on the existing schedule. Use the existing PySpark logic to run the ETL job on the EMR cluster. Output the results to a `processed` location in Amazon S3 that is accessible for downstream use.
    - [ ] B. 把原始数据写到Amazon S3。创建一个AWS Glue EL作业，对输入数据进行ETL处理。在PySpark中编写ETL工作，以利用现有的逻辑。创建一个新的AWS Glue触发器，根据现有的时间表触发ETL工作。配置ETL工作的输出目标，将其写入Amazon S3中的 "处理 "位置，供下游使用。 Write the raw data to Amazon S3. Create an AWS Glue EL job to perform the ETL processing against the input data. Write the ETL job in PySpark to leverage the existing logic. Create a new AWS Glue trigger to trigger the ETL job based on the existing schedule. Configure the output target of the ETL job to write to a `processed` location in Amazon S3 that is accessible for downstream use.
    - [ ] C. 把原始数据写到Amazon S3。安排一个AWS Lambda函数在现有的时间表上运行，并处理来自Amazon S3的输入数据。用Python编写Lambda逻辑，并实现现有的PySpark逻辑，以执行ETL过程 让Lambda函数将结果输出到Amazon S3中的`处理`位置，供下游使用。 Write the raw data to Amazon S3. Schedule an AWS Lambda function to run on the existing schedule and process the input data from Amazon S3. Write the Lambda logic in Python and implement the existing PySpark logic to perform the ETL process Have the Lambda function output the results to a `processed` location in Amazon S3 that is accessible for downstream use.
    - [ ] D. 使用Amazon Kinesis Data Analytics来流化输入数据，并对流进行实时SQL查询，以在流中进行所需的转换。将输出结果交付给Amazon S3中的`处理`位置，供下游使用。 Use Amazon Kinesis Data Analytics to stream the input data and perform real-time SQL queries against the stream to carry out the required transformations within the stream. Deliver the output results to a `processed` location in Amazon S3 that is accessible for downstream use.
  
    <details>
       <summary>Answer</summary>

       答案B。

    </details>

34. 一个机器学习团队在Amazon S3中拥有几个大型CSV数据集。历史上，用Amazon SageMaker Linear Learner算法建立的模型在类似大小的数据集上训练需要花费数小时。该团队的领导需要加快训练过程。机器学习专家可以做些什么来解决这个问题？ A Machine Learning team has several large CSV datasets in Amazon S3. Historically, models built with the Amazon SageMaker Linear Learner algorithm have taken hours to train on similar-sized datasets. The team’s leaders need to accelerate the training process. What can a Machine Learning Specialist do to address this concern?
    - [ ] A. 使用Amazon SageMaker Pipe模式。 Use Amazon SageMaker Pipe mode.
    - [ ] B. 使用亚马逊机器学习来训练模型。 Use Amazon Machine Learning to train the models.
    - [ ] C. 使用Amazon Kinesis将数据流到Amazon SageMaker。 Use Amazon Kinesis to stream the data to Amazon SageMaker.
    - [ ] D. 使用AWS Glue将CSV数据集转换为JSON格式。 Use AWS Glue to transform the CSV dataset to the JSON format.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

35. 从由以下两个句子组成的文本语料库中，建立了一个使用单字和大字的术语频率-逆文档频率（tf-idf）矩阵。1 `Please call the number below` 2 `Please do not call us` tf-idf矩阵的尺寸是多少？ A term frequency-inverse document frequency (tf-idf) matrix using both unigrams and bigrams is built from a text corpus consisting of the following two sentences: 1 Please call the number below. 2 Please do not call us. What are the dimensions of the tf-idf matrix?
    - [ ] A. (2, 16)
    - [ ] B. (2, 8)
    - [ ] C. (2, 10)
    - [ ] D. (8, 10)

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

36. 一家大型移动网络运营公司正在建立一个机器学习模型，以预测那些有可能退订服务的客户。该公司计划为这些客户提供奖励，因为客户流失的成本远远大于奖励的成本。在对100个客户的测试数据集进行评估后，该模型产生了以下混淆矩阵。根据模型的评估结果，为什么说这是一个可行的生产模型？ A large mobile network operating company is building a machine learning model to predict customers who are likely to unsubscribe from the service. The company plans to offer an incentive for these customers as the cost of churn is far greater than the cost of the incentive. The model produces the following confusion matrix after evaluating on a test dataset of 100 customers. Based on the model evaluation results, why is this a viable model for production?

    |n=100|PREDICTED: Yes|PREDICTED: No|
    |----|-------------|--------------|
    |Actual: Yes|10|4|
    |Actual: No|10|76|

    - [ ] A. 该模型的精确度为86%，公司因假阴性而产生的成本小于假阳性。 The model is 86 accurate and the cost incurred by the company as a result of false negatives is less than the false positives.
    - [ ] B. 该模型的精确度为86%o，小于该模型的精确度。 The precision of the model is 86%o, which is less than the accuracy of the model.
    - [ ] C. 该模型的准确度为86%，公司因假阳性而产生的成本小于假阴性。 The model is 86% accurate and the cost incurred by the company as a result of false positives is less than the false negatives.
    - [ ] D. 该模型的精确度为86%，大于模型的精确度。 The precision of the model is 86%, which is greater than the accuracy of the model.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

37. 一位机器学习专家正在为一家公司设计一个改善销售的系统。目标是利用该公司拥有的大量关于用户行为和产品偏好的信息，根据用户与其他用户的相似性来预测用户会喜欢哪些产品。专家应该怎样做才能达到这个目标？ A Machine Learning Specialist is designing a system for improving sales for a company. The objective is to use the large amount of information the company has on users' behavior and product preferences to predict which products users would like based on the users' similarity to other users. What should the Specialist do to meet this objective?
    - [ ] A. 在Amazon EMR上用Apache Spark ML构建一个基于内容的过滤推荐引擎。 Build a content-based filtering recommendation engine with Apache Spark ML on Amazon EMR
    - [ ] B. 在Amazon EMR上用Apache Spark ML建立一个协作过滤推荐引擎。 Build a collaborative filtering recommendation engine with Apache Spark ML on Amazon EMR
    - [ ] C. 在Amazon EMR上用Apache Spark AIL建立一个基于模型的过滤推荐引擎。 Build a model-based filtering recommendation engine with Apache Spark AIL on Amazon EMR
    - [ ] D. 在Amazon EMR上用Apache Spark ML建立一个组合式过滤推荐引擎。 Build a combinative filtering recommendation engine with Apache Spark ML on Amazon EMR

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

38. 一家移动网络运营商正在建立一个分析平台，使用Amazon Athena和Amazon S3来分析和优化公司的运营。源系统实时发送CSV格式的数据。数据工程团队希望在将数据存储到Amazon S3之前将其转换为Apache Parquet格式，哪种解决方案的实施工作量最小？ A Mobile Network Operator is building an analytics platform to analyze and optimize a company‘’s operations using Amazon Athena and Amazon S3. The source systems send data in CSV format in real time. The Data Engineering team wants to transform the data to the Apache Parquet format before storing it on Amazon S3 Which solution takes the LEAST effort to implement?
    - [ ] A. 使用Apache Kafka Streams在Amazon EC2实例上摄取CSV数据，并使用Kafka Connect S3将数据序列化为Parquet。 Ingest CSV data using Apache Kafka Streams on Amazon EC2 instances and use Kafka Connect S3 to serialize data as Parquet.
    - [ ] B. 从Amazon Kinesis数据流摄取CSV数据，并使用Amazon Glue将数据转换为Parquet。 Ingest CSV data from Amazon Kinesis Data Streams and use Amazon Glue to convert data into Parquet.
    - [ ] C. 在Amazon MR集群中使用Apache Spark结构化流摄取CSV数据，并使用Apache Spark将数据转换为Parquet。 Ingest CSV data using Apache Spark Structured Streaming in an Amazon MR cluster and use Apache Spark to convert data into Parquet.
    - [ ] D. 从Amazon Kinesis数据流摄取CSV数据，并使用Amazon Kinesis Data Firehose将数据转换为Parquet。 Ingest CSV data from Amazon Kinesis Data Streams and use Amazon Kinesis Data Firehose to convert data into Parquet.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

39. 一个城市希望监测其空气质量，以解决空气污染的后果。一位机器学习专家需要预测该城市未来2天的空气质量，单位为百万分之一的污染物。由于这是一个原型，只有过去一年的每日数据可用，哪个模型最有可能在Amazon SageMaker中提供最佳结果？ A city wants to monitor its air quality to address the consequences of air pollution. A Machine Learning Specialist needs to forecast the air quality in parts per million of contaminates for the next 2 days in the city. As this is a prototype, only daily data from the last year is available Which model is MOST likely to provide the best results in Amazon SageMaker?
    - [ ] A. 在由全年数据组成的单一时间序列上使用Amazon SageMaker的K-Nearest-Neighbors（KNN）算法，并使用预测器类型的回归器。 Use the Amazon SageMaker K-Nearest-Neighbors (KNN) algorithm on the single time series consisting of the full year of data with a predictor type of regressor.
    - [ ] B. 在由全年数据组成的单一时间序列上使用Amazon SageMaker随机切割森林（RCF）。 Use Amazon SageMaker Random Cut Forest (RCF) on the single time series consisting of the full year of data.
    - [ ] C. 在由全年数据组成的单一时间序列上使用Amazon SageMaker Linear Learner算法，其预测器类型为regressor。Use the Amazon SageMaker Linear Learner algorithm on the single fire series consisting of the full year of data with a predictor type of regressor.
    - [ ] D. 在由全年数据组成的单一时间序列上使用Amazon SageMaker Linear Learner算法，预测器类型为分类器。 Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor type of classifier.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

40. 一个城市希望监测其空气质量，以解决空气污染的后果。一位机器学习专家需要预测该城市未来2天的空气质量，单位为百万分之一的污染物。由于这是一个原型，只有过去一年的每日数据可用，哪个模型最有可能在Amazon SageMaker中提供最佳结果？ A city wants to monitor its air quality to address the consequences of air pollution. A Machine Learning Specialist needs to forecast the air quality in parts per million of contaminates for the next 2 days in the city. As this is a prototype, only daily data from the last year is available Which model is MOST likely to provide the best results in Amazon SageMaker?
    - [ ] A. 在由全年数据组成的单一时间序列上使用Amazon SageMaker的K-Nearest-Neighbors（KNN）算法，并使用预测器类型的回归器。 Use the Amazon SageMaker K-Nearest-Neighbors (KNN) algorithm on the single time series consisting of the full year of data with a predictor type of regressor.
    - [ ] B. 在由全年数据组成的单一时间序列上使用Amazon SageMaker随机切割森林（RCF）。 Use Amazon SageMaker Random Cut Forest (RCF) on the single time series consisting of the full year of data.
    - [ ] C. 在由全年数据组成的单一时间序列上使用Amazon SageMaker Linear Learner算法，其预测器类型为regressor。 Use the Amazon SageMaker Linear Learner algorithm on the single fire series consisting of the full year of data with a predictor type of regressor.
    - [ ] D. 在由全年数据组成的单一时间序列上使用Amazon SageMaker Linear Learner算法，预测器类型为分类器。 Use the Amazon SageMaker Linear Learner algorithm on the single time series consisting of the full year of data with a predictor type of classifier.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

41. 一位机器学习专家在企业VPC的一个私有子网中使用Amazon SageMaker笔记本实例。ML专家的重要数据存储在Amazon SageMaker笔记本实例的Amazon EBS卷上，需要对该EBS卷进行快照。然而，ML专家在VPC中找不到Amazon SageMaker笔记本实例的EBS卷或Amazon EC2实例。为什么ML专家在VPC中看不到该实例？ A Machine Learning Specialist is using an Amazon SageMaker notebook instance in a private subnet of a corporate VPC. The ML Specialist has important data stored on the Amazon SageMaker notebook instance’s Amazon EBS volume and needs to take a snapshot of that EBS volume. However, the ML Specialist cannot find the Amazon SageMaker notebook instance’s EBS volume or Amazon EC2 instance within the VPC. Why Is the ML Specialist not seeing the instance visible in the VPC?
    - [ ] A. Amazon SageMaker 笔记本实例是基于客户账户内的 EC2 实例，但它们在 VPC 之外运行。 Amazon SageMaker notebook instances are based on the EC2 instances within the customer account, but they run outside of VPCs.
    - [ ] B. Amazon SageMaker 笔记本实例是基于客户账户内的 Amazon ECS 服务。 Amazon SageMaker notebook instances are based on the Amazon ECS service within customer accounts.
    - [ ] C. Amazon SageMaker笔记本实例是基于AWS服务账户内运行的EC2实例。 Amazon SageMaker notebook instances are based on EC2 instances running within AWS service accounts.
    - [ ] D. Amazon SageMaker笔记本实例是基于在AWS服务账户内运行的AWS ECS实例。 Amazon SageMaker notebook instances are based on AWS ECS instances running within AWS service accounts.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

42. 一位机器学习专家正在建立一个模型，将使用Amazon SageMaker执行时间序列预测。专家已经完成了模型的训练，现在计划在端点上进行负载测试，以便他们能够为模型变体配置自动扩展。哪种方法可以让专家在负载测试期间审查延迟、内存利用率和CPU利用率？ A Machine Learning Specialist is building a model that will perform time series forecasting using Amazon SageMaker. The Specialist has finished training the model and is now planning to perform load testing on the endpoint so they can configure Auto Scaling for the model variant. Which approach will allow the Specialist to review the latency, memory utilization, and CPU utilization during the load test?
    - [ ] A. 通过利用Amazon Athena和Amazon QuickSight来查看已经写入Amazon S3的SageMaker日志，以便在产生日志的时候进行可视化。 Review SageMaker logs that have been written to Amazon S3 by leveraging Amazon Athena and Amazon QuickSight to visualize logs as they are being produced.
    - [ ] B. 生成一个Amazon CloudWatch仪表盘，为Amazon SageMaker输出的延迟、内存利用率和CPU利用率指标创建一个单一的视图。 Generate an Amazon CloudWatch dashboard to create a single view for the latency, memory utilization, and CPU utilization metrics that are outputted by Amazon SageMaker.
    - [ ] C. 建立自定义的Amazon CloudWatch日志，然后利用Amazon ES和Kibana来查询和可视化由Amazon SageMaker生成的日志数据。 Build custom Amazon CloudWatch Logs and then leverage Amazon ES and Kibana to query and visualize the log data as it is generated by Amazon SageMaker.
    - [ ] D. 将Amazon SageMaker生成的Amazon Cloud Watch Logs发送到Amazon ES，并使用Kibana查询和可视化日志数据。 Send Amazon CloudWatch Logs that were generated by Amazon SageMaker to Amazon ES and use Kibana to query and visualize the log data.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

43. 一家制造公司有结构化和非结构化的数据存储在Amazon S3桶中。一位机器学习专家希望使用SQL对这些数据进行查询，哪种解决方案需要最少的努力才能查询这些数据？ A manufacturing company has structured and unstructured data stored in an Amazon S3 bucket. A Machine Learning Specialist wants to use SQL to run queries on this data Which solution requires the LEAST effort to be able to query this data?
    - [ ] A. 使用AWS Data Pipeline来转换数据，并使用Amazon RDS来运行查询。 Use AWS Data Pipeline to transform the data and Amazon RDS to run queries.
    - [ ] B. 使用AWS Glue对数据进行编目，并使用Amazon Athena来运行查询。 Use AWS Glue to catalogue the data and Amazon Athena to run queries.
    - [ ] C. 使用AWS Batch在数据上运行ETL和Amazon Aurora运行查询。 Use AWS Batch to run ETL on the data and Amazon Aurora to run the queries.
    - [ ] D. 使用AWS Lambda来转换数据，使用Amazon Kinesis Data Analytics来运行查询。 Use AWS Lambda to transform the data and Amazon Kinesis Data Analytics to run queries.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

44. 一位机器学习专家正在为一个应用程序开发一个定制的视频推荐模型。用于训练这个模型的数据集非常大，有数百万个数据点，托管在一个Amazon S3桶中。专家希望避免将所有这些数据加载到Amazon SageMaker笔记本实例上，因为移动这些数据需要几个小时，而且会超过笔记本实例上附加的5GB Amazon EBS卷。哪种方法允许专家使用所有的数据来训练模型？ A Machine Learning Specialist is developing a custom video recommendation model for an application. The dataset used to train this model is very large with millions of data points and is hosted in an Amazon S3 bucket. The Specialist wants to avoid loading all of this data onto an Amazon SageMaker notebook instance because it would take hours to move and will exceed the attached 5 GB Amazon EBS volume on the notebook instance. Which approach allows the Specialist to use all the data to train the model?
    - [ ] A. 将一个较小的数据子集加载到SageMaker笔记本中，并在本地进行训练。确认训练代码正在执行，并且模式参数看起来很合理。使用S3桶中的完整数据集，使用管道输入模式启动SageMaker训练作业。 Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing, and the mode parameters seem reasonable. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.
    - [ ] B. 用AWS深度学习AMI启动一个亚马逊EC2实例，并将S3桶连接到该实例。在少量的数据上进行训练，以验证训练代码和超参数。回到亚马逊Speaker，使用完整的数据集进行训练。 Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to the instance. Train on a small amount of the data to verify the training code and hyperparameters. Go back to Amazon Speaker and train using the full dataset.
    - [ ] C. 使用AWS Glue来训练一个模型，使用一小部分数据来确认数据与Amazon SageMaker兼容。使用S3桶中的完整数据集，使用管道输入模式启动SageMaker训练作业。 Use AWS Glue to train a model using a small subset of the data to confirm that the data will be compatible with Amazon SageMaker. Initiate a SageMaker training job using the full dataset from the S3 bucket using Pipe input mode.
    - [ ] D. 将一个较小的数据子集加载到SageMaker笔记本中，并在本地进行训练。确认训练代码正在执行，并且模型参数看起来很合理。启动一个带有AWS深度学习AMI的亚马逊EC2实例，并附加S3桶来训练完整的数据集。 Load a smaller subset of the data into the SageMaker notebook and train locally. Confirm that the training code is executing, and the model parameters seem reasonable. Launch an Amazon EC2 instance with an AWS Deep Learning AMI and attach the S3 bucket to train the full dataset.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

45. 一家公司正在建立一个系统来管理它存储在Amazon S3中的所有数据集。该公司希望能自动运行数据的转换工作，并维护有关数据集的元数据目录。该解决方案应该需要最少的设置和维护。哪种解决方案可以让该公司实现其目标？ A company is setting up a system to manage all of the datasets it stores in Amazon S3. The company would like to automate running transformation jobs on the data and maintaining a catalog of the metadata concerning the datasets. The solution should require the least amount of setup and maintenance. Which solution will allow the company to achieve Its goals?
    - [ ] A.  创建一个安装了Apache Hive的Amazon EMR集群。然后，创建一个Hive元存储和一个脚本，按计划运行转换工作。 Create an Amazon EMR cluster with Apache Hive installed. Then, create a Hive metastore and a script to run transformation jobs on a schedule.
    - [ ] B. 创建一个AWS Glue爬虫来填充AWS Glue数据目录。然后，编写一个AWS Glue ETL作业，并为数据转换作业设置一个时间表。 Create an AWS Glue crawler to populate the AWS Glue Data Catalog. Then, author an AWS Glue ETL job, and set up a schedule for data transformation jobs.
    - [ ] C. 创建一个安装有Apache Spark的Amazon EMR集群。然后，命一个Apache Hive元存储和一个脚本，在时间表上运行转换工作。 Create an Amazon EMR cluster with Apache Spark installed. Then, fate an Apache Hive metastore and a script to run transformation jobs on a schedule.
    - [ ] D. 创建一个AWS数据管道，对数据进行转换。然后，创建一个Apache Hive元存储和一个脚本，在时间表上运行转换工作。 Create an AWS Data Pipeline that transforms the data. Then, create an Apache Hive metastore and a script to run transformation jobs on a schedule.

    <details>
       <summary>Answer</summary>

       答案B：AWS Glue是正确的答案，因为这个选项需要最少的设置和维护，因为它是无服务器的，而且它不需要管理基础设施。A、C和D都是可以解决问题的方案，但需要更多的配置步骤，并且需要更高的运行和维护开销。

    </details>

46. 一位数据科学家在训练过程中通过改变多个参数来优化一个模型。该数据科学家观察到，在参数相同的多次运行中，损失函数收敛到不同的、但稳定的值。数据科学家应该怎么做来改善训练过程？ A Data Scientist is working on optimizing a model during the training process by varying multiple parameters. The Data Scientist observes that during multiple runs with identical parameters, the loss function converges to different, yet stable, values. What should the Data Scientist do to improve the training process?
    - [ ] A. 增加学习率。保持批次大小不变。 Increase the learning rate. Keep the batch size the same.
    - [ ] B. 减少批次大小。降低学习率。Reduce the batch size. Decrease the learning rate.
    - [ ] C. 保持批次大小不变。降低学习率。 Keep the batch size the same. Decrease the learning rate.
    - [ ] D. 不要改变学习率。增加批量大小。 Do not change the learning rate. Increase the batch size.

    <details>
       <summary>Answer</summary>

       答案B：最有可能的是，损失函数是非常弯曲的，并且有多个局部最小值，在那里训练会被卡住。减少批量大小将帮助数据科学家随机地走出局部最小值的障碍。降低学习率可以防止过度地处理全局损失函数的最小值。

    </details>

47. 一位机器学习专家正在配置Amazon SageMaker，以便多个数据科学家能够访问笔记本、训练模型和部署端点。为了确保最佳的操作性能，专家需要能够跟踪科学家部署模型的频率，部署的SageMaker端点的GPU和CPU利用率，以及调用端点时产生的所有错误。哪些服务与Amazon SageMaker集成，以跟踪这些信息？(选择两个) A Machine Learning Specialist is configuring Amazon SageMaker so multiple Data Scientists can access notebooks, train models, and deploy endpoints. To ensure the best operational performance, the Specialist needs to be able to track how often the Scientists are deploying models, GPU and CPU utilization on the deployed SageMaker endpoints, and all errors that are generated when an endpoint is invoked. Which services are integrated with Amazon SageMaker to track this information? (Choose two)
    - [ ] A. AWS CloudTrail
    - [ ] B. AWS Health
    - [ ] C. AWS Trusted Advisor
    - [ ] D. Amazon CloudWatch
    - [ ] E. AWS Config

    <details>
       <summary>Answer</summary>

       答案AD。

    </details>

48. 一家零售连锁店一直在使用Amazon Kinesis Data Firehose将采购记录从其20,000家商店的网络中摄入到Amazon S3。为了支持训练一个改进的机器学习模型，训练记录将需要新的但简单的转换，一些属性将被合并。该模型需要每天重新训练。考虑到大量的存储和传统的数据摄取，哪个变化需要最少的开发工作？ A retail chain has been ingesting purchasing records from its network of 20,000 stores to Amazon S3 using Amazon Kinesis Data Firehose. To support training an improved machine learning model, training records will require new but simple transformations, and some attributes will be combined. The model needs to be retrained daily. Given the large number of stores and the legacy data ingestion, which change will require the LEAST amount of development effort?
    - [ ] A. 要求商店切换到在AWS存储网关上捕获他们的本地数据，以加载到Amazon S3，然后使用AWS Glue来进行转换。 Require that the stores to switch to capturing their data locally on AWS Storage Gateway for loading into Amazon S3, then use AWS Glue to de the transformation.
    - [ ] B. 部署一个运行Apache Spark和转换逻辑的Amazon EMR集群，并让该集群每天在Amazon S3的累积记录上运行，将新的/转换的记录输出到Amazon S3。 Deploy an Amazon EMR cluster running Apache Spark with the transformation logic, and have the cluster run each day on the accumulating records in Amazon S3, outputting new/transformed records to Amazon S3.
    - [ ] C. 建立一个具有转换逻辑的亚马逊EC2实例群，让它们转换亚马逊S3上积累的数据记录，并将转换后的记录输出到亚马逊S3。 Spin up a fleet of Amazon EC2 instances with the transformation logic, have them transform the data records accumulating on Amazon S3, and output the transformed records to Amazon S3.
    - [ ] D. 在Kinesis Data Firehose流的下游插入一个Amazon Kinesis Data Analytics流，使用SQL将原始记录属性转化为简单的转化值。 Insert an Amazon Kinesis Data Analytics stream downstream of the Kinesis Data Firehose stream that transforms raw record attributes into simple transformed values using SQL.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

49. 一位机器学习专家正在构建一个卷积神经网络（CNN），它将对10种类型的动物进行分类。该专家在一个神经网络中建立了一系列的层，它将接收一个动物的输入图像，通过一系列的卷积层和池化层，最后再通过一个有10个节点的密集全连接层。专家希望从神经网络中得到一个输出，这个输出是输入图像属于10个类别中每个类别的概率分布，哪个函数会产生所需的输出？ A Machine Learning Specialist is building a convolutional neural network (CNN) that will classify 10 types of animals. The Specialist has built a series of layers in a neural network that will take an input image of an animal, pass it through a series of convolutional and pooling layers, and then finally pass it through a dense and fully connected layer with 10 nodes. The Specialist would like to get an output from the neural network that is a probability distribution of how likely it is that the input image belongs to each of the 10 classes Which function will produce the desired output?
    - [ ] A Dropout
    - [ ] B. Smooth L1 loss
    - [ ] C. Softmax
    - [ ] D. Rectified linear units (ReLU)

    <details>
       <summary>Answer</summary>

       答案C, Softmax分类器可以理解为逻辑回归分类器面对多分类问题的一般化归纳。

    </details>

50. 一位机器学习专家训练了一个回归模型，但第一个迭代需要优化。专家需要了解该模型是更频繁地高估还是低估了目标值。专家可以使用什么选项来确定它是否高估或低估了目标值？ A Machine Learning Specialist trained a regression model, but the first iteration needs optimizing. The Specialist needs to understand whether the model is more frequently overestimating or underestimating the target. What option can the Specialist use to determine whether it is overestimating underestimating the target value?
    - [ ] A. 均方根误差(RMSE) Root Mean Square Error (RMSE)
    - [ ] B. 残差图 Residual plots
    - [ ] C. 曲线下面积 Area under the curve
    - [ ] D. 混淆矩阵 Confusion matrix

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

51. 一家公司希望将用户行为分类为欺诈行为或正常行为。根据内部研究，一位机器学习专家希望建立一个基于两个特征的二进制分类器：账户年龄和交易月份。这些特征的类别分布如图所示。基于这些信息，哪个模型对欺诈类的召回率最高？ A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a Machine Learning Specialist would like to build a binary classifier based on two features: age of account and transaction month. The class distribution for these features is illustrated in the figure provided. Based on this information, which model would have the HIGHEST recall with respect to the fraudulent class?

    ![51](./img/51.png)

    - [ ] A. 决策树 Decision tree
    - [ ] B. 线性支持向量机(SVM) Linear support vector machine (SVM)
    - [ ] C. 奈何贝叶斯分类器 Naive Bayesian classifier
    - [ ] D. 带有西格玛激活函数的单感知器 Single Perceptron with sigmoidal activation function
  
    <details>
       <summary>Answer</summary>

       答案A。

    </details>

52. 一位机器学习专家使用Amazon SageMaker，以ROC曲线下面积（AU）为目标指标，为一个基于树的集合模型启动了超参数调整工作。这个工作流程最终将被部署在一个管道中，每天晚上重新训练和调整超参数，为每4小时变质的数据建立点击模型。为了减少训练这些模型所需的时间，并最终降低成本，专家希望重新配置输入的超参数范围，哪种可视化方法可以实现这一目标？ A Machine Learning Specialist kicks off a hyperparameter tuning job for a tree-based ensemble model using Amazon SageMaker with Area Under the ROC Curve (AU) as the objective metric. This workflow will eventually be deployed in a pipeline that retrains and tunes hyperparameters each night to model click-through on data that goes stale every 4 hours. With the goal of decreasing the amount of me it takes to train these models, and ultimately to decrease costs, the Specialist wants to reconfigure the input hyperparameter range(s) Which visualization will accomplish this?
    - [ ] A. 显示最重要的输入特征是否为高斯的直方图。 A histogram showing whether the most important input feature is Gaussian.
    - [ ] B. 使用t分布的随机邻接嵌入（t-SNE）的散点图，将大量的输入变量在一个更容易阅读的维度上可视化。 A scatter plot with points colored by target variable that uses t-distributed Stochastic Neighbor Embedding (t-SNE) to visualize the large number of input variables in an easier-to-read dimension.
    - [ ] C. 显示每个训练迭代中目标指标性能的散点图。 A scatter plot showing the performance of the objective metric over each training iteration
    - [ ] D. 显示最大树深和目标度量之间的相关性的散点图。 A scatter plot showing the correlation between maximum tree depth and the objective metric.
  
    <details>
       <summary>Answer</summary>

       答案D。

    </details>

53. 一位机器学习专家正在创建一个新的自然语言处理应用程序，处理一个由一百万句子组成的数据集。其目的是运行Word2Vec来生成句子的嵌入，并实现不同类型的预测。下面是数据集中的一个例子。"棕色的狐狸跳过懒惰的狗"。以下哪些是专家需要执行的操作，以正确消毒和准备数据的可重复方式？(选择三个) A Machine Learning Specialist is creatin a new natural language processing application that processes a dataset comprised of one million sentences. The aim is to then run Word2Vec to generate embeddings of the sentences and enable different types of predictions. Here is an example from the dataset: “The quck BROWN FOX jumps over the lazy dog.” Which of the following are the operations the specialist needs to perform to correctly sanitize and prepare the data a repeatable manner? (Choose three)
    - [ ] A. 进行语义部分标记，只保留动作动词和名词。 Perform part-of-speech tagging and keep the action verb and the nouns only.
    - [ ] B. 通过使句子小写来规范所有单词。 Normalize all words by making the sentence lowercase.
    - [ ] C. 使用英语停止词词典删除停止词。 Remove stop words using an English stop word dictionary.
    - [ ] D. 将 "quck "的排版修改为 "quick"。 Correct the typography on "quck" to "quick.
    - [ ] E. 对句子中的所有单词进行单热编码。 One-hot encode all words in the sentence.
    - [ ] F. 将该句子标记为单词。 Tokenize the sentence into words.

    <details>
       <summary>Answer</summary>

       答案BCF。

    </details>

54. A Data Scientist is evaluating different binary classification models. A false positive result is 5 times more expensive (from a business perspective) than a false negative result. The models should be evaluated based on the following criteria: 1) Must have a recall rate of at least 80%. 2) Must have a false positive rate of 10 or less. 3) Must minimize business costs. After creating each binary classification model, the Data Scientist generates the corresponding confusion matrix Which confusion matrix represents the model that satisfies the requirements?
    - [ ] A.TN=91, FP=9, FN=22, TP=78
    - [ ] B.TN=99, FP=1, FN=21, TP=79
    - [ ] C.TN=96, FP=4, FN=10, TP=90
    - [ ] D.TN=98, FP=2, FN=18, TP=82

    <details>
       <summary>Answer</summary>

       答案D。

       ![54](img/54.png)

    </details>

55. 一位数据科学家使用逻辑回归来建立一个欺诈检测模型。虽然模型的准确率为99%，但90%的欺诈案件没有被模型发现。什么行动可以明确地帮助模型检测出10%以上的欺诈案件？ A Data Scientist uses logistic regression to build a fraud detection model. While the model accuracy is 99%, 90% of the fraud cases are not detected by the model. What action will definitively help the model detect more than 10% of fraud cases?
    - [ ] A. 使用不足的采样来平衡数据集 Using under sampling to balance the dataset
    - [ ] B. 降低类别概率阈值 Decreasing the class probability threshold
    - [ ] C. 使用正则化来减少过度拟合 Using regularization to reduce overfitting
    - [ ] D. 使用过量取样来平衡数据集 Using oversampling to balance the dataset

    <details>
       <summary>Answer</summary>

       答案B：降低类别概率阈值使模型更加敏感，因此，将更多的案例标记为阳性类别，在这种情况下就是欺诈。这将增加欺诈检测的可能性。然而，它是以降低精确度为代价的。

    </details>

56. 机器学习专家正在建立一个模型，以预测基于广泛的经济行为者的未来就业率，同时探索数据，专家注意到输入特征的大小差异很大。专家不希望幅度较大的变量在模型中占主导地位。专家应该做什么来准备模型训练的数据？ Machine Learning specialist is building a model to predict future employment rates based on a wide range of economic actors while exploring the data, the Specialist notices that the magnitude of the input features vary greatly. The Specialist does not want variables with a larger magnitude to dominate the model. What should the Specialist do to prepare the data for model training?
    - [ ] A. 应用量化分档法，将数据分为分类分档，通过用分布代替幅度来保持数据中的任何关系。 Apply quantile binning to group the data into categorical bins to keep any relationships in the data by replacing the magnitude with distribution.
    - [ ] B. 应用笛卡尔乘积转换，创建独立于幅度的新组合字段。 Apply the Cartesian product transformation to create new combinations of fields that are independent of the magnitude.
    - [ ] C. 应用归一化，确保每个字段的平均值为0，方差为1，以消除任何重要的幅度。 Apply normalization to ensure each field will have a mean of 0 and a variance of I to remove any significant magnitude.
    - [ ] D. 应用正交稀疏大图（OSB）变换，应用固定大小的滑动窗口来产生类似幅度的新特征。 Apply the orthogonal sparse bigram (OSB) transformation to apply a fixed-size sliding window to generate new features of a similar magnitude.

    <details>
       <summary>Answer</summary>

       答案C: [ref](https://docs.aws.amazon.com/zh_cn/machine-learning/latest/dg/data-transformations-reference.html)

    </details>

57. 一位机器学习专家必须建立一个流程，使用Amazon Athena查询Amazon S3上的数据集。该数据集包含超过800.000条记录，以纯文本CSV文件形式存储。每条记录包含200列，大小约为1.5MB。大多数查询将只跨越5到10列。机器学习专家应该如何转换数据集以减少查询的运行时间？ A Machine Learning Specialist must build out a process to query a dataset on Amazon S3 using Amazon Athena. The dataset contains more than 800.000 records stored as plaintext CSV files. Each record contains 200 columns and is approximately 1. 5 MB in size. Most queries will span 5 to 10 columns only. How should the Machine Learning Specialist transform the dataset to minimize query runtime?
    - [ ] A. Convert the records to Apache Parquet format
    - [ ] B. Convert the records to JSON format
    - [ ] C. Convert the records to GZIP CSV format
    - [ ] D. Convert the records to XML format

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

58. 一位机器学习专家正在开发一个包含多个ETL工作的日常ETL工作流程。该工作流程由以下过程组成。1）一旦数据被上传到Amazon S3，立即启动工作流程。2) 当所有的数据集在Amazon S3中可用时，启动ETL作业，将上传的数据集与已经存储在Amazon S3中的多个TB级的数据集连接起来。3) 将连接数据集的结果存储在Amazon S3中。4) 如果其中一个作业失败，向管理员发送一个通知。哪种配置可以满足这些要求？ A Machine Learning Specialist is developing a daily ETL workflow containing multiple ETL jobs. The workflow consists of the following processes: 1) Start the workflow as soon as data is uploaded to Amazon S3. 2) When all the datasets are available in Amazon S3, start an ETL job to join the uploaded datasets with multiple terabyte-sized datasets already stored n Amazon S3. 3) Store the results of joining datasets in Amazon S3. 4) If one of the jobs fails, send a notification to the Administrator. Which configuration will meet these requirements?
    - [ ] A. 使用AWS Lambda来触发AWS Step Functions工作流，以等待数据集在Amazon S3中完成上传。使用AWS Glue来连接数据集。使用Amazon CloudWatch警报，在失败的情况下向管理员发送SNS通知。 Use AWS Lambda to trigger an AWS Step Functions workflow to wait for dataset uploads to complete in Amazon S3. Use AWS Glue to join the datasets. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.
    - [ ] B. 使用AWS Lambda开发ETL工作流程，启动Amazon SageMaker笔记本实例。使用生命周期配置脚本来连接数据集并将结果持久化在Amazon S3中。使用Amazon Cloud Watch警报，在发生故障时向管理员发送SNS通知。 Develop the ETL workflow using AWS Lambda to start an Amazon SageMaker notebook instance. Use a lifecycle configuration script to join the datasets and persist the results in Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure
    - [ ] C. 使用AWS Batch开发ETL工作流程，当数据上传到Amazon S3时触发ETL工作的开始。使用AWS Glue在Amazon S3的数据集上。使用Amazon Cloud Watch警报，在发生故障时向管理员发送SNS通知。 Develop the ETL workflow using AWS Batch to trigger the start of ETL jobs when data is uploaded to Amazon S3. Use AWS Glue to on the datasets in Amazon S3. Use an Amazon Cloud Watch alarm to send an SNS notification to the Administrator in the case of a failure.
    - [ ] D. 使用AWS Lambda连锁其他Lambda函数，在数据上传到Amazon S3后立即读取和加入Amazon S3中的数据集。使用Amazon CloudWatch警报，在发生故障时向管理员发送SNS通知。 Use AWS Lambda to chain other Lambda functions to read and join the datasets in Amazon S3 as soon as the data is uploaded to Amazon S3. Use an Amazon CloudWatch alarm to send an SNS notification to the Administrator in the case of a failure.

    <details>
       <summary>Answer</summary>

       答案A: [ref](https://aws.amazon.com/cn/step-functions/use-cases/)

    </details>

59. 一个机构在一个国家内收集人口普查信息，以确定各省市的医疗保健和社会项目需求。普查表收集了每个公民对大约500个问题的回答。哪种算法组合可以提供适当的洞察力？(选择两个) An agency collects census information within a country to determine healthcare and social program needs by province and city. The census form collects response for approximately 500 questions from each citizen. Which combination of algorithms would provide the appropriate insights? (Select TWO)
    - [ ] A. 因式分解机（FM）算法。 The factorization machines (FM) algorithm.
    - [ ] B. Latent Dirichlet Allocation（LDA）算法。 The Latent Dirichlet Allocation (LDA) algorithm.
    - [ ] C. 主成分分析（PCA）算法。 The principal component analysis (PCA)algorithm.
    - [ ] D. k-means算法。 The k-means algorithm.
    - [ ] E. 随机切割森林（RCF）算法。 The Random Cut Forest (RCF) algorithm.

    <details>
       <summary>Answer</summary>

       答案CD。

    </details>

60. 一家消费品制造商有以下产品在销售。1）34种不同的牙膏品种。2) 48种不同的牙刷。3) 48种不同的漱口水。所有这些产品的整个销售历史都可以在Amazon S3中找到。目前，该公司正在使用定制的自回归综合移动平均模型（ARIMA）来预测这些产品的需求。该公司希望预测即将推出的新产品的需求。机器学习专家应该应用哪种解决方案？ A are consumer goods manufacturer has the following products on sale: 1) 34 different toothpaste variants. 2) 48 different toothbrush variants. 3) 48 different mouthwash variants. The entire sales history of all these products is available in Amazon S3. Currently, the company is using custom-built autoregressive integrated moving average (ARIMA) models to forecast demand for these products. The company wants to predict the demand for a new product that will soon be launched. Which solution should a Machine Learning Specialist apply?
    - [ ] A. 训练一个自定义的ARIMA模型来预测新产品的需求。 Train a custom ARIMA model to forecast demand for the new product.
    - [ ] B. 训练一个Amazon SageMaker DeepAR算法来预测新产品的需求。 Train an Amazon SageMaker DeepAR algorithm to forecast demand for the new product.
    - [ ] C. 训练一个Amazon SageMaker k-means聚类算法来预测新产品的需求。Train an Amazon SageMaker k-means clustering algorithm to forecast demand for the new product.
    - [ ] D. 训练一个自定义的XGBboost模型来预测新产品的需求。 Train a custom XGBboost model to forecast demand for the new product.

    <details>
       <summary>Answer</summary>

       答案B：Amazon SageMaker DeepAR预测算法是一种监督学习算法，用于使用循环神经网络（RNN）预测标量（一维）时间序列。经典的预测方法，如自回归综合移动平均法（ARIMA）或指数平滑法（ETS），对每个单独的时间序列拟合一个模型。然后他们使用该模型将时间序列推断到未来。

    </details>

61. 一位机器学习专家将一个数据集上传到Amazon S3桶中，并使用AWS KMS进行服务器端加密保护。ML专家应该如何定义Amazon SageMaker笔记本实例，以便它可以从Amazon S3读取相同的数据集？ A Machine Learning Specialist uploads a dataset to an Amazon S3 bucket protected with server-side encryption using AWS KMS. How should the ML Specialist define the Amazon SageMaker notebook instance so it can read the same dataset from Amazon S3?
    - [ ] A. 定义安全组，允许所有HTTP入站/出站流量，并将这些安全组分配给Amazon SageMaker笔记本实例。 Define security group(s) to allow all HTTP inbound/outbound traffic and assign those security group(s) to the Amazon SageMaker notebook instance.
    - [ ] B. 配置Amazon SageMaker笔记本实例，使其能够访问VPC。在KMS密钥策略中为笔记本的KMS角色授予权限。 Configure the Amazon SageMaker notebook instance to have access to the VPC. Grant permission in the KMS key policy to the notebooks KMS role.
    - [ ] C. 为Amazon SageMaker笔记本分配一个IAM角色，使其具有对数据集的S3读取权限。在KMS密钥策略中为该角色授予权限。 Assign an IAM role to the Amazon SageMaker notebook with S3 read access to the dataset. Grant permission in the KMS key policy to that role.
    - [ ] D. 将用于在Amazon S3中加密数据的相同KMS密钥分配给Amazon SageMaker笔记本实例。 Assign the same KMS key used to encrypt data in Amazon S3 to the Amazon SageMaker notebook instance.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

62. 一家公司对建立一个欺诈检测模型感兴趣。目前，由于欺诈案件的数量很少，数据科学家没有足够的信息量。哪种方法最可能检测到最多数量的有效欺诈案件？ A company is interested in building a fraud detection model. Currently the Data Scientist does not have a sufficient amount of information due to the low number of fraud cases. Which method is MOST likely to detect the GREATEST number of valid fraud cases?
    - [ ] A. 使用引导法进行过度取样 Oversampling using bootstrapping
    - [ ] B. 低度取样 Undersampling
    - [ ] C. 使用SMOTE的过度取样 Oversampling using SMOTE
    - [ ] D. 类别权重调整 Class weight adjustment

    <details>
       <summary>Answer</summary>

       答案C：对于没有完全填充的数据集，SMOTE通过向少数类添加合成数据点来增加新的信息。在这种情况下，这种技术将是最有效的。

    </details>

63. 一位机器学习工程师正在准备一个数据框架，用于使用Amazon SageMaker近似学习者算法的监督学习请求。该ML工程师注意到目标标签类是高度不平衡的，并且多个特征列包含缺失值。在整个数据框架中，缺失值的比例小于5。M工程师应该做什么来减少由于缺失值造成的偏差？ A Machine Learning Engineer is preparing a data frame for a supervised learning ask with the Amazon SageMaker near Learner algorithm. The ML Engineer notices the target label classes are highly imbalanced and multiple feature columns contain missing values. The proportion of missing values across the entire data frame is less than 5. What should the M Engineer do to minimize bias due to missing values?
    - [ ] A. 用同一行中非缺失值的平均数或中位数来替换每个缺失值。 Replace each missing value by the mean or median across non-missing values in same row.
    - [ ] B. 删除包含缺失值的观察值，因为这些观察值只占数据的50％以下。 Delete observations that contain missing values because these represent less than 50 of the data.
    - [ ] C. 用同一列中非缺失值的平均数或中位数替换每个缺失值。 Replace each missing value by the mean or median across non-missing values in the same column.
    - [ ] D. 对于每个特征，使用基于其他特征的监督学习来近似计算缺失值。 For each feature, approximate the missing values using supervised learning based on other features.

    <details>
       <summary>Answer</summary>

       答案D：使用监督学习来预测基于其他特征值的缺失值。不同的监督学习方法可能有不同的表现，但任何正确实施的监督学习方法都应该提供与平均数或中位数近似相同或更好的近似，如回答A和C中提出的监督学习应用于缺失值的推算是一个活跃的研究领域。

    </details>

64. 一位机器学习专家使用少量的数据样本为一家公司完成了概念验证，现在专家准备使用Amazon SageMaker在AWS中实施一个端到端的解决方案，历史训练数据存储在Amazon RDS中。专家应该使用哪种方法来训练使用该数据的模型？ A Machine Learning Specialist has completed a proof of concept for a company using a small data sample, and now the Specialist is ready to implement an end-to-end solution in AWS using Amazon SageMaker The historical training data is stored in Amazon RDS. Which approach should the Specialist use for training a model using that data?
    - [ ] A. 在笔记本内写一个与SQL数据库的直接连接，把数据拉进来。 Write a direct connection to the SQL database within the notebook and pull data in.
    - [ ] B. 使用AWS数据管道将数据从Microsoft SQL Server推送到Amazon S3，并在笔记本中提供S3位置。 Push the data from Microsoft SQL Server to Amazon S3 using an AWS Data Pipeline and provide the S3 location within the notebook.
    - [ ] C. 将数据移到亚马逊DynamoDB，并在笔记本内设置与DynamoDB的连接，以拉入数据。 Move the data to Amazon DynamoDB and set up a connection to DynamoDB within the notebook to pull data in.
    - [ ] D. 使用AWS DMS将数据移到Amazon ElastiCache，并在笔记本内设置一个连接，以拉入数据，实现快速访问。 Move the data to Amazon ElastiCache using AWS DMS and set up a connection within the notebook to pull data in for fast access.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

65. 一位机器学习专家为一个在线购物网站接收客户数据。这些数据包括人口统计学，过去的访问，和地区信息。专家必须开发一种机器学习方法来识别客户的购物模式、偏好和趋势，以加强网站的服务和智能推荐。该专家应该推荐哪种解决方案？ A Machine Learning Specialist receives customer data for an online shopping website. The data includes demographics, past visits, and locality information. The Specialist must develop a machine learning approach to identify the customer shopping patterns, preferences, and trends to enhance the website-for better service and smart recommendations. Which solution should the Specialist recommend?
    - [ ] A. 对于给定的离散数据集合，采用Latent Dirichlet Allocation（LDA）来识别客户数据库中的模式。 Latent Dirichlet Allocation (LDA) for the given collection of discrete data to identify patterns in the customer database.
    - [ ] B. 一个至少有三层和随机初始权重的神经网络来识别客户数据库中的模式。 A neural network with a minimum of three layers and random initial weights to identify patterns in the customer database.
    - [ ] C. 基于用户互动和相关关系的协同过滤，以识别客户数据库中的模式。 Collaborative filtering based on user interactions and correlations to identify patterns in the customer database.
    - [ ] D. 通过随机子样本的随机切割森林（RCF）来识别客户数据库中的模式。 Random Cut Forest (RCF) over random subsamples to identify patterns in the customer database.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>
