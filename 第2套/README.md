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

21. 数据科学家正在一个有多个类别的数据集上训练一个多层感知器（MLP）。与数据集中的其他类相比，感兴趣的目标类是独特的，但它没有达到可接受的召回指标。数据科学家已经尝试改变MPS隐藏层的数量和大小，但结果没有明显改善。必须尽快实施改善召回率的解决方案。应该使用哪些技术来满足这些要求？ Data Scientist is training a multilayer perceptron (MLP) on a dataset with multiple classes. The target class of interest is unique compared to the other classes within the dataset, but it does not achieve an acceptable recall metric. The Data Scientist has already tried varying the number and size of the MPS hidden layers, which has not significantly improved the results. A solution to improve recall must be implemented as quickly as possible. Which techniques should be used to meet these requirements?
    - [ ] A. 使用Amazon Mechanical Turk收集更多数据，然后重新训练。 Gather more data using Amazon Mechanical Turk and then retrain.
    - [ ] B. 训练一个异常检测模型而不是MLP。 Train an anomaly detection model instead of an MLP.
    - [ ] C. 训练一个XGBboost模型而不是MLP。 Train an XGBboost model instead of an MLP.
    - [ ] D. 在MLP的损失函数中加入类的权重，然后再重新训练。 Add class weights to the MLP's loss function and the retrain.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

22. 一个网络安全供应商需要从世界各地运行的数以百万计的端点中获取遥测数据。这些数据每30秒以包含50个字段的记录形式传输。每条记录的大小为1KB。这些数据是使用亚马逊Kinesis数据流摄入的。在Amazon Athena查询时，需要每小时对可用记录进行汇总。针对Athena运行的查询将针对包括7-12列数据的不同集合。哪种解决方案在转换和存储摄入的数据时涉及的定制化程度最低？ A network security vendor needs to ingest telemetry data from millions of endpoints running all over the world. This data is transmitted every 30 seconds in the form of records containing 50 fields. Each record is up to 1KB in size. The data is being ingested using Amazon Kinesis Data streams. Hourly summaries of the available records are needed for querying in Amazon Athena. The queries running against Athena will target different sets that include 7-12 columns of data. Which solution involves the LEAST amount of customization for transforming and storing the ingested data?
    - [ ] A. 配置Kinesis数据流，通过流将数据发送到AWS Lambda。配置Lambda每小时汇总数据，并将其发送到Amazon Kinesis Data Firehose交付流，配置为将数据以Apache Parquet格式存储在Amazon S3。 Configure Kinesis Data Streams to send the data through a stream to AWS Lambda. configure Lambda to aggregate the data hourly and send it to an Amazon Kinesis Data Firehose delivery stream configured to store the data in Amazon S3 in Apache Parquet format.
    - [ ] B. 配置Kinesis数据流，通过流来发送数据，每小时将数据汇总到Amazon Kinesis Data Firehose交付流。配置交付流，将数据发送到Amazon EMR，并将数据以Apache Parquet格式存储在Amazon S3中。 Configure Kinesis Data Streams to send the data through a stream to aggregate the data hourly to an Amazon Kinesis Data Firehose delivery stream. Configure the delivery stream to send the data to Amazon EMR and store the data in Amazon S3 in Apache Parquet format.
    - [ ] C. 配置Kinesis数据流，通过流来发送数据，每小时将数据汇总到Amazon Kinesis数据分析。配置Kinesis Data Analytics，将数据处理到Amazon Kinesis Data Firehose交付流，配置为将数据以Apache Parquet格式存储在Amazon S3。 Configure Kinesis Data Streams to send the data through a stream to aggregate the data hourly to Amazon Kinesis Data Analytics. Configure Kinesis Data Analytics to process the data to an Amazon Kinesis Data Firehose delivery stream configured to store the data in Amazon S3 in Apache Parquet format.
    - [ ] D. 配置Kinesis数据流，通过一个流来发送数据，每小时将数据汇总到Amazon Kinesis Data Firehose交付流中。配置AWS Lambda来处理数据，并将数据以Apache Parquet格式存储在Amazon S3中。 Configure Kinesis Data Streams to send the data through a stream to aggregate the data hourly to an Amazon Kinesis Data Firehose delivery stream. Configure AWS Lambda to process the data and store the data in Amazon S3 in Apache Parquet format.
  
    <details>
       <summary>Answer</summary>

       答案C: [ref](https://docs.aws.amazon.com/zh_cn/kinesisanalytics/latest/sqlref/analytics-sql-reference.html)

    </details>

23. 一个数据科学家使用Amazon SageMaker笔记本实例来进行数据探索和分析。这需要在笔记本实例上安装Amazon SageMaker上没有的某些Python包。机器学习专家如何确保所需的包在笔记本实例上自动可用，供数据科学家使用？ A Data Scientist uses an Amazon SageMaker notebook instance to conduct data exploration and analysis. This requires certain Python packages that are not natively available on Amazon SageMaker to be installed on the notebook instance. How can a Machine Learning Specialist ensure that required packages are automatically available on the notebook instance for the Data Scientist to use?
    - [ ] A. 在底层的Amazon EC实例上安装AWS系统管理器代理，并使用系统管理器自动化来执行包安装命令。 Install AWS System Manager Agent on the underlying Amazon EC instance and use Systems Manager Automation to execute the package installation commands.
    - [ ] B. 创建一个Jupyter笔记本文件（.ipynb），其中包含要执行的包安装命令的单元，并将该文件放在每个Amazon SageMaker笔记本实例的`/etc/init`目录下。 Create a Jupyter notebook file (.ipynb) with cells containing the package installation commands to execute and place the file under the `/etc/init` directory of each Amazon SageMaker notebook instance.
    - [ ] C. 在Jupyter笔记本控制台中使用conda包管理器，将必要的conda包应用于笔记本的默认内核。 Use the conda package manager from within the Jupyter notebook console to apply the necessary conda packages to the default kernel of the notebook.
    - [ ] D. 使用包安装命令创建Amazon SageMaker生命周期配置，并将生命周期配置分配给笔记本实例。 Create an Amazon SageMaker lifecycle configuration with package installation commands and assign the lifecycle configuration to the notebook instance.

    <details>
       <summary>Answer</summary>

       答案D: [ref](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/notebook-lifecycle-config.html)

    </details>

24. 一位机器学习专家正试图建立一个线性回归模型。仅从显示的残差图来看，该模型最可能出现的问题是什么？ A Machine Learning Specialist is attempting to build a linear regression model. Given the displayed residual plot only, what is the MOST likely problem with the model?

    ![89](img/89.png)

    - [ ] A. 线性回归是不合适的。残差不具有恒定方差 Linear regression is inappropriate. The residuals do not have constant variance
    - [ ] B. 线性回归是不恰当的。基础数据有离群值 Linear regression is inappropriate. The underlying data has outliers
    - [ ] C. 线性回归是合适的。残差的平均值为零 Linear regression is appropriate. The residuals have a zero mean
    - [ ] D. 线性回归是合适的。残差有恒定的方差 Linear regression is appropriate. The residuals have constant variance

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

25. 一家报纸出版商有一个客户数据表，该表由几个数字和分类特征组成，如年龄和教育历史，以及订阅状态。该公司希望建立一个有针对性的营销模型，以根据表格数据预测订阅状态，哪个Amazon SageMaker内置算法应该被用来建立有针对性的营销模型？ A newspaper publisher has a table of customer data that consists of several numerical and categorical features, such as age and education history, as well as subscription status. The company wants to build a targeted marketing model for predicting the subscription status based on the table data Which Amazon SageMaker built-in algorithm should be used to model the targeted marketing?
    - [ ] A. Random Cut Forest (RCF)
    - [ ] B. XGBboost
    - [ ] C Neural Topic Model (NTM)
    - [ ] D. DeepAR forecasting

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

26. 一位机器学习专家正在研究一个线性回归模型，并注意到该模型正在过度拟合。该专家应用L1正则化参数，并再次运行模型。不幸的是，这导致所有特征的权重为零。机器学习专家应该怎么做来改善模型的结果？ A Machine Learning Specialist is working on a linear regression model and notices the model is overfitting. The specialist applies an L1 regularization parameter and runs the model again. Unfortunately, this results in all features having zero weights. What should the Machine Learning Specialist do to improve the model results?
    - [ ] A. 增加L1正则化参数。 Increase the L1 regularization parameter.
    - [ ] B. 减少L1正则化参数。 Decrease the L1 regularization parameter.
    - [ ] C. 增加L2正则化参数。 Increase the L2 regularization parameter.
    - [ ] D. 减少L2正则化参数。 Decrease the L2 regularization parameter.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

27. 一位机器学习专家为一家水果加工公司工作，需要建立一个系统，将苹果分为三种类型。该专家收集了一个数据集，其中包含每种苹果的150张图片，并在一个神经网络上应用了转移学习，该网络是在ImageNet上用这个数据集预训练的。该公司要求至少有85%的准确率才能利用该模型。经过详尽的网格搜索，最佳超参数产生如下。训练集上的准确率为68%。在验证集上的准确率为67%。机器学习专家可以做些什么来提高系统的准确性？ A machine learning specialist works for a fruit processing company and needs to build a system that categorizes apples into three types. The specialist has collected a dataset that contains 150 images for each type of apple and applied transfer learning on a neural network that was pretrained on ImageNet with this dataset. The company requires at least 85% accuracy to make use of the model. After an exhaustive grid search, the optimal hyperparameters produced the following: 68% accuracy on the training set. 67% accuracy on the validation set. What can the machine learning specialist do to improve the system' s accuracy?
    - [ ] A. 把模型上传到Amazon SageMaker笔记本实例，并使用Amazon SageMaker HPO功能来优化模型的超参数。 Upload the model to an Amazon SageMaker notebook instance and use the Amazon SageMaker HPO feature to optimize the model’s hyperparameters.
    - [ ] B. 向训练集添加更多的数据，并使用转移学习重新训练模型以减少偏差。 Add more data to the training set and retrain the model using transfer learning to reduce the bias.
    - [ ] C. 使用具有更多层的神经网络模型，在ImageNet上进行预训练，并应用转移学习来增加方差。 Use a neural network model with more layers that are pretrained on ImageNet and apply transfer learning to increase the variance.
    - [ ] D. 使用当前的神经网络架构训练一个新的模型。 Train a new model using the current neural network architecture.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

28. 一位数据科学家已经探索并净化了一个数据集，为监督学习任务的建模阶段做准备。统计学上的分散性在不同的特征之间会有很大的差异，有时会有几个数量级的差异。在进入建模阶段之前，数据科学家希望确保对生产数据的预测性能尽可能准确。数据科学家应该采取哪一序列的步骤来满足这些要求？ A Data Scientist has explored and sanitized a dataset in preparation for the modeling phase of a supervised learning task. The statistical dispersion can vary widely between features, sometimes by several orders of magnitude. Before moving on to the modeling phase, the Data Scientist wants to ensure that the prediction performance on the production data is as accurate as possible. Which sequence of steps should the Data Scientist take to meet these requirements?
    - [ ] A. 对数据集应用随机抽样。然后将数据集分成训练集、验证集和测试集。 Apply random sampling to the dataset. Then split the dataset into training, validation, and test sets
    - [ ] B. 将数据集分成训练、验证和测试集。然后重新调整训练集的比例，并对验证集和测试集采用同样的比例。 Split the dataset into training, validation, and test sets. Then rescale the training set and apply the same scaling to the validation and test sets
    - [ ] C. 重新调整数据集的比例。然后将数据集分成训练集、验证集和测试集。 Rescale the dataset. Then split the dataset into training, validation, and test sets
    - [ ] D. 将数据集分成训练集、验证集和测试集。然后独立地重新调整训练集、验证集和测试集的比例。 Split the dataset into training, validation, and test sets. Then rescale the training set, the validation set, and the test set independently.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

29. 一位数据科学家正在建立一个线性回归模型，并将使用产生的P值来评估每个系数的统计意义。在检查数据集时，数据科学家发现大多数特征都是正态分布。图中显示了数据集中的一个特征的图。为了满足线性回归模型的统计假设，数据科学家应该应用什么转换？ A Data Scientist is building a linear regression model and will use resulting p-values o evaluate the statistical significance of each coefficient. Upon inspection of the dataset the Data Scientist discovers that most of the features are normally distributed. The plot of one feature in the dataset is shown in the graphic. What transformation should the Data Scientist apply to satisfy the statistical assumptions of the linear regression model?

    ![94](img/94.png)

    - [ ] A. 指数转换 Exponential transformation
    - [ ] B. 对数转换 Logarithmic transformation
    - [ ] C. 多项式转换 Polynomial transformation
    - [ ] D. 正弦波变换 Sinusoidal transformation

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

30. 某制造商经营着大量的工厂，供应链关系复杂，一台机器的意外停机会导致几个工厂的生产停止。一台机器的传感器读数可以包括多达200个数据点，包括温度、电压、振动、RPMS和压力读数。为了收集这些传感器数据，制造商在整个工厂部署了Wi-Fi和ANS。即使许多工厂地点没有可靠的或高速的互联网连接，制造商仍希望保持近乎实时的推理能力。 该模型的哪种部署架构能解决这些业务需求？ A manufacturer is operating a large number of factories with a complex supply chain relationship where unexpected downtime of a machine can cause production to stop at several factories A data scientist wants to analyze sensor data from the factories to identify equipment in need of preemptive maintenance and then dispatch a service team to prevent unplanned downtime. The sensor readings from a single machine can include up to 200 data points including temperatures, voltages, vibrations, RPMS, and pressure readings. To collect this sensor data, the manufacturer deployed Wi-Fi and ANS across the factories. Even though many factory locations do not have reliable or high-speed internet connectivity, the manufacturer would like to maintain near-real-time inference capabilities Which deployment architecture for the model will address these business requirements?
    - [ ] A. 在Amazon SageMaker中部署该模型 通过该模型运行传感器数据，预测哪些机器需要维护。 Deploy the model in Amazon SageMaker Run sensor data through this model to predict which machines need maintenance.
    - [ ] B. 在每个工厂的AWS IoT Greengrass上部署该模型 通过该模型运行传感器数据来推断哪些机器需要维护。 Deploy the model on AWS IoT Greengrass in each factory Run sensor data through this model to infer which machines need maintenance.
    - [ ] C. 将该模型部署到Amazon SageMaker的批量转换作业中。在每日批处理报告中生成推断，以确定需要维护的机器。 Deploy the model to an Amazon SageMaker batch transformation job. Generate inferences in a daily batch report to identify machines that need maintenance.
    - [ ] D. 在Amazon SageMaker中部署该模型，并使用OT规则将数据写入Amazon DynamoDB表。用AWS Lambda函数从该表消耗一个DYNAMODB流来调用端点。 Deploy the model in Amazon SageMaker and use an IoT rule to write data to an Amazon DynamoDB table. Consume a DynamoDB stream from the table with an AWS Lambda function to invoke the endpoint.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

31. 一位机器学习专家正在为一家在线零售商工作，该零售商希望对每个客户的访问进行分析，通过机器学习管道进行处理。数据需要以每秒100个交易的速度被亚马逊Kinesis数据流摄取，SON数据blob的大小为100KB。专家应该使用Kinesis数据流中的最小分片数来成功摄取这些数据？ A Machine Learning Specialist is working for an online retailer that wants to run analytics on every customer visit, processed through a machine learning pipeline. The data needs to be ingested by Amazon Kinesis Data Streams at up to 100 transactions per second, and the SON data blob is 100 KB in size. What in the MINIMUM number of shards in Kinesis Data Streams the Specialist should use to successfully ingest this data?
    - [ ] A. 1 shard
    - [ ] B. 10 shards
    - [ ] C. 100 shards
    - [ ] D. 1,000 shards
  
    <details>
       <summary>Answer</summary>

       答案B: [ref](https://docs.aws.amazon.com/zh_cn/streams/latest/dev/service-sizes-and-limits.html)

    </details>

32. 一家公司使用商店货架上陈列的物品顶部的相机图像来确定哪些物品被移走，哪些物品仍然存在。经过几个小时的数据标记，该公司总共有1，000张手工标记的图像，涵盖了10个不同的物品。训练结果很差。哪种机器学习方法能满足该公司的长期需求？ A company uses camera images of the tops of items displayed on store shelves to determine which items were removed and which ones remain. After several hours of data labeling, the company has a total of 1, 000 hand-labeled images covering 10 distinct items. The training results were poor. Which machine learning approach fulfills the company’s long-term needs?
    - [ ] A. 将图像转换成灰度并重新训练模型。 Convert the images to grayscale and retrain the model.
    - [ ] B. 将不同项目的数量从10个减少到2个，建立模型，并进行迭代。 Reduce the number of distinct items from 10 to 2, build the model, and iterate.
    - [ ] C. 给每个项目贴上不同颜色的标签，再次拍摄图像，并建立模型。 Attach different colored labels to each item, take the images again, and build the model.
    - [ ] D. 使用图像变体如反转和翻译来增加每个项目的训练数据，建立模型，并进行迭代。 Augment training data for each item using image variants like inversions and translations, build the model, and iterate.
  
    <details>
       <summary>Answer</summary>

       答案D。

    </details>

33. 一家公司希望将用户行为分类为欺诈行为或正常行为。根据内部研究，一位机器精简专家将根据两个特征建立一个二进制分类器，账户年龄用x表示，交易月份用y表示，类别分布在提供的图中说明。正面类用红色表示，负面类用黑色表示。哪个模型的准确率最高？ A company wants to classify user behavior as either fraudulent or normal. Based on internal research, a machine leaning specialist will build a binary classifier based on two features age of account, denoted by x, and transaction month denoted by y. The class distributions are illustrated in the provided figure. The positive class is portrayed in red, while the negative class is portrayed in black. Which model would have the HIGHEST accuracy?

    ![98](img/98.png)
    - [ ] A. 线性支持向量机（SVM） Linear support vector machine (SVM)
    - [ ] B. 决策树 Decision Tree
    - [ ] C. 带有径向基函数核的支持向量机 Support vector machine (SVM) with a radial basis function kernel
    - [ ] D. 具有Tanh激活函数的单感知器 Single perceptron with a Tanh activation function
  
    <details>
       <summary>Answer</summary>

       答案B。

    </details>

34. 一位机器学习专家正在对一个有1,000条记录和50个特征的数据集应用线性最小二乘回归模型。在训练之前，ML专家注意到有两个特征是完全线性依赖的。为什么这可能是线性最小二乘回归模型的一个问题？ A Machine Learning Specialist is applying a linear least squares regression model to a dataset with 1, 000 record and 50 features. Prior to train he ML Specialist notices at two features are perfectly linearly dependent. Why could this be an issue for the linear least square regression model?
    - [ ] A. 它可能导致反向传播算法在训练中失败。 It could cause the backpropagation algorithm to fail during training.
    - [ ] B. 它可能在优化过程中产生一个奇异矩阵，从而无法定义一个唯一的解决方案。 It could create a singular matrix during optimization, which fails to define a unique solution.
    - [ ] C. 它可能在优化过程中修改损失函数，导致它在训练中失败。 It could modify the loss function during optimization, causing it to fail during training.
    - [ ] D. 它可能在数据中引入非线性的依赖关系，这可能使模型的线性假设失效。 It could introduce non-linear dependencies within the data, which could invalidate the linear assumptions of the model.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

35. 一家信用卡公司想建立一个信用评分模型，以帮助预测一个新的信用卡申请人是否会拖欠信用卡付款。该公司收集了大量的数据，其中有数千条原始属性。早期的分类模型实验表明，许多属性是高度相关的，大量的特征使训练速度明显减慢，而且存在一些过拟合问题。这个项目的数据科学家希望在不损失原始数据集的大量信息的情况下加快模型的训练时间。该数据科学家应该使用哪种特征工程技术来达到目标？ A credit card company wants to build a credit scoring model to help predict whether a new credit card applicant will default on a credit card payment. The company has collected data from a large number of sources with thousands of raw attributes. Early experiments to rain a classification model revealed that many attributes are highly correlated, the large number of features slows down the training speed significantly, and that there are some overfitting issues. The Data Scientist on this project would like to speed up the model training time without losing a lot of information from the original dataset. Which feature engineering technique should the Data Scientist use to meet the objectives?
    - [ ] A. 在所有的特征上运行自相关，删除高度相关的特征。 Run self-correlation on all features and remove highly correlated features.
    - [ ] B. 将所有的数值归一化为0和1之间。 Normalize all numerical values to be between 0 and 1.
    - [ ] C. 使用自动编码器或主成分分析（PCA），用新的特征替换原来的特征。 Use an autoencoder or principal component analysis (PCA) to replace original features with new features.
    - [ ] D. 使用k-means对原始数据进行聚类，并使用每个聚类的样本数据来建立一个新的数据集。 Cluster raw data using k-means and use sample data from each cluster to build a new dataset.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

36. 一位机器学习专家将OT土壤传感器数据存储在Amazon DynamoDB表中，将天气事件数据作为JSON文件存储在Amazon S3中。DynamoDB中的数据集大小为10GB，Amazon S3中的数据集大小为5GB。专家想在这些数据上训练一个模型，以帮助预测土壤湿度，作为使用Amazon SageMaker的天气事件的一个函数。哪种解决方案可以完成必要的转换，以训练Amazon SageMaker模型，并且管理费用最少？ A Machine Learning Specialist stores OT soil sensor data in an Amazon DynamoDB table and stores weather event data as JSON files in Amazon S3. The dataset in DynamoDB is 10 GB in size and the dataset in Amazon S3 is 5 GB in size. The Specialist wants to train a model on this data to help predict soil moisture levels as a function of weather events using Amazon SageMaker. Which solution will accomplish the necessary transformation to train the Amazon SageMaker model with the LEAST amount of administrative overhead?
    - [ ] A. 启动一个Amazon EMR集群。为 DYNAMODB 表和 S3 数据创建一个 Apache Hive 外部表。连接Hive表，并将结果写到Amazon S3。 Launch an Amazon EMR cluster. Create an Apache Hive external table for the DYNAMODB table and S3 data. Join the Hive tables and write the results out to Amazon S3.
    - [ ] B. 使用AWS Glue爬虫来抓取数据。编写一个AWS Glue ETL作业，合并两个表并将输出写入Amazon Redshift集群。 Crawl the data using AWS Glue crawlers. Write an AWS Glue ETL job that merges the two tables and writes the output to an Amazon Redshift cluster.
    - [ ] C. 在传感器表上启用Amazon DynamoDB Streams。编写一个AWS Lambda函数，消耗流并将结果附加到Amazon S3中的现有天气文件。 Enable Amazon DynamoDB Streams on the sensor table. Write an AWS Lambda function that consumes the stream and appends the results to the existing weather files in Amazon S3.
    - [ ] D. 使用AWS Glue爬虫来抓取数据。编写一个AWS Glue ETL作业，将两个表合并，并将CSV格式的输出写到Amazon S3。 Crawl the data using AWS Glue crawlers. Write an AWS Glue ETL job that merges the two tables and writes the output in CSV format to Amazon S3.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

37. 一家飞机发动机制造公司正在测量时间序列中的200个性能指标。工程师们希望在测试过程中以近乎实时的方式检测出关键的制造缺陷。所有的数据都需要被存储起来，以便进行离线分析。哪种方法对进行近实时的缺陷检测最有效？ An aircraft engine manufacturing company is measuring 200 performance metrics in a time-series. Engineers want to detect critical manufacturing defects in near-real time during testing. All of the data needs to be stored for offline analysis. What approach would be the MOST effect to perform near-real time defect detection?
    - [ ] A. 使用AWS IoT Analytics进行摄取、存储和进一步分析。使用AWS IT分析中的Jupyter笔记本来进行异常情况的分析。 Use AWS IoT Analytics for ingestion, storage, and further analysis. Use Jupyter notebooks from within AWS IT Analytics to carry our analysis for anomalies.
    - [ ] B. 使用Amazon S3进行摄取存储和进一步分析。使用亚马逊EMR集群来进行Apache Spark ML k-means聚类，以确定异常情况。 Use Amazon S3 for ingestion storage, and further analysis. Use an Amazon EMR cluster to carry out Apache Spark ML k-means clustering to determine anomalies.
    - [ ] C. 使用亚马逊S3进行摄取、存储和进一步分析。使用Amazon SageMaker随机切割森林（RCF）算法来确定异常情况。 Use Amazon S3 for ingestion, storage, and further analysis. Use the Amazon SageMaker Random Cut Forest (RCF) algorithm to determine anomalies.
    - [ ] D. 使用Amazon Kinesis Data Firehose进行摄取，使用Amazon Kinesis Data Analytics Random Cut Forest（RCF）进行异常检测。使用Kinesis Data Firehose将数据存储在Amazon S3中，以便进一步分析。 Use Amazon Kinesis Data Firehose for ingestion and Amazon Kinesis Data Analytics Random Cut Forest (RCF) to perform anomaly detection. Use Kinesis Data Firehose to store data in Amazon S3 for further analysis.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

38. 一家科技创业公司正在使用复杂的深度神经网络和GPU计算，根据每个客户的习惯和互动，向其现有客户推荐公司的产品。该解决方案目前从亚马逊S3桶中提取每个数据集，然后将数据加载到从公司Git库中提取的TensorFlow模型中，在本地运行。然后，这项工作运行几个小时，同时不断地将其进度输出到同一个S3桶。工作可以暂停、重启，并在发生故障的情况下随时继续，并且从中央队列中运行。高级管理人员对解决方案资源管理的复杂性和定期重复该过程所涉及的成本感到担忧。他们要求将工作量自动化，以便从周一开始每周运行一次，并在周五营业结束前完成。应该使用哪种架构来以最低的成本扩展解决方案？ A technology startup is using complex deep neural networks and GPU compute to recommend the company’s products to its existing customers based upon each customers habits and interactions. The solution currently pulls each dataset from an Amazon S3 bucket before loading the data into a TensorFlow model pulled from the company’s Git repository that runs locally. This job then runs for several hours while continually outputting its progress to the same S3 bucket. The job can be paused, restarted, and continued at any time in the event of a failure, and is run from a central queue. Senior managers are concerned about the complexity of the solutions resource management and the costs involved in repeating the process regularly. They ask for workload to be automated so it runs once a week starting Monday and completing by the close of business Friday. Which architecture should be used to scale the solution at the lowest cost?
    - [ ] A. 使用AWS深度学习容器实现该解决方案，并在GPU兼容的Sport Instance上使用AWS Batch将容器作为作业运行。 Implement the solution using AWS Deep Learning Containers and run the container as a job using AWS Batch on a GPU-compatible Sport Instance.
    - [ ] B. 使用低成本的GPU兼容的亚马逊EC2实例来实现该解决方案，并使用AWS实例调度器来安排任务。 Implement the solution using a low-cost GPU-compatible Amazon EC2 instance and use the AWS Instance Scheduler to schedule the task.
    - [ ] C. 使用AWS深度学习容器实现解决方案，使用在Spot Instance上运行的AWS Fargate运行工作负载，然后使用内置的任务调度器安排任务。 Implement the solution using AWS Deep Learning Containers, run the workload using AWS Fargate running on Spot Instance, and then schedule the task using the built-in task scheduler.
    - [ ] D. 使用在Spot Instance上运行的Amazon ECS实现解决方案，并使用ECS服务调度器安排任务。 Implement the solution using Amazon ECS running on Spot Instances and schedule the task using the ECS service scheduler.

    <details>
       <summary>Answer</summary>

       答案C: [ref](https://aws.amazon.com/cn/blogs/aws/aws-fargate-spot-now-generally-available/)

    </details>

39. 一家金融公司正试图检测信用卡欺诈。该公司观察到，平均有2%的信用卡交易是欺诈性的。一位数据科学家在一年的信用卡交易数据上训练了一个分类器。该模型需要将欺诈性交易（阳性）从正常交易（阴性）中识别出来。该公司的目标是尽可能多地准确捕获阳性交易。数据科学家应该使用哪些指标来优化该模式？(选择两个) A financial company is trying to detect credit card fraud. The company observed that, on average, 2% of credit card transactions were fraudulent. A Data Scientist trained a classifier on a year’s worth of credit card transactions data. The model needs to identify the fraudulent transactions (positives) from the regular ones (negatives). The company’s goal is to accurately capture as many positives as possible. Which metrics should the Data Scientist use to optimize the mode? (Select TWO)
    - [ ] A. 特异性 Specificity
    - [ ] B. 假阳性率 False positive rate
    - [ ] C. 准确率 Accuracy
    - [ ] D. 精度-记忆曲线下的面积 Area Under the precision-recall curve
    - [ ] E. 真阳性率 True positive rate

    <details>
       <summary>Answer</summary>

       答案DE。

    </details>

40. 你的任务是将一个传统的应用程序从你的数据中心内运行的虚拟机转移到亚马逊VPC上。不幸的是，这个应用程序需要访问一些场所服务，而且配置这个应用程序的人都不在你的公司工作。更糟糕的是，没有相关的文档。怎样才能让在VPC内运行的应用程序在不被重新配置的情况下达到回访其内部依赖关系的目的？选择3个答案 You are tasked with moving a legacy application from a virtual machine running inside your data center to an Amazon VPC. Unfortunately, this app requires access to a number of premises services and no one who configured the app still works for your company. Even worse, there is no documentation for it. What will allow the application running inside the VPC to reach back and access its internal dependencies without being reconfigured? Choose 3 answers
    - [ ] A. 当前虚拟机的一个虚拟机导入。 A VM Import of the current virtual machine.
    - [ ] B. 一个互联网网关，允许VPN连接。 An Internet Gateway to allow a VPN connection.
    - [ ] C. 亚马逊Route53中的条目，允许Instance解决其依赖的IP地址。 Entries in Amazon Route53 that allow the Instance to resolve its dependencies IP addresses.
    - [ ] D. 一个与企业内部不冲突的IP地址空间。 An IP address space that does not conflict with the one on-premises.
    - [ ] E. VPC实例上的一个Elastic IP地址。 An Elastic IP address on the VPC instance.
    - [ ] F. 在VPC和容纳内部服务的网络之间有一个AWS直接连接链接。 An AWS Direct Connect link between the VPC and the network housing the internal services.

    <details>
       <summary>Answer</summary>

       答案ADF。

    </details>

41. 一家制造商在其整个工厂部署了一个由50,000个传感器组成的阵列，以预测部件的故障。数据科学家在Gluon中建立了一个长短期记忆（STM）模型，并正在使用Amazon SageMaker API对其进行训练。数据科学家们正在使用一个有一千万个例子的时间序列来训练这个模型。训练目前需要100个小时，数据科学家正试图通过使用多个GPUS来加快训练速度。然而，当他们修改代码以使用8个GPU时，它的运行速度比在一个GPU上略低。目前的超参数设置是。1）炒作参数值 批量大小182，2）剪辑梯度10，3）自回归窗口160，4）学习率0.01，5）间隔时间80。为了加快8个GPUS的训练速度，同时保持测试的准确性，建议将以下哪些变化放在一起？(选择两个) A manufacturer has deployed an array of 50,000 sensors throughout its plant to predict failures in components. Data Scientists have built a long short-term memory (STM) model in Gluon and are training it using the Amazon SageMaker API. The Data Scientists are training the model using a time series with ten million examples. Training is currently taking 100 ours and Data Scientists are attempting to speed it up by using multiple GPUS. However, when they modified the code to use eight GPUs, it is running slightly lower than on one GPU. The current hyperparameter settings are: 1) Hype Parameter Value Batch size 182, 2) Clip gradient 10, 3) Autoregressive window 160, 4) Learning rate 0.01, 5) Epochs 80. Which of the following changes together are recommended to speed up training on 8 GPUS while maintaining test accuracy? (Select TWO)
    - [ ] A. 将批量大小增加8倍。 Increase the batch size by a factor of 8.
    - [ ] B. 将剪辑梯度增加8。 Increase the clip gradient by 8.
    - [ ] C. 将自回归窗口增加8倍。 Increase the autoregressive window by a factor of 8.
    - [ ] D. 将学习率提高8倍。 Increase the learning rate by a factor of 8.
    - [ ] E. 将epochs的数量减少20。 Decrease the number of epochs by 20.

    <details>
       <summary>Answer</summary>

       答案AD。

    </details>

42. 一位数据科学家需要创建一个用于欺诈检测的模型。该数据集由2年的记录交易组成，每个交易都有一组小的特征。数据集中的所有交易都是人工标注的。由于欺诈并不经常发生，该数据集是高度不平衡的。不到2%的数据集被标记为欺诈性的。哪种解决方案为欺诈活动的分类提供了最佳预测能力？ A Data Scientist needs to create a model for fraud detection. The dataset is composed of 2 years’ worth of logged transactions, each with a small set of features. All of the transactions in the dataset were manually labeled. Since fraud does not occur frequently, the dataset is highly imbalanced. Less than 2% of the dataset was labeled as fraudulent. Which solution provides the optimal predictive power for classifying fraudulent activity?
    - [ ] A. 使用聚类技术对数据集进行过度取样，将准确性作为目标指标，并应用随机切割森林（RCP）。 Oversample the dataset using a clustering technique, use accuracy as the objective metric, and apply Random Cut Forest (RCP).
    - [ ] B. 使用聚类技术对数据集中的多数类进行未取样，使用精确度作为目标指标，并应用随机切割森林（RCF）。 Undersample the majority class in the dataset using a clustering technique, use precision as the objective metric, and apply Random Cut Forest (RCF).
    - [ ] C. 重新取样数据集，使用F1分数作为目标指标，并应用XGBboost。 Resample the dataset, use the F1 score as the objective metric, and apply XGBboost.
    - [ ] D. 对数据集进行重新取样，使用精度作为目标指标，并应用XGBboost。 Resample the dataset, use accuracy as the objective metric, and apply XGBboost.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

43. 一位机器学习专家正在使用监督学习算法训练一个模型。专家对数据集进行了分割，使用80%的数据进行训练，保留20%的数据进行测试。在评估该模型时，专家发现该模型对训练数据集的准确率为97%，对测试数据集的准确率为75%。造成这种差异的原因是什么，专家应该采取什么措施？ A Machine Learning Specialist is training a model using a supervised learning algorithm. The Specialist split the dataset to use 80% of the data for training and reserved 20% of the data for testing. While evaluating the model the Specialist discovers that the mode is 97% accurate for the training dataset and 75% accurate for the test dataset. What is the reason for the discrepancy and what action should the Specialist take?
    - [ ] A. 大量训练数据的高准确率意味着该模型已经完成。将该模型部署到生产中。 The high accuracy for the larger amount of training data means that the model is finished. Deploy the model to production.
    - [ ] B. 该模型目前对训练数据过度拟合，在它以前没有见过的数据上没有表现出应有的水平。改变超参数以简化和泛化模型，然后重新训练。 The mode is currently overfitting the training data and not performing as well as it should on data it has not seen before. Change the hyperparameters to simplify and generalize the model, then retrain.
    - [ ] C. 需要测试集中的额外数据来平衡模式的得分 在训练集和测试集中更均匀地重新分配数据。 Additional data in the test set is needed to balance the scoring of the model Redistribute the data more evenly across training and test sets.
    - [ ] D. 该模式目前是欠拟合的，没有足够的复杂性来捕捉数据集的全部范围。改变超参数，使模式更加具体和复杂，然后重新训练。 The mode is currently underfitting and does not have enough complexity to capture the full scope of the dataset. Change the hyperparameters to make the model more specific and complex, then retrain.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

44. 一位机器学习专家正在开发一个回归模型来预测一场即将到来的音乐会的门票销售。历史门票销售数据由1,000多条记录组成，包含20个数字变量。在探索性数据分析阶段，专家发现有33条记录的数值在框图上四分位数的最右边有一个可耕地的数值。专家与一位业务用户确认，这些数值是不寻常的，但也是合理的。还有70条记录中另一个数字变量是空白的。专家应该做什么来纠正这些问题？ A Machine Learning Specialist is developing a regression model to predict ticket sales for an upcoming concert. The historical ticket sales data consists of more than 1, 000 records containing 20 numerical variables During the exploratory data analysis phase, the Specialist discovered 33 records have values for a numerical arable in the far right of the box plots upper quartile. The Specialist confirmed with a business user that those values are unusual, but plausible. There are also 70 records where another numerical variable is blank. What should the Specialist do to correct these problems?
    - [ ] A. 删除不寻常的记录，用平均值代替空白值。 Drop the unusual records and replace the blank values with the mean value.
    - [ ] B. 将不寻常的数据规范化，为空白值创建一个单独的布尔变量。 Normalize unusual data and create a separate Boolean variable for blank values.
    - [ ] C. 删除不寻常的记录，将空白值填为0。 Drop the unusual records and fill in the blank values with 0.
    - [ ] D. 使用异常数据，并为空白值创建一个单独的布尔变量。 Use unusual data and create a separate Boolean variable for blank values.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

45. 一位机器学习专家正在开发一种模式，将制造过程中的缺陷部件归类为八种缺陷类型之一。训练数据包括每个缺陷类型的100,000张图像。在图像分类模型的初始训练中，专家注意到验证准确率为89. 5%，而训练准确率为90%。众所周知，这种类型的图像分类的人类水平性能约为97%。专家应该考虑如何提高该模型的性能？(选择两个。） A Machine Learning Specialist is developing a mode that classifies defective parts from a manufacturing process into one of eight defect types. The training data consists of 100,000 images per defect type. During the initial training of the image classification model, the Specialist notices that the validation accuracy is 89. 5% while the training accuracy is 90%. It is known that human-level performance for this type of image classification is around 97%. What should the Specialist consider improving the performance of the model? (Select TWO.)
    - [ ] A. 延长训练时间 A longer training time
    - [ ] B. 扩充数据 Data augmentation
    - [ ] C. 获得更多的训练数据 Getting more training data
    - [ ] D. 一个不同的优化器 A different optimizer
    - [ ] E. 二级正则化 L2 regularization

    <details>
       <summary>Answer</summary>

       答案BC。

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
