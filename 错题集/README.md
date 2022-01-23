# AWS MLS 131-228

1. 一家游戏公司推出了一款网络游戏，人们可以免费开始玩，但如果他们选择使用某些功能就需要付费。该公司需要建立一个自动系统来预测一个新用户是否会在一年内成为付费用户。该公司已经收集了一个来自一百万用户的标记数据集。训练数据集包括1000个正样本（来自一年内最终付费的用户）和99.9万个负样本（来自没有使用任何付费功能的用户）。每个数据样本由200个特征组成，包括用户年龄、设备、位置和游戏模式。使用这个数据集进行训练，数据科学团队训练了一个随机森林模型，在训练集上收敛了99%以上的准确性。然而，测试数据集上的预测结果并不令人满意。数据科学团队应该采取以下哪种方法来缓解这个问题？(选择两个) A gaming company has launched an online game where people can start playing for free, but they need to pay if they choose to use certain features. The company needs to build an automated system to predict whether a new user will become a paid user within one year. The company has gathered a labeled dataset from one million users. The training dataset consists of 1,000 positive sample (from users who ended up paying within one year) and 999,000 negative samples (from users who did not use any paid features). Each data sample consists of 200 features including user age, device, location, and play patterns. Using this dataset for training, the Data Science team trained a random forest model that converged with over 99% accuracy on the training set. However, the prediction results on a test dataset were not satisfactory. Which of the following approaches should the Data Science team take to mitigate this issue? (Choose two)
   - [ ] A. 在随机森林中加入更多的深度树，使模型能够学习更多的特征。 Add more deep trees to the random forest to enable the model to learn more features.
   - [ ] B. 在训练数据集中包括一份测试数据集中的样本。 Include a copy of the samples in the test dataset in the training dataset.
   - [ ] C. 通过复制阳性样本并在复制的数据中加入少量的噪声，产生更多的阳性样本。 Generate more positive samples by duplicating the positive samples and adding a small amount of noise to the duplicated data.
   - [ ] D. 改变成本函数，使假阴性对成本值的影响高于假阳性。 Change the cost function so that false negatives have a higher impact on the cost value than false positives.
   - [ ] E. 改变成本函数，使假阳性对成本值的影响高于假阴性。 Change the cost function so that false, positives have a higher impact on the cost value than false negatives.

    <details>
      <summary>Answer</summary>

      答案CD：
      - C: 因为我们需要一个平衡的数据集。
      - D: 阳性样本的数量很大，所以模型倾向于对所有情况预测为0（阴性），导致假阴性问题。我们应该尽量减少这种情况。

   </details>

2. 一位机器学习专家部署了一个模型，在一家公司的网站上提供产品推荐。起初，该模型表现非常好，导致客户平均购买更多产品。然而，在过去的几个月里，该专家注意到产品推荐的效果已经减弱，客户开始回到他们原来的习惯，减少消费。专家不确定发生了什么，因为该模型与一年多前的最初部署相比没有变化。专家应该尝试哪种方法来提高模型的性能？ A Machine Learning Specialist deployed a model that provides product recommendations on a company's website. Initially, the model was performing very well and resulted in customers buying more products on average. However, within the past few months the Specialist has noticed that the effect of product recommendations has diminished, and customers are starting to return to their original habits of spending less. The Specialist is unsure of what happened, as the model has not changed from its initial deployment over a year ago. Which method should the Specialist try to improve model performance?
   - [ ] A. 该模型需要完全重新设计，因为它无法处理产品库存变化。 The model needs to be completely re-engineered because it is unable to handle product inventory changes.
   - [ ] B. 应该对模型的超参数进行可预测的更新以防止漂移。 The model's hyperparameters should be predicably updated to prevent drift.
   - [ ] C. 该模型应定期使用原始数据从头开始训练，同时增加一个正则化项来处理产品库存变化。 The model should be periodically retrained from scratch using the original data while adding a regularization term to handle product inventory changes.
   - [ ] D. 该模型应定期使用原始训练数据和产品库存变化时的新数据进行重新训练。 The model should be periodically retrained using the original training data plus new data as product inventory changes.

   <details>
     <summary>Answer</summary>

     答案D。

   </details>

3. 一位机器学习工程师正在准备一个数据框架，用于使用Amazon SageMaker近似学习者算法的监督学习请求。该ML工程师注意到目标标签类是高度不平衡的，并且多个特征列包含缺失值。在整个数据框架中，缺失值的比例小于5%。ML工程师应该怎么做才能将缺失值造成的偏差降到最低？ A Machine Learning Engineer is preparing a data frame for a supervised learning ask with the Amazon SageMaker near Learner algorithm. The ML Engineer notices the target label classes are highly imbalanced and multiple feature columns contain missing values. The proportion of missing values across the entire data frame is less than 5%. What should the ML Engineer do to minimize bias due to missing values?
   - [ ] A. 用同一行中非缺失值的平均数或中位数来替换每个缺失值。 Replace each missing value by the mean or median across non-missing values in same row.
   - [ ] B. 删除包含缺失值的观察值，因为这些观察值只占数据的50％以下。 Delete observations that contain missing values because these represent less than 50 of the data.
   - [ ] C. 用同一列中非缺失值的平均数或中位数替换每个缺失值。 Replace each missing value by the mean or median across non-missing values in the same column.
   - [ ] D. 对于每个特征，使用基于其他特征的监督学习来近似计算缺失值。 For each feature, approximate the missing values using supervised learning based on other features.

   <details>
     <summary>Answer</summary>

     答案D：使用监督学习来预测基于其他特征值的缺失值。不同的监督学习方法可能有不同的表现，但任何正确实施的监督学习方法都应该提供与平均数或中位数近似相同或更好的近似，如回答A和C中提出的监督学习应用于缺失值的推算是一个活跃的研究领域。

   </details>

4. 一位机器学习专家正在为一项计算机视觉任务在亚马逊EC2 P3实例集群上训练卷积神经网络（CNN）。然而，训练的时间比预期的长。该专家试图降低图像的分辨率以加快训练速度，但准确率下降到可接受的水平以下。哪种技术可以在不明显降低准确性的情况下加快训练速度？(选择两个 A machine learning specialist is raining a convolutional neural network (CNN) on a cluster of Amazon EC2 P3 instances for a computer vision task. However, training is taking longer than expected. The specialist tried reducing the resolution of images to speed up training, but the accuracy dropped below an acceptable level. Which technique may speed up training without significantly degrading accuracy? (Select Two)
   - [ ] A. 使用混合精度。 Use mixed precision.
   - [ ] B. 用全连接层取代卷积层。 Replace convolutional layers with fully connected layers.
   - [ ] C. 从预训练的CNN开始，在训练期间冻结大多数层。 Start with a pretrained CNN and freeze most layers during training.
   - [ ] D. 用G3实例替换P3实例。 Replace the P3 instances with G3 instances.
   - [ ] E. 对每一类的训练数据进行降样，同时保持类的比例平衡。 Downsample training data for each class while keeping the class ratios balanced.

   <details>
     <summary>Answer</summary>

     答案AC。

   </details>

5. 一位数据科学家正在开发一个机器学习模型，以对一项金融交易是否是欺诈性的进行分类。可供训练的标记数据包括100,000个非欺诈性的观察值和1,000个欺诈性的观察值。数据科学家将XGBboost算法应用于数据，当训练好的模型应用于之前未见过的验证数据集时，会产生以下混淆矩阵。该模型的准确率为99.1%，但数据科学家被要求减少假阴性的数量。数据科学家应该采取哪种步骤组合来减少该模型的假阳性预测的数量？(选择两个) A Data Scientist is developing a machine learning model to classify whether a financial transaction is fraudulent. The labeled data available for raining consists of 100,000 non-fraudulent observations and 1.000 fraudulent observations. The Data Scientist applies the XGBboost algorithm o the data, resulting in the following confusion matrix when the trained model is applied to a previously unseen validation dataset. The accuracy of the model is 99.1%, but the Data Scientist has been asked to reduce the number of false negatives. Which combination of steps should the Data Scientist take to reduce the number of false positive predictions by the model? (Choose two)

   ![75](../第2套/img/75.png)

   - [ ] A. 改变XGBboost`eval_metric`参数，根据RMSE而不是误差进行优化。 Change the XGBboost `eval_metric` parameter to optimize based on RMSE instead of error.
   - [ ] B. 增加XGBboost `scale_pos_weight`参数来调整正负权重的平衡。 Increase the XGBboost `scale_pos_weight` parameter to adjust the balance of positive and negative weights.
   - [ ] C. 增加XGBboost `max_depth`参数，因为模型目前对数据的拟合不足。 Increase the XGBboost `max_depth` parameter because the model is currently underfitting the data.
   - [ ] D. 改变XGBOOST `eval_metric`参数，根据AUC而不是误差来优化。 Change the XGBOOST `eval_metric` parameter to optimize based on AUC instead of error.
   - [ ] E. 减少XGBOOST `max_depth`参数，因为该模型目前过度拟合数据。 Decrease the XGBOOST `max_depth` parameter because the model is currently overfitting the data.

   <details>
      <summary>Answer</summary>

      答案BD。

   </details>

6. 数据科学家正在一个有多个类别的数据集上训练一个多层感知器（MLP）。与数据集中的其他类相比，感兴趣的目标类是独特的，但它没有达到可接受的召回指标。数据科学家已经尝试改变MPS隐藏层的数量和大小，但结果没有明显改善。必须尽快实施改善召回率的解决方案。应该使用哪些技术来满足这些要求？ Data Scientist is training a multilayer perceptron (MLP) on a dataset with multiple classes. The target class of interest is unique compared to the other classes within the dataset, but it does not achieve an acceptable recall metric. The Data Scientist has already tried varying the number and size of the MPS hidden layers, which has not significantly improved the results. A solution to improve recall must be implemented as quickly as possible. Which techniques should be used to meet these requirements?
   - [ ] A. 使用Amazon Mechanical Turk收集更多数据，然后重新训练。 Gather more data using Amazon Mechanical Turk and then retrain.
   - [ ] B. 训练一个异常检测模型而不是MLP。 Train an anomaly detection model instead of an MLP.
   - [ ] C. 训练一个XGBboost模型而不是MLP。 Train an XGBboost model instead of an MLP.
   - [ ] D. 在MLP的损失函数中加入类的权重，然后再重新训练。 Add class weights to the MLP's loss function and the retrain.

   <details>
     <summary>Answer</summary>

     答案D。

   </details>

7. 一位机器学习专家正在研究一个线性回归模型，并注意到该模型正在过度拟合。该专家应用L1正则化参数，并再次运行模型。不幸的是，这导致所有特征的权重为零。机器学习专家应该怎么做来改善模型的结果？ A Machine Learning Specialist is working on a linear regression model and notices the model is overfitting. The specialist applies an L1 regularization parameter and runs the model again. Unfortunately, this results in all features having zero weights. What should the Machine Learning Specialist do to improve the model results?
   - [ ] A. 增加L1正则化参数。 Increase the L1 regularization parameter.
   - [ ] B. 减少L1正则化参数。 Decrease the L1 regularization parameter.
   - [ ] C. 增加L2正则化参数。 Increase the L2 regularization parameter.
   - [ ] D. 减少L2正则化参数。 Decrease the L2 regularization parameter.

   <details>
     <summary>Answer</summary>

     答案B。

   </details>

8. 在一个高度敏感的10TB数据集上用Amazon SageMaker训练的模型，即使经过几次训练，预测结果也不佳。在进一步审查数据集时，一位机器学习专家注意到，一些字段是从同一数据源中提取的，包含重叠的信息。该专家应该如何清理数据，并为未来的训练工作安全地访问它？ A model trained with Amazon SageMaker on a highly sensitive 10 TB dataset is delivering poor prediction results, even after several training jobs. Upon further review of the dataset a machine learning specialist notices that some fields were pulled from the same data source and contain overlapping information. How should the specialist clean up the data and securely access it for future training jobs?
   - [ ] A. 使用AWS Glue来丢弃相关字段，并将新的数据集存储在Amazon S3上。为Amazon SageMaker添加一个S3 VPC端点，用于数据访问。 Use AWS Glue to discard the related fields and store the new dataset on Amazon S3. Add an S3 VPC endpoint for Amazon SageMaker for data access.
   - [ ] B. 使用Amazon EMR来丢弃相关字段，并将新的数据集存储在Apache Hadoop分布式文件系统（HDFS）上。在EMR VPC中为Amazon SageMaker添加一个私有子网，用于数据访问。 Use Amazon EMR to discard the elated elds and store the new dataset on an Apache Hadoop Distributed File system (HDFS). Add a private subnet in an EMR VPC for Amazon SageMaker for data access.
   - [ ] C. 使用AWS Lambda来连接相关字段，并将新的数据集存储在Amazon SageMaker上。使用SageMaker EBS卷进行数据访问。 Use AWS Lambda to join the related fields and store the new dataset on Amazon SageMaker. Use SageMaker EBS volumes for data access.
   - [ ] D. 使用Amazon Athena来连接相关字段，并将新的数据集存储在Amazon S3上。为Amazon SageMaker添加一个NAT网关用于数据访问。 Use Amazon Athena to join the related fields and store the new dataset on Amazon S3. Add a NAT gateway for Amazon SageMaker for data access.

   <details>
     <summary>Answer</summary>

     答案D。

   </details>

9. 一个企业使用长短期记忆（LSTM）模型评估与特定能源部门相关的风险变量。该程序分析了多页文本文件，并将每个短语归类为构成危险或不构成风险。尽管数据科学家对几种网络结构进行了广泛的实验，并对相关的超参数进行了调整，但该模型的表现仍然不佳。哪种模型和目标属性组合最适合这个问题？ A business evaluates the risk variables associated with a specific energy sector using a long short-term memory (LSTM) model. The program analyzes multi-page text documents and categorizes each phrase as either posing a danger or posing no risk. The model is underperforming, despite the Data Scientist's extensive experimentation with several network architectures and tuning of the associated hyperparameters. Which model and target attribute combination is most suitable for this problem?
   - [ ] A. 通过术语频率-反文档频率（TF-IDF）向量来初始化单词，该向量在与能源部门相关的大量新闻文章上进行了预训练。 Initialize the words by term frequency-inverse document frequency (TF-IDF) vectors pretrained on a large collection of news articles related to the energy sector.
   - [ ] B. 使用门控递归单元（GRUS）而不是LSTM，并运行训练过程直到验证损失停止下降。 Use gated recurrent units (GRUS) instead of LSTM and run the training process until the validation loss stops decreasing.
   - [ ] C. 降低学习率，运行训练过程，直到训练损失停止下降。 Reduce the learning rate and run the training process until the training loss stops decreasing.
   - [ ] D. 通过在与能源部门相关的大量新闻文章上预训练的word2vec嵌入来初始化单词。 Initialize the words by word2vec embeddings pretrained on a large collection of news articles related to the energy sector.

   <details>
     <summary>Answer</summary>

     答案D。

   </details>

10. 一位数据科学家收到了一组保险记录，每条记录包括一个记录ID，200个类别中的最终结果，以及最终结果的日期。还提供了一些关于索赔内容的部分信息，但只针对200个类别中的几个。对于每个结果类别，有数百条记录分布在过去三年中。数据科学家希望提前几个月预测每个类别的索赔数量。提前几个月。应该使用哪种类型的机器学习模型？ A Data Scientist received a set of insurance records, each consisting of a record ID, the final outcome among 200 categories, and the date of the final outcome. Some partial information on claim contents is also provided, but only for a few of the 200 categories. For each outcome category, there are hundreds of records distributed over the past three years. The Data Scientist wants to predict how many claims to expect in each category from month to month a few months in advance. What type of machine learning model should be used?
    - [ ] A. 使用基于索赔内容的200个类别的监督学习，按月分类。 Classification month-to-month using supervised learning of the 200 categories based on claim contents.
    - [ ] B. 使用索赔ID和时间戳的强化学习，代理将识别每个类别中的索赔数量，以期望从月到月。 Reinforcement learning using claim IDs and timestamps where the agent will identify how many claims in each category to expect from month to month.
    - [ ] C. 使用索赔ID和时间戳进行预测，以确定每个类别中每个月会有多少索赔。 Forecasting using claim IDs and timestamps to identify how many claims in each category to expect from month to month.
    - [ ] D. 通过监督学习对提供部分索赔内容信息的类别进行分类，并使用索赔D和时间戳对所有其他类别进行预测。 Classification with supervised learning of the categories for which partial information on claim contents is provided, and forecasting using claim Ds and timestamps for all other categories.
  
    <details>
       <summary>Answer</summary>

       答案D。

    </details>

11. 一家电信公司需要预测客户的流失（即决定转投竞争对手的客户）。该公司有每个客户的历史记录，包括每月的消费模式，给客户服务的电话，以及客户是否最终退出服务。所有这些数据都存储在Amazon S3中。该公司需要知道哪些客户可能很快就会流失，以便他们能够赢回客户的忠诚度。满足这些要求的最佳方法是什么？ A telecommunications company needs to predict customer churn (i.e., customers who decide to switch to a competitor). The company has historic records of each customer, including monthly consumption patterns, calls to customer service, and whether the customer ultimately quit the service. All of this data is stored in Amazon S3. The company needs to know which customers are likely going to churn soon so that they can win back their loyalty. What is the optimal approach to meet these requirements?
    - [ ] A. 使用亚马逊机器学习服务，根据存储在Amazon S3的数据集建立二进制分类模型。该模型将被定期用于预测现有客户的流失属性。 Use the Amazon Machine Learning service to build the binary classification model based on the dataset stored in Amazon S3. The model will be used regularly to predict churn attribute for existing customers.
    - [ ] B. 使用AWS QuickSight将其连接到存储在Amazon S3中的数据，以获得必要的业务洞察力。绘制流失趋势图来推断现有客户的流失可能性。 Use AWS QuickSight to connect it to data stored in Amazon S3 to obtain the necessary business insight. Plot the churn trend graph to extrapolate churn likelihood for existing customers.
    - [ ] C. 使用EMR来运行Hive查询，以建立一个流失客户的档案。对现有的客户应用档案，以确定客户流失的可能性。 Use EMR to run the Hive queries to build a profile of a churning customer. Apply a profile to existing customers to determine the likelihood of churn.
    - [ ] D. 使用Redshift集群从Amazon S3复制数据。在Redshift中创建一个用户定义的函数来计算客户流失的可能性。 Use a Redshift cluster to COPY the data from Amazon S3. Create a User Defined Function in Redshift that computes the likelihood of churn.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>
