# AWS MLS 1-100

1. 一家互动式在线词典希望添加一个小部件，显示在类似语境中使用的单词。一位机器学习专家被要求为支持小工具的下游近邻模型提供单词特征。该专家应该怎样做才能满足这些要求？ An interactive online dictionary wants to add a widget that displays words used in similar contexts. A Machine Learning Specialist is asked to provide word features for the downstream nearest neighbor model powering the widget. What should the Specialist do to meet these requirements?
   - [ ] A. 创建单热词编码向量。Create one-hot word encoding vectors.
   - [ ] B. 使用Amazon Mechanical Turk为每个词制作一组同义词。 Produce a set of synonyms for every word using Amazon Mechanical Turk.
   - [ ] C. 创建单词嵌入向量，存储与每个其他单词的编辑距离。 Create word embedding vectors that store edit distance with every other word.
   - [ ] D. 下载在大型语料库中预先训练的词嵌入。 Download word embeddings pre-trained on a large corpus.

   <details>
      <summary>Answer</summary>

      答案D：因为不是专业的词典，需要去下载专业的语料库去训练模型。

   </details>

2. 一家公司正在使用Amazon Polly将纯文本文件翻译成语音，用于自动发布公司公告。然而，在目前的文件中，公司的首字母缩写被误读了。机器学习专家应该如何为未来的文件解决这个问题？ A company is using Amazon Polly to translate plaintext documents to speech for automated company announcements. However, company acronyms are being mispronounced in the current documents. How should a Machine Learning Specialist address this issue for future documents?
   - [ ] A. 将目前的文件转换成带有发音标签的SSML。 Convert current documents to SSML with pronunciation tags.
   - [ ] B. 创建一个合适的发音词库。 Create an appropriate pronunciation lexicon.
   - [ ] C. 输出语音标记来指导发音。 Output speech marks to guide in pronunciation.
   - [ ] D. 使用Amazon Lex来预处理文本文件的发音。Use Amazon Lex to preprocess the text files for pronunciation.

   <details>
      <summary>Answer</summary>

      答案B：[ref](https://docs.aws.amazon.com/zh_cn/polly/latest/dg/managing-lexicons.html)

   </details>

3. 一家保险公司正在开发一种用于车辆的新设备，该设备使用一个摄像头来观察司机的行为，并在他们出现分心时发出警报。该公司在一个受控环境中创建了大约10,000张训练图像，机器学习专家将用这些图像来训练和评估机器学习模型。在模型评估过程中，该专家注意到，随着历时数的增加，训练错误率降低得更快，而且该模型在未见过的测试图像上不能准确推断出来。以下哪种方法应该用来解决这个问题？（选择两项） An insurance company is developing a new device for vehicles that uses a camera to observe drivers’ behavior and alert them when they appear distracted. The company created approximately 10,000 training images in a controlled environment that a Machine Learning Specialist will use to train and evaluate machine learning models. During the model evaluation, the Specialist notices that the training error rate diminishes faster as the number of epochs increases, and the model is not accurately inferring on the unseen test images. Which of the following should be used to resolve this issue? (Choose TWO)
   - [ ] A. 在模型中加入消失的梯度。 Add vanishing gradient to the model.
   - [ ] B. 对训练数据进行数据增强。 Perform data augmentation on the training data.
   - [ ] C. 使神经网络结构复杂化。 Make the neural network architecture complex.
   - [ ] D. 在模型中使用梯度检查。 Use gradient checking in the model.
   - [ ] E. 在模型中加入L2正则化。 Add L2 regularization to the model.

   <details>
      <summary>Answer</summary>

      答案BE：该模型一定是过拟合了。正则化有助于解决机器学习中的过拟合问题以及数据增量。

   </details>

4. 当使用内置算法提交Amazon SageMaker训练作业时，必须指定哪些常用参数？(选择三个)。 When submitting Amazon SageMaker training jobs using one of the built-in algorithms, which common parameters MUST be specified? (Choose three.)
   - [ ] A. 训练通道，确定训练数据在Amazon S3桶中的位置。 The training channel, identifying the location of training data on an Amazon S3 bucket.
   - [ ] B. 验证通道，确定验证数据在Amazon S3 bucket上的位置。 The validation channel identifying the location of validation data on an Amazon S3 bucket.
   - [ ] C. Amazon SageMaker可以承担的IAM角色，代表用户执行任务。 Amazon SageMaker可以承担的IAM角色，代表用户执行任务。 The IAM role that Amazon SageMaker can assume to perform tasks on behalf of the users.
   - [ ] D. 在JSON数组中的超参数，如所使用的算法的记录。 Hyperparameters in a JSON array as documented for the algorithm used.
   - [ ] E. 亚马逊EC2实例类，指定训练将使用CPU或GPU运行。 The Amazon EC2 instance class specifying whether training will be run using CPU or GPU.
   - [ ] F. 输出路径，指定训练后的模型在Amazon S3桶中的位置。 The output path specifying where on an Amazon S3 bucket the trained model wilt persist.

   <details>
      <summary>Answer</summary>

      答案CEF：[ref](https://docs.aws.amazon.com/zh_cn/sagemaker/latest/APIReference/API_CreateTrainingJob.html)
   </details>

5. 一个监控服务每分钟产生1TB的规模指标记录数据。一个研究团队使用Amazon Athena对这些数据进行查询。由于大量的数据，查询运行缓慢，该团队需要更好的性能。这些记录应该如何存储在Amazon S3中以提高查询性能？ A monitoring service generates 1 TB of scale metrics record data ever minute. A Research team performs queries on this data using Amazon Athena. The Queries run slowly due to the large volume of data, and the team requires better performance. How should the records be stored in Amazon S3 to improve query performance?
   - [ ] A. CSV files
   - [ ] B. Parquet file
   - [ ] C. Compressed JSON
   - [ ] D. RecordIO

   <details>
      <summary>Answer</summary>

      答案B。

   </details>

6. 机器学习专家正在与一家媒体公司合作，对该公司网站上的热门文章进行分类。该公司正在使用随机森林来对一篇文章在发表前的受欢迎程度进行分类。下表是正在使用的数据样本。考虑到这个数据集，专家想把"Day_Of_Week"列转换为二进制值。应该使用什么技术将这一列转换为二进制值？Machine Learning Specialist is working with a media company to perform classification on popular article from the company's website. The company is using random forests to classify how popular an article will be before it is published. A sample of the data being used is in the following table. Given the dataset, the Specialist wants to convert the Day_Of_Week column to binary values. What technique should be used to convert this column to binary values?

   |Article_Title|Author|Top_Keywords|Day_Of_Week|URL_of_Article|Page_Views|
   |-------------|------|------------|-----------|--------------|----------|
   |Building a Big Data Platform|Jane Doe|Big Data, Spark, Hadoop|Tuesday|<http://examplecorp.com/data_platform.html>|1300456|
   |Getting Started with Deep Learning|Jane Doe|Deep Learning, Machine Learning, Spark|Tuesday|<http://examplecorp.com/started_deep_learning.html>|1230661|
   |MXNet ML Guide|Jane Doe|Machine Learning, MXNet, Logistic Regression|Thursday|<http://examplecorp.com/mxnet_guide.html>|937291|
   |Intro NoSQL Databases|Mary Major|NoSQL, Operations, Database|Monday|<http://examplecorp.com/nosql_intro_guide.html>|407821|

   - [ ] A. Binarization
   - [ ] B. One-hot encoding
   - [ ] C. Tokenization
   - [ ] D. Normalization transformation

    <details>
      <summary>Answer</summary>

      答案B。

   </details>

7. 一家游戏公司推出了一款网络游戏，人们可以免费开始玩，但如果他们选择使用某些功能就需要付费。该公司需要建立一个自动系统来预测一个新用户是否会在一年内成为付费用户。该公司已经收集了一个来自一百万用户的标记数据集。训练数据集包括1000个正样本（来自一年内最终付费的用户）和99.9万个负样本（来自没有使用任何付费功能的用户）。每个数据样本由200个特征组成，包括用户年龄、设备、位置和游戏模式。使用这个数据集进行训练，数据科学团队训练了一个随机森林模型，在训练集上收敛了99%以上的准确性。然而，测试数据集上的预测结果并不令人满意。数据科学团队应该采取以下哪种方法来缓解这个问题？(选择两个) A gaming company has launched an online game where people can start playing for free, but they need to pay if they choose to use certain features. The company needs to build an automated system to predict whether a new user will become a paid user within one year. The company has gathered a labeled dataset from one million users. The training dataset consists of 1,000 positive sample (from users who ended up paying within one year) and 999,000 negative samples (from users who did not use any paid features). Each data sample consists of 200 features including user age, device, location, and play patterns. Using this dataset for training, the Data Science team trained a random forest model that converged with over 99% accuracy on the training set. However, the prediction results on a test dataset were not satisfactory. Which of the following approaches should the Data Science team take to mitigate this issue? (Choose two)
   - [ ] A. 在随机森林中加入更多的深度树，使模型能够学习更多的特征。 Add more deep trees to the random forest to enable the model to learn more features.
   - [ ] B. 在训练数据集中包括一份测试数据集中的样本。 Include a copy of the samples in the test dataset in the training dataset.
   - [ ] C. 通过复制阳性样本并在复制的数据中加入少量的噪声，产生更多的阳性样本。 Generate more positive samples by duplicating the positive samples and adding a small amount of noise to the duplicated data.
   - [ ] D. 改变成本函数，使假阴性对成本值的影响高于假阳性。 Change the cost function so that false negatives have a higher impact on the cost value than false positives.
   - [ ] E. 改变成本函数，使假阳性对成本值的影响高于假阴性。 Change the cost function so that false, positives have a higher impact on the cost value than false negatives.

    <details>
      <summary>Answer</summary>

      答案CD。

   </details>

8. 一位数据科学家正在开发一个机器学习模型，根据收集到的关于每个病人和他们的治疗计划的信息，预测未来病人的结果。该模型应该输出一个连续值作为其预测值。可用的数据包括一组4，000名患者的标记结果。研究的对象是一群65岁以上的人，他们患有一种已知会随着年龄增长而恶化的特殊疾病。最初的模型表现不佳。在审查基础数据时，数据科学家注意到，在4,000个病人观察中，有450个病人的年龄被输入为0。数据科学家应该如何纠正这个问题。 A Data Scientist is developing a machine learning model to predict future patient outcomes based on information collected about each patient and their treatment plans. The model should output a continuous value as its prediction. The data available includes labeled outcomes for a set of 4, 000 patients. The study was conducted on a group of individuals over the age of 65 who have a particular disease that is known to worsen with age. Initial models have performed poorly. While reviewing the underlying data, the Data Scientist notices that, out of 4,000 patient observations, there are 450 where the patient age has been input as 0. The other features for these observations appear normal compared to the rest of the sample population. How should the Data Scientist correct this issue?
   - [ ] A. 从数据集中删除所有年龄被设置为0的记录。 Drop all records from the dataset where age has been set to 0.
   - [ ] B. 用数据集中的平均值或中位数来替换年龄为0的记录的档案值。 Replace the age filed value for records with a value of 0 with the mean or median value from the dataset.
   - [ ] C. 从数据集中删除年龄特征，用其余的特征训练模型。 Drop the age feature from the dataset and train the model using the rest of the features.
   - [ ] D. 使用k-means聚类法来处理缺失的特征。 Use k-means clustering to handle missing features.

   <details>
      <summary>Answer</summary>

      答案D。

   </details>

9. 一个数据科学团队正在设计一个数据集存储库，它将存储大量机器学习模型中常用的训练数据。由于数据科学家每天可能会创建任意数量的新数据集，该解决方案必须能够自动扩展，并具有成本效益。另外，必须能够使用SQL来探索数据。哪种存储方案最适合于这种情况？ A Data Science team is designing a dataset repository where it will store a large amount of training data commonly used in its machine learning models. As Data Scientists may create an arbitrary number of new datasets every day, the solution has to scale automatically and be cost-effective. Also, it must be possible to explore the data using SQL. Which storage scheme is MOST adapted to this scenario?
   - [ ] A. 将数据集作为文件存储在Amazon S3中。 Store datasets as files in Amazon S3.
   - [ ] B. 将数据集作为文件存储在连接到Amazon EC2实例的Amazon EBS卷中。 Store datasets as files in an Amazon EBS volume attached to an Amazon EC2 instance.
   - [ ] C. 将数据集作为表存储在一个多节点的Amazon Redshift集群中。 Store datasets as tables in a multi-node Amazon Redshift cluster.
   - [ ] D. 将数据集作为全局表存储在Amazon DynamoDB中。 Store datasets as global tables in Amazon DynamoDB.

   <details>
      <summary>Answer</summary>

      答案A。

   </details>

10. 一位机器学习专家部署了一个模型，在一家公司的网站上提供产品推荐。起初，该模型表现非常好，导致客户平均购买更多产品。然而，在过去的几个月里，该专家注意到产品推荐的效果已经减弱，客户开始回到他们原来的习惯，减少消费。专家不确定发生了什么，因为该模型与一年多前的最初部署相比没有变化。专家应该尝试哪种方法来提高模型的性能？ A Machine Learning Specialist deployed a model that provides product recommendations on a company's website. Initially, the model was performing very well and resulted in customers buying more products on average. However, within the past few months the Specialist has noticed that the effect of product recommendations has diminished, and customers are starting to return to their original habits of spending less. The Specialist is unsure of what happened, as the model has not changed from its initial deployment over a year ago. Which method should the Specialist try to improve model performance?
    - [ ] A. 该模型需要完全重新设计，因为它无法处理产品库存变化。 The model needs to be completely re-engineered because it is unable to handle product inventory changes.
    - [ ] B. 应该对模型的超参数进行可预测的更新以防止漂移。 The model's hyperparameters should be predicably updated to prevent drift.
    - [ ] C. 该模型应定期使用原始数据从头开始训练，同时增加一个正则化项来处理产品库存变化。 The model should be periodically retrained from scratch using the original data while adding a regularization term to handle product inventory changes.
    - [ ] D. 该模型应定期使用原始训练数据和产品库存变化时的新数据进行重新训练。 The model should be periodically retrained using the original training data plus new data as product inventory changes.

    <details>
       <summary>Answer</summary>

       答案D。

    </details>

11. 一位为一家在线时尚公司工作的机器学习专家希望为该公司基于Amazon S3的数据湖建立一个数据摄取解决方案。该专家希望创建一套摄取机制，以实现未来的能力，包括 -实时分析； -历史数据的互动分析； -点击流分析； -产品推荐。该专家应该使用哪些服务？ A Machine Learning Specialist working for an online fashion company wants to build a data ingestion solution for the company’s Amazon S3-based data lake. The Specialist wants to create a set of ingestion mechanisms that will enable future capabilities comprised of: -Real-time analytics; -Interactive analytics of historical data; -Clickstream analytics; -Product recommendations. Which services should the Specialist use?
    - [ ] A. AWS Glue作为数据目录；Amazon Kinesis Data Streams和Amazon Kinesis Data Analytics用于实时数据分析；Amazon Kinesis Data Firehose用于交付给Amazon ES进行点击流分析；Amazon EMR用于生成个性化的产品推荐。 AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for real-time data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations.
    - [ ] B. Amazon Athena作为数据目录；Amazon Kinesis Data Streams和Amazon Kinesis Data Analytics用于近实时的数据洞察；Amazon Kinesis Data Firehose用于点击流分析；AWS Glue用于生成个性化产品推荐。 B. Amazon Athena as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for near-real-time data insights; Amazon Kinesis Data Firehose for clickstream analytics; AWS Glue to generate personalize product recommendations.
    - [ ] C. AWS Glue作为数据目录；Amazon Kinesis Data Streams和Amazon Kinesis Data Analytics用于历史数据洞察；Amazon Kinesis Data Firehose用于交付给Amazon ES进行点击流分析；Amazon EMR用于生成个性化产品推荐。 AWS Glue as the data catalog; Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon Kinesis Data Firehose for delivery to Amazon ES for clickstream analytics; Amazon EMR to generate personalized product recommendations.
    - [ ] D. Amazon Athena作为数据目录。亚马逊Kinesis数据流和亚马逊Kinesis数据分析用于历史数据洞察；亚马逊DynamoDB流用于点击流分析；AWS Glue用于生成个性化的产品推荐。 Amazon Athena as the data catalog: Amazon Kinesis Data Streams and Amazon Kinesis Data Analytics for historical data insights; Amazon DynamoDB streams for clickstream analytics; AWS Glue to generate personalized product recommendations.

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

12. 一家公司在对Amazon SageMaker中默认的内置图像分类算法进行训练时观察到准确率很低。数据科学团队希望使用Inception神经网络架构而不是Resnet架构。以下哪种方法可以达到这个目的？(选择两个。) A company is observing low accuracy while training on the default built-in image classification algorithm in Amazon SageMaker. The Data Science team wants to use an Inception neural network architecture instead of a Resnet architecture. Which of the following will accomplish this? (Choose two.)
    - [ ] A. 定制内置的图像分类算法以使用Inception，并将其用于模型训练。 Customize the built-in image classification algorithm to use Inception and use this for model training.
    - [ ] B. 与SageMaker团队创建一个支持案例，将默认的图像分类算法改为Inception。 Create a support case with the SageMaker team to change the default image classification algorithm to Inception.
    - [ ] C. 将一个Docker容器与加载了Inception网络的TensorFlow Estimator捆绑在一起，并使用它进行模型训练。 Bundle a Docker container with TensorFlow Estimator loaded with an Inception network and use this for model training.
    - [ ] D. 在Amazon SageMaker中使用自定义代码，用TensorFlow Estimator来加载带有Inception网络的模型，并使用它来进行模型训练。 Use custom code in Amazon SageMaker with TensorFlow Estimator to load the model with an Inception network, and use this for model training,
    - [ ] E. 下载和apt-get安装inception网络代码到Amazon EC2实例中，并在Amazon SageMaker中使用该实例作为Jupyter笔记本。 Download and apt-get install the inception network code into an Amazon EC2 instance and use this instance as a Jupyter notebook in Amazon SageMaker.

    <details>
       <summary>Answer</summary>

       答案CD。

    </details>

13. 一位机器学习专家建立了一个图像分类深度学习模型。然而，该专家遇到了一个过拟合问题，训练和测试的准确率分别为99%和75%。该专家应该如何解决这个问题，其背后的原因是什么？ A Machine Learning Specialist built an image classification deep learning model. However, the Specialist ran into an overfitting problem in which the training and testing accuracies were 99% and 75%, respectively. How should the Specialist address this issue and what is the reason behind it?
    - [ ] A. 应该提高学习率，因为优化过程被困于局部最小值。 The learning rate should be increased because the optimization process was trapped at a local minimum.
    - [ ] B. 应该提高平坦层的dropout率，因为模型的泛化程度不够高。 The dropout rate at the flatten layer should be increased because the model is not generalized enough.
    - [ ] C. 挨着扁平层的密集层的维度应该增加，因为模型不够复杂。 The dimensionality of dense layer next to the flatten layer should be increased because the model is not complex enough.
    - [ ] D. 应增加历时数，因为优化过程在达到全局最小值之前就被终止了。 The epoch number should be increased because the optimization process was terminated before it reached the global minimum.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

14. 一个机器学习团队使用Amazon SageMaker来训练一个Apache MXNet手写数字分类器模型，使用的是一个研究数据集。该团队希望在模型过拟合时收到通知。审计人员希望查看Amazon SageMaker的日志活动报告，以确保没有未经授权的API调用。机器学习团队应该怎么做，才能以最少的代码和最少的步骤来解决这些要求？ A Machine Learning team uses Amazon SageMaker to train an Apache MXNet handwritten digit classifier model using a research dataset. The team wants to receive a notification when the model is overfitting. Auditors want to view the Amazon SageMaker log activity report to ensure there are no unauthorized API calls. What should the Machine learning team do to address the requirements with the least amount of code and fewest steps?
    - [ ] A. 实施一个AWS Lambda函数，将Amazon SageMaker API调用记录到Amazon S3。添加代码，将自定义指标推送到Amazon CloudWatch。在CloudWatch中创建一个警报与Amazon SNS创建一个警报，以便在模型过拟合时收到通知。 Implement an AWS Lambda function to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.
    - [ ] B. B 使用AWS CloudTrail来记录Amazon SageMaker API调用到Amazon S3。添加代码来推送自定义指标到Amazon CloudWatch。在CloudWatch中用Amazon SNS创建一个警报，以便在模型过拟合时收到通知。 Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.
    - [ ] C. 实现一个AWS Lambda函数，将Amazon SageMaker API调用记录到AWS CloudTrail。添加代码以推送自定义指标到Amazon CloudWatch。在CloudWatch中用Amazon SNS创建一个警报，以便在模型过拟合时收到通知。 Implement an AWS Lambda function to log Amazon SageMaker API calls to AWS CloudTrail. Add code to push a custom metric to Amazon CloudWatch. Create an alarm in CloudWatch with Amazon SNS to receive a notification when the model is overfitting.
    - [ ] D. 使用AWS CloudTrail将Amazon SageMaker API调用记录到Amazon S3.设置Amazon SNS以在模型过拟合时接收通知。 Use AWS CloudTrail to log Amazon SageMaker API calls to Amazon S3. Set up Amazon SNS to receive a notification when the model is overfitting.

    <details>
       <summary>Answer</summary>

       答案B。

    </details>

15. 一位机器学习专家正在使用线性模型，如线性回归和逻辑回归，为大量的特征建立预测模型。在探索性的数据分析过程中，专家观察到许多特征是高度相关的。这可能使模型不稳定。应该怎样做才能减少有这么多特征的影响？ A Machine Learning Specialist is building a prediction model for a large number of features using linear models, such as linear regression and logistic regression. During exploratory data analysis, the Specialist observes that many features are highly correlated with each other. This may make the model unstable. What should be done to reduce the impact of having such a large number of features?
    - [ ] A. 对高度相关的特征进行一次编码。 Perform one-hot encoding on highly correlated features.
    - [ ] B. 在高度相关的特征上使用矩阵乘法。 Use matrix multiplication on highly correlated features.
    - [ ] C. 使用主成分分析（PCA）创建一个新的特征空间。 Create a new feature space using principal component analysis (PCA).
    - [ ] D. 应用皮尔逊相关系数。 Apply the Pearson correlation coefficient.

    <details>
       <summary>Answer</summary>

       答案C。

    </details>

16. 一位机器学习专家正在对描述纽约市公共交通的数据集实施一个完整的贝叶斯网络。其中一个随机变量是离散的，它代表了纽约人等待公交车的分钟数，因为公交车每10分钟循环一次，平均为3分钟。对于这个变量，ML专家应该使用哪个先验概率分布？ A Machine Learning Specialist is implementing a full Bayesian network on a dataset that describes public transit in New York City. One of the random variables is discrete, and represents the number of minutes New Yorkers wait for a bus given that the buses cycle every 10 minutes, with a mean of 3 minutes. Which prior probability distribution should the ML Specialist use for this variable?
    - [ ] A. 泊松分布 Poisson distribution
    - [ ] B. 均匀分布 Uniform distribution
    - [ ] C. 正态分布 Normal distribution
    - [ ] D. 二项分布 Binomial distribution

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

17. 一家大公司的数据科学团队使用Amazon SageMaker笔记本来访问存储在Amazon S3桶中的数据。IT安全团队担心支持互联网的笔记本实例会产生安全漏洞，在实例上运行的恶意代码可能会破坏数据隐私。公司规定，所有的实例都在一个安全的VPC内，没有互联网接入，数据通信流量必须留在AWS网络内。数据科学团队应该如何配置笔记本实例的位置以满足这些要求？ A Data Science team within a large company uses Amazon SageMaker notebooks to access data stored in Amazon S3 buckets. The IT Security team is concerned that internet-enabled notebook instances create a security vulnerability where malicious code running on the instances could compromise data privacy. The company mandates that all instances stay within a secured VPC with no internet access and data communication traffic must stay within the AWS network. How should the Data Science team configure the notebook instance placement to meet these requirements?
    - [ ] A. 将Amazon SageMaker笔记本与VPC中的一个私有子网联系起来。将Amazon SageMaker终端和S3桶放在同一个VPC内。 Associate the Amazon SageMaker notebook with a private subnet in a VPC. Place the Amazon SageMaker endpoint and S3 buckets within the same VPC.
    - [ ] B. 将Amazon SageMaker笔记本与VPC中的一个私有子网联系起来。使用IAM策略来授予对 Amazon S3 和 Amazon SageMaker 的访问权。 Associate the Amazon SageMaker notebook with a private subnet in a VPC. Use IAM policies to grant access to Amazon S3 and Amazon SageMaker
    - [ ] C. 将Amazon SageMaker笔记本与VPC中的一个私有子网联系起来。确保VPC有S3 VPC端点和Amazon SageMaker VPC端点连接到它。 Associate the Amazon SageMaker notebook with a private subnet in a VPC. Ensure the VPC has S3 VPC endpoints and Amazon SageMaker VPC endpoints attached to it.
    - [ ] D. 将Amazon SageMaker笔记本与VPC中的一个私有子网联系起来。确保VPC有一个NAT网关和一个相关的安全组，只允许向外连接Amazon S3和Amazon SageMaker。 Associate the Amazon SageMaker notebook with a private subnet in a VPC. Ensure the VPC has a NAT gateway and an associated security group allowing only outbound connections to Amazon S3 and Amazon SageMaker.

    <details>
       <summary>Answer</summary>

       简单题，答案C。

    </details>

18. 一位机器学习专家创建了一个深度学习神经网络模型，在训练数据上表现良好，但在测试数据上表现不佳。该专家应该考虑使用以下哪种方法来纠正这种情况？(选择三个) A Machine Learning Specialist has created a deep learning neural network model that performs well on the training data but performs poorly on the test data. Which of the following methods should the Specialist consider using to correct this? (Choose three)
    - [ ] A. 降低正则化程度 Decrease regularization
    - [ ] B. 增加正规化 Increase regularization
    - [ ] C. 增加dropout Increase dropout
    - [ ] D. 减少dropout Decrease dropout
    - [ ] E. 增加特征组合 Increase feature combinations
    - [ ] F. 减少特征组合 Decrease feature combinations

    <details>
       <summary>Answer</summary>

       答案BCF。

    </details>

19. 一位数据科学家需要为高速、实时流数据创建一个无服务器的摄取和分析解决方案。摄取过程必须缓冲并将传入的记录从JSON转换为查询优化的柱状格式，而不会有数据损失。输出的数据存储必须是高度可用的，分析师必须能够对数据运行SQL查询，并连接到现有的商业智能仪表板。数据科学家应该建立哪种解决方案来满足这些要求？ A Data Scientist needs to create a serverless ingestion and analytics solution for high-velocity, real-time streaming data. The ingestion process must buffer and convert incoming records from JSON to a query optimized, columnar format without data loss. The output datastore must be highly available, and Analysts must be able to run SQL queries against the data and connect to existing business intelligence dashboards. Which solution should the Data Scientist build to satisfy the requirements?
    - [ ] A. Create a schema in the AWS Glue Data Catalog of the incoming data format. Use an Amazon Kinesis Data Firehose delivery stream to stream the data and transform the data to Apache Parquet or ORC format using the AWS Glue Data Catalog before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena and connect to BI tools using the Athena Java Database Connectivity (JDBC) connector. 在AWS Glue数据目录中创建一个传入数据格式的模式。使用Amazon Kinesis Data Firehose交付流来流化数据，并在交付到Amazon S3之前使用AWS Glue Data Catalog将数据转换为Apache Parquet或ORC格式。让分析师使用Amazon Athena直接从Amazon S3查询数据，并使用Athena Java数据库连接（JDBC）连接器连接到BI工具。
    - [ ] B.  Write each JSON record to a staging location in Amazon S3. Use the S3. Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and writes the data to a processed data location in Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena and connect to Bl tools using the Athena Java Database Connectivity (JDBC) connector. 将每个JSON记录写到Amazon S3的一个暂存位置。使用S3。Put事件触发AWS Lambda函数，将数据转换为Apache Parquet或ORC格式，并将数据写入Amazon S3的处理数据位置。让分析师使用Amazon Athena直接从Amazon S3查询数据，并使用Athena Java数据库连接（JDBC）连接器连接到BI工具。
    - [ ] C. Write each JSON record to a staging location in Amazon S3. Use the S3 Put event to trigger an AWS Lambda function that transforms the data into Apache Parquet or ORC format and inserts it into an Amazon RDS PostgreSQL database. Have the Analysts query and run dashboards from the RDS database. 将每条JSON记录写到Amazon S3的一个暂存位置。使用S3 Put事件触发AWS Lambda函数，将数据转换为Apache Parquet或ORC格式，并将其插入到Amazon RDS PostgreSQL数据库。让分析师从RDS数据库中查询和运行仪表盘。
    - [ ] D. Use Amazon Kinesis Data Analytics to ingest the streaming data and perform real-time SQL queries to convert the records to Apache Parquet before delivering to Amazon S3. Have the Analysts query the data directly from Amazon S3 using Amazon Athena and connect to Bl tools using the Athena Java Database Connectivity (JDBC) connector. 使用Amazon Kinesis Data Analytics来摄取流媒体数据，并执行实时SQL查询，将记录转换为Apache Parquet，然后交付给Amazon S3。让分析师使用Amazon Athena直接从Amazon S3查询数据，并使用Athena Java数据库连接（JDBC）连接器连接到BI工具。

    <details>
       <summary>Answer</summary>

       答案A。

    </details>

20. 一个在线经销商有一个大型的、多列的数据集，其中有一列缺少30%的数据。一位机器学习专家认为，数据集中的某些列可以用来重建丢失的数据。该专家应该使用哪种重建方法来保持数据集的完整性？ An online reseller has a large, multi-column dataset with one column missing 30% of its data. A Machine Learning Specialist believes that certain columns in the dataset could be used to reconstruct the missing data. Which reconstruction approach should the Specialist use to preserve the integrity of the dataset?
    - [ ] A. 列表式删除 Listwise deletion
    - [ ] B. 最后的观察结果向前推进 Last observation carried forward
    - [ ] C. 多重归因 Multiple imputation
    - [ ] D. 平均替代 Mean substitution

    <details>
       <summary>Answer</summary>

       答案C。

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

26. A Solutions Architect must create a cost-effective backup solution for a company's 500MB source repository of proprietary and sensitive applications. The repository runs on Linux and backs up daily to tape. Tape backups are stored for 1 year. The current solutions are not meeting the company's needs because it is a manual process that is prone to error, expensive to maintain, and does not meet the need for a Recover Point Objective (RPO) of 1 hour or Recovery Time Objective (RTO) of 2 hours. The new disaster recovery requirement is for backups to be stored offsite and to be able to restore a single file if needed. Which solution meets the customer's needs for RTO, RPO, and disaster recover with the LEAST effort and expense?
    - [ ] A. Replace local tapes with an AWS Storage Gateway virtual tape library to integrate with current backup software. Run backups nightly and store the virtual tapes on Amazon S3 standard storage in US-EAST-1. Use cross-region replication to create a second copy in US-WEST-2. Use Amazon S3 lifecycle policies to perform automatic migration to Amazon Glacier and deletion of expired backups after 1 year?
    - [ ] B. Configure the local source repository to synchronize files to an AWS Storage Gateway file Amazon gateway to store backup copies in an Amazon S3 Standard bucket. Enable versioning on the Amazon S3 bucket. Create Amazon S3 lifecycle policies to automatically migrate old versions of objects to Amazon S3 Standard Infrequent Access, then Amazon Glacier, then delete backups after 1 year.
    - [ ] C. Replace the local source repository storage with a Storage Gateway stored volume. Change the default snapshot frequency to 1 hour. Use Amazon S3 lifecycle policies to archive snapshots to Amazon Glacier and remove old snapshots after 1 year. Use cross-region replication to run on create a copy of the snapshots in US-WEST-2.
    - [ ] D. Replace the local source repository storage with a Storage Gateway cached volume. Create a snapshot schedule to take hourly snapshots. Use an Amazon CloudWatch Events schedule expression rule to run on hourly AWS Lambda task to copy snapshots from US-EAST-1 to US-WEST-2.

    <details>
       <summary>Answer</summary>

       - [ ] A. RPO无法实现1小时
       - [x] B. 正确
       - [ ] C. 无法恢复单个文件
       - [ ] D. 缓存卷不用于文件备份

    </details>

27. A company CFO recently analyzed the company's AWS monthly bill and identified an opportunity to reduce the cost for AWS Elastic Beanstalk environments in use. The CFO has asked a Solutions Architect to design a highly available solution that will spin up an Elastic Beanstalk environment in the morning and terminate it at the end of the day. The solution should be designed with minimal operational overhead and to minimize costs. It should also be able to handle the increased use of Elastic Beanstalk environments among different teams and must provide a one-stop scheduler solution for all teams to keep the operational costs low. What design will meet these requirements?
    - [ ] A. Set up a Linux EC2 Micro instance. Configure an IAM role to allow the start and stop of the Elastic Beanstalk environment and attach it to the instance. Create scripts on the instance to start and stop the Elastic Beanstalk environment. Configure cron jobs on the instance to execute the scripts.
    - [ ] B. Develop AWS Lambda functions to start and stop the Elastic Beanstalk environment. Configure a Lambda execution role granting Elastic Beanstalk environment start/stop permissions and assign the role to the Lambda functions. Configure cron expression Amazon CloudWatch Events rules to trigger the Lambda functions.
    - [ ] C. Develop an AWS Step Functions state machine with `wait` as its type to control the start and stop time. Use the activity task to start and stop the Elastic Beanstalk environment. Create a role for Step Functions to allow it to start and stop the Elastic Beanstalk environment. Invoke Step Functions daily.
    - [ ] D. Configure a time-based Auto Scaling group. In the morning, have the Auto Scaling group scale up an Amazon EC2 instance and put the Elastic Beanstalk environment start command in the EC2 instance user data. At the end of the day, scale down the instance number to 0 to terminate the EC2 instance.

    <details>
       <summary>Answer</summary>

       简单题，答案B -> [ref](https://aws.amazon.com/cn/premiumsupport/knowledge-center/schedule-elastic-beanstalk-stop-restart/)

    </details>

28. A company plans to move regulated and security-sensitive businesses to AWS. The Security team is developing a framework to validate the adoption of AWS best practice and industry recognized compliance standards. The AWS Management Console is the preferred method for teams to provision resources. Which strategies should a Solutions Architect use to meet the business requirements and continuously assess, audit, and monitor the configurations of AWS resources? (Choose two.)
    - [ ] A. Use AWS Config rules to periodically audit changes to AWS resources and monitor the compliance of the configuration. Develop AWS Config custom rules using AWS Lambda to establish a test-driven development approach, and further automate the evaluation of configuration changes against the required controls.
    - [ ] B. Use Amazon CloudWatch Logs agent to collect all the AWS SDK logs. Search the log data using a pre-defined set of filter patterns that machines mutating API calls. Send notifications using Amazon CloudWatch alarms when unintended changes are performed. Archive log data by using a batch export to Amazon S3 and then Amazon Glacier for a long-term retention and auditability.
    - [ ] C. Use AWS CloudTrail events to assess management activities of all AWS accounts. Ensure that CloudTrail is enabled in all accounts and available AWS services. Enable trails, encrypt CloudTrail event log files with an AWS KMS key, and monitor recorded activities with CloudWatch Logs.
    - [ ] D. Use the Amazon CloudWatch Events near-real-time capabilities to monitor system events patterns, and trigger AWS Lambda functions to automatically revert non-authorized changes in AWS resources. Also, target Amazon SNS topics to enable notifications and improve the response time of incident responses.
    - [ ] E. Use CloudTrail integration with Amazon SNS to automatically notify unauthorized API activities. Ensure that CloudTrail is enabled in all accounts and available AWS services. Evaluate the usage of Lambda functions to automatically revert non-authorized changes in AWS resources.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. CloudWatch不用于账号审计
       - [x] C. 正确
       - [ ] D. 同B
       - [ ] E. CloudTrail没有filter功能

    </details>

29. A company is running a high-user-volume media-sharing application on premises. It currently hosts about 400 TB of data with millions of video files. The company is migrating this application to AWS to improve reliability and reduce costs. The Solutions Architecture team plans to store the videos in an Amazon S3 bucket and use Amazon CloudFront to distribute videos to users. The company needs to migrate this application to AWS 10 days with the least amount of downtime possible. The company currently has I Gbps connectivity to the Internet with 30 percent free capacity. Which of the following solutions would enable the company to migrate the workload to AWS and meet all of the requirements?
    - [ ] A. Use a multi-part upload in Amazon S3 client to parallel-upload the data to the Amazon S3 bucket over the Internet. Use the throttling feature to ensure that the Amazon S3 client does not use more than 30 percent of available Internet capacity.
    - [ ] B. Request an AWS Snowmobile with 1 PB capacity to be delivered to the data center. Load the data into Snowmobile and send it back to have AWS download that data to the Amazon S3 bucket. Sync the new data that was generated while migration was in flight.
    - [ ] C. Use an Amazon S3 client to transfer data from the data center to the Amazon S3 bucket over the Internet. Use the throttling feature to ensure the Amazon S3 client does not use more than 30 percent of available Internet capacity.
    - [ ] D. Request multiple AWS Snowball devices to be delivered to the data center. Load the data concurrently into these devices and send it back. Have AWS download that data to the Amazon S3 bucket. Sync the new data that was generated while migration was in flight.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

30. A company has developed a new billing application that will be released in two weeks. Developers are testing the application running on 10 EC2 instances managed by an Auto Scaling group in subnet 172.31.0.0/24 within VPC A with CIDR block 172.31.0.0/16. The Developers noticed connection timeout errors in the application logs while connecting to an Oracle database running on an Amazon EC2 instance in the same region within VPC B with CIDR block 172.50.0.0/16. The IP of the database instance is hard in the application instances. Which recommendations should a Solutions Architect present to the Developers to solve the problem in a secure way with minimal maintenance and overhead?
    - [ ] A. Disable the `SrcDestCheck` attribute for all instances running the application and Oracle Database. Change the default route of VPC A to point ENI of the Oracle Database that has an IP address assigned within the range of 172.50.0.0/26.
    - [ ] B. Create and attach internet gateways for both VPCs. Configure default routes to the Internet gateways for both VPCs. Assign an Elastic IP for each Amazon EC2 instance in VPC A.
    - [ ] C. Create a VPC peering connection between the two VPCs and add a route to the routing table of VPC A that points to the IP address range of 172.50.0.0/16.
    - [ ] D. Create an additional Amazon EC2 instance for each VPC as a customer gateway; create one virtual private gateway (VGW) for each VPC, configure an end-to-end VPC, and advertise the routes for 172.50.0.0/16.

    <details>
       <summary>Answer</summary>

       简单题，答案C

    </details>

31. A Solutions Architect has been asked to look at a company's Amazon Redshift cluster, which has quickly become an integral part of its technology and supports key business process. The Solutions Architect is to increase the reliability and availability of the cluster and provide options to ensure that if an issue arises, the cluster can either operate or be restored within four hours. Which of the following solution options BEST addresses the business need in the most cost effective manner?
    - [ ] A. Ensure that the Amazon Redshift cluster has been set up to make use of Auto Scaling groups with the nodes in the cluster spread across multiple Availability Zones.
    - [ ] B. Ensure that the Amazon Redshift cluster creation has been template using AWS CloudFormation so it can easily be launched in another Availability Zone and data populated from the automated Redshift back-ups stored in Amazon S3.
    - [ ] C. Use Amazon Kinesis Data Firehose to collect the data ahead of ingestion into Amazon Redshift and create clusters using AWS CloudFormation in another region and stream the data to both clusters.
    - [ ] D. Create two identical Amazon Redshift clusters in different regions (one as the primary, one as the secondary). Use Amazon S3 cross-region replication from the primary to secondary). Use Amazon S3 cross-region replication from the primary to secondary region, which triggers an AWS Lambda function to populate the cluster in the secondary region.
  
    <details>
       <summary>Answer</summary>

       - [ ] A. Amazon Redshift cluster只有单可用区
       - [x] B. 正确
       - [ ] C. 四小时的RPO所以不需要复制集群
       - [ ] D. Lambda有可能超时

    </details>

32. A company prefers to limit running Amazon EC2 instances to those that were launched from AMIs pre-approved by the Information Security department. The Development team has an agile continuous integration and deployment process that cannot be stalled by the solution. Which method enforces the required controls with the LEAST impact on the development process? (Choose two.)
    - [ ] A. Use IAM policies to restrict the ability of users or other automated entities to launch EC2 instances based on a specific set of pre-approved AMIs, such as those tagged in a specific way by Information Security.
    - [ ] B. Use regular scans within Amazon Inspector with a custom assessment template to determine if the EC2 instance that the Amazon Inspector Agent is running on is based upon a pre-approved AMI. If it is not, shut down the instance and inform information Security by email that this occurred.
    - [ ] C. Only allow launching ofEC2 instances using a centralized DevOps team, which is given work packages via notifications from an internal ticketing system. Users make requests for resources using this ticketing tool, which has manual information security approval steps to ensure that EC2 instances are only launched from approved AMIs.
    - [ ] D. Use AWS Config rules to spot any launches of EC2 instances based on non-approved AMIs, trigger an AWS Lambda function to automatically terminate the instance, and publish a message to an Amazon SNS topic to inform Information Security that this occurred.
    - [ ] E. Use a scheduled AWS Lambda function to scan through the list of running instances within the virtual private cloud (VPC) and determine if any of these are based on unapproved AMIs. Publish a message to an SNS topic to inform Information Security that this occurred and then shut down the instance.
  
    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. AWS Inspector是用于扫描漏洞的，并不是用去AMI
       - [ ] C. 这个不属于敏捷
       - [x] D. 正确
       - [ ] E. 并不存在预定的预定的AWS Lambda

    </details>

33. A Company has a security event whereby an Amazon S3 bucket with sensitive information was made public. Company policy is to never have public S3 objects, and the Compliance team must be informed immediately when any public objects are identified. How can the presence of a public S3 object be detected, set to trigger alarm notifications, and automatically remediated in the future? (Choose two.)
    - [ ] A. Turn on object-level logging for Amazon S3. Turn on Amazon S3 event notifications to notify by using an Amazon SNS topic when a PutObject API call is made with a public-read permission.
    - [ ] B. Configure an Amazon CloudWatch Events rule that invokes an AWS Lambda function to secure the S3 bucket.
    - [ ] C. Use the S3 bucket permissions for AWS Trusted Advisor and configure a CloudWatch event to notify by using Amazon SNS.
    - [ ] D. Turn on object-level logging for Amazon S3. Configure a CloudWatch event to notify by using an SNS topic when a PutObject API call with public-read permission is detected in the AWS CloudTrail logs.
    - [ ] E. Schedule a recursive Lambda function to regularly change all object permissions inside the S3 bucket.
  
    <details>
       <summary>Answer</summary>

       - [ ] A. S3 notification中不存在权限信息
       - [ ] B. 这一看就错的
       - [ ] C. Trusted Advisor不干这事儿
       - [x] D. 正确
       - [x] E. 正确

    </details>

34. A company is using an Amazon CloudFront distribution to distribute both static and dynamic content from a web application running behind an Application Load Balancer. The web application requires user authorization and session tracking for dynamic content. The CloudFront distribution has a single cache behavior configured to forward the Authorization, Host, and User-Agent HTTP whitelist headers and a session cookie to the origin. All other cache behavior settings are set to their default value. A valid ACM certificate is applied to the CloudFront distribution with a matching CNAME in the distribution settings. The ACM certificate is also applied to the HTTPS listener for the Application Load Balancer. The CloudFront origin protocol policy is set to HTTPS only. Analysis of the cache statistics report shows that the miss rate for this distribution is very high. What can the Solutions Architect do to improve the cache hit rate for this distribution without causing the SSL/TLS handshake between CloudFront and the Application Load Balancer to fail?
    - [ ] A. Create two cache behaviors for static and dynamic content. Remove the User-Agent and Host HTTP headers from the whitelist headers section on both if the cache behaviors. Remove the session cookie from the whitelist cookies section and the Authorization HTTP header from the whitelist headers section for cache behavior configured for static content.
    - [ ] B. Remove the User-Agent and Authorization HTTPS headers from the whitelist headers section of the cache behavior. Then update the cache behavior to use presigned cookies for authorization.
    - [ ] C. Remove the Host HTTP header from the whitelist headers section and remove the session cookie from the whitelist cookies section for the default cache behavior. Enable automatic object compression and use Lambda@Edge viewer request events for user authorization.
    - [ ] D. Create two cache behaviors for static and dynamic content. Remove the User-Agent HTTP header from the whitelist headers section on both cache behaviors. Remove the session cookie from the whitelist cookies section and the Authorization HTTP header from the whitelist headers section for cache behavior configured for static content.

    <details>
       <summary>Answer</summary>

       因为它同时分发静态和动态内容。你应该有两个缓存行为。所以选项B和C被排除了。现在在A和D之间，Host HTTP headers是必须的，而且你不能删除。所以唯一有效的选项是D -> [ref](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/understanding-the-cache-key.html
)

    </details>

35. An organization has a write-intensive mobile application that uses Amazon API Gateway, AWS Lambda, and Amazon DynamoDB. The application has scaled well, however, costs have increased exponentially because of higher than anticipated Lambda costs. The application's use is unpredictable, but there has been a steady 20% increase in utilization every month. While monitoring the current Lambda functions, the Solutions Architect notices that the execution time averages 4.5 minutes. Most of the wait time is the result of a high-latency network call to a 3TB MySQL database server that is on-premises. A VPN is used to connect to the VPC, so the Lambda functions have been configured with a five-minute timeout. How can the Solutions Architect reduce the cost of the current architecture?
    - [ ] A. Replace the VPN with AWS Direct Connect to reduce the network latency to the on-premises MySQL database. Enable local caching in the mobile application to reduce the Lambda function invocation calls. Monitor the Lambda function performance; gradually adjust the timeout and memory properties to lower values while maintaining an acceptable execution time. Offload the frequently accessed records from DynamoDB to Amazon ElastiCache.
    - [ ] B. Replace the VPN with AWS Direct Connect to reduce the network latency to the on-premises MySQL database. Cache the API Gateway results to Amazon CloudFront. Use Amazon EC2 Reserved Instances instead of Lambda. Enable Auto Scaling on EC2 and use Spot Instances during peak times. Enable DynamoDB Auto Scaling to manage target utilization.
    - [ ] C. Migrate the MySQL database server into a Multi-AZ Amazon RDS for MySQL. Enable caching of the Amazon API Gateway results in Amazon CloudFront to reduce the number of Lambda function invocations. Monitor the Lambda function performance; gradually adjust the timeout and memory properties to lower values while maintaining an acceptable execution time. Enable DynamoDB Accelerator for frequently accessed records and enable the DynamoDB Auto Scaling feature.
    - [ ] D. Migrate the MySQL database server into a Multi-AZ Amazon RDS for MySQL. Enable API caching on API Gateway to reduce the number of Lambda function invocations. Continue to monitor the AWS Lambda function performance; gradually adjust the timeout and memory properties to lower values while maintaining an acceptable execution time. Enable Auto Scaling in DynamoDB.

    <details>
       <summary>Answer</summary>

       由于数据库在本地，所以首先考虑迁移到AWS上，排除AB，C选项使用DAX还是挺贵的，答案D

    </details>

36. A company runs a video processing platform. Files are uploaded by users who connect to a web server, which stores them on an Amazon EFS share. This web server is running on a single Amazon EC2 instance. A different group of instances, running in an Auto Scaling group, scans the EFS share directory structure for new files to process and generates new videos (thumbnails, different resolution, compression, etc.) according to the instructions file, which is uploaded along with the video files. A different application running on a group of instances managed by an Auto Scaling group processes the video files and then deletes them from the EFS share. The results are stored in an S3 bucket. Links to the processed video files are emailed to the customer. The company has recently discovered that as they add more instances to the Auto Scaling Group, many files are processed twice, so image processing speed is not improved. The maximum size of these video files is 2GB. What should the Solutions Architect do to improve reliability and reduce the redundant processing of video files?
    - [ ] A. Modify the web application to upload the video files directly to Amazon S3. Use Amazon CloudWatch Events to trigger an AWS Lambda function every time a file is uploaded and have this Lambda function put a message into an Amazon queue for new files and use the queue depth metric to scale instances in the video processing Auto Scaling group.
    - [ ] B. Set up a cron job on the web server instance to synchronize the contents of the EFS share into Amazon S3. Trigger an AWS Lambda function every time a file is uploaded to process the video file and store the results in Amazon S3. Using Amazon CloudWatch Events trigger an Amazon SES job to send an email to the customer containing the link to the processed file.
    - [ ] C. Rewrite the web application to run directly from Amazon S3 and use Amazon API Gateway to upload the video files to an S3 bucket. Use an S3 trigger to run an AWS Lambda function each time a file is uploaded to process and store new video files in a different bucket. Using CloudWatch Events, trigger an SES job to send an email to the customer containing the link to the processed file.
    - [ ] D. Rewrite the application to run from Amazon S3 and upload the video files to an S3 bucket. Each time a new file is uploaded, trigger an AWS Lambda function to put a message in an SQS queue containing the link and the instructions. Modify the video processing application to read from the SQS queue and the S3 bucket. Use the queue depth metric to adjust the size of the Auto Scaling group for video processing instances.

    <details>
       <summary>Answer</summary>

       - [ ] A. CloudWatch里没有S3事件 -> [ref](https://docs.aws.amazon.com/zh_cn/AmazonCloudWatch/latest/events/WhatIsCloudWatchEvents.html)
       - [ ] B. 同A
       - [ ] C. Lambda有同时执行限制（1000）
       - [x] D. 正确

    </details>

37. A Solutions Architect must establish a patching plan for a large mixed fleet of Windows and Linux servers. The patching plan must be implemented securely, be audit ready, and comply with the company's business requirements. Which option will meet these requirements with MINIMAL effort?
    - [ ] A. Install and use an OS-native patching service to manage the update frequency and release approval for all instances. Use AWS Config to verify the OS state on each instance and report on any patch compliance issues.
    - [ ] B. Use AWS Systems Manager on all instances to manage patching. Test patches outside of production and then deploy during a maintenance window with the appropriate approval.
    - [ ] C. Use AWS OpsWorks for Chef Automate to run a set of scripts that will iterate through all instances of a given type. Issue the appropriate OS command to get and install updates on each instance, including any required restarts during the maintenance window.
    - [ ] D. Migrate all applications to AWS OpsWorks and use OpsWorks automatic patching support to keep the OS up-to-date following the initial installation. Use AWS Config to provide audit and compliance reporting.

    <details>
       <summary>Answer</summary>

       - [ ] A. 应该使用AWS提供的方法
       - [x] B. 正确
       - [ ] C. OpsWorks for Chef Automate不是干这事儿的
       - [ ] D. 同C

    </details>

38. A Solutions Architect must design a highly available, stateless, REST service. The service will require multiple persistent storage layers for service object meta information and the delivery of content. Each request needs to be authenticated and securely processed. There is a requirement to keep costs as low as possible. How can these requirements be met?
    - [ ] A. Use AWS Fargate to host a container that runs a self-contained REST service. Set up an Amazon ECS service that is fronted by an Application Load Balancer (ALB). Use a custom authenticator to control access to the API Store request meta information in Amazon DynamoDB with Auto Scaling and static content in a secured S3 bucket. Make secure signed requests for Amazon S3 objects and proxy the data through the REST service interface.
    - [ ] B. Use AWS Fargate to host a container that runs a self-contained REST service. Set up an ECS service that is fronted by a cross-zone ALB. Use an Amazon Cognito user pool to control access to the API Store request meta information in DynamoDB with Auto Scaling and static content in a secured S3 bucket. Generate pre-signed URLs when returning references to content stored in Amazon S3.
    - [ ] C. Set up Amazon API Gateway and create the required API resources and methods. Use an Amazon Cognito user pool to control access to the API. Configure the methods to use AWS Lambda proxy integrations and process each resource with a unique AWS Lambda function. Store request meta information in DynamoDB with Auto Scaling and static content in a secured S3 bucket. Generate presigned URLs when returning references to content stored in Amazon S3.
    - [ ] D. Set up Amazon API Gateway and create the required API resources and methods. Use an Amazon API Gateway custom authorizer to control access to the API. Configure the methods to use AWS Lambda custom integrations and process each resource with a unique Lambda function. Store request meta information in an Amazon ElastiCache Multi-AZ cluster and static content in a secured S3 bucket. Generate presigned URLs when returning references to content stored in Amazon S3.

    <details>
       <summary>Answer</summary>

       - [ ] A. 一个容器无法实现高可用性
       - [ ] B. 同A
       - [x] C. 正确
       - [ ] D. 涉及到Lambda集成问题，目前不太清楚 -> [ref](https://docs.aws.amazon.com/zh_cn/apigateway/latest/developerguide/set-up-lambda-integrations.html)

    </details>

39. A large company experienced a drastic increase in its monthly AWS spend. This is after Developers accidentally launched Amazon EC2 instances in unexpected regions. The company has established practices around least privileges for Developers and controls access to on-premises resources using Active Directory groups. The company now want to control costs by restricting the level of access that Developers have to the AWS Management Console without impacting their productivity. The company would also like to allow Developers to launch Amazon EC2 in only one region, without limiting access to other services in any region. How can this company achieve these new security requirements while minimizing the administrative burden on the Operations team?
    - [ ] A. Set up SAML-based authentication tied to an IAM role that has an AdministrativeAccess managed policy attached to it. Attach a customer managed policy that denies access to Amazon EC2 in each region except for the one required.
    - [ ] B. Create an IAM user for each Developer and add them to the developer IAM group that has the PowerUserAccess managed policy attached to it. Attach a customer managed policy that allows the Developers access to Amazon EC2 only in the required region.
    - [ ] C. Set up SAML-based authentication tied to an IAM role that has a PowerUserAccess managed policy and a customer managed policy that deny all the Developers access to any AWS services except AWS Service Catalog. Within AWS Service Catalog, create a product containing only the EC2 resources in the approved region.
    - [ ] D. Set up SAML-based authentication tied to an IAM role that has the PowerUserAccess managed policy attached to it. Attach a customer managed policy that denies access to Amazon EC2 in each region except for the one required.

    <details>
       <summary>Answer</summary>

       - [ ] A. AdministrativeAccess权限太大，他们依然可以使用IAM去给与启EC2的权限
       - [ ] B. 没毛用啊，他们依然可以去别得区域
       - [ ] C. Service Catalog只能完成部分事务，太严格了
       - [x] D. 正确

    </details>

40. A company is finalizing the architecture for its backup solution for applications running on AWS. All the applications run on AWS and use at least two Availability Zones in each tier. Company policy requires IT to durably store nightly backups f all its data in at least two locations:production and disaster recovery. The locations must be in different geographic regions. The company also needs the backup to be available to restore immediately at the production data center, and within 24 hours at the disaster recovery location. All backup processes must be fully automated. What is the MOST cost-effective backup solution that will meet all requirement?
    - [ ] A. Back up all the data to a large Amazon EBS volume attached to the backup media server in the production region. Run automated scripts to snapshot these volumes nightly and copy these snapshots to the disaster recovery region.
    - [ ] B. Back up all the data to Amazon S3 in the disaster recover region. Use a lifecycle policy to move this data to Amazon Glacier in the production region immediately. Only the data is replicated; remove the data from the S3 bucket in the disaster recovery region.
    - [ ] C. Back up all the data to Amazon Glacier in the production region. Set up cross-region replication of this data to Amazon Glacier in the disaster recovery region. Set up a lifecycle policy to delete any data older than 60 days.
    - [ ] D. Back up all the data to Amazon S3 in the production region. Set up cross-region replication of this S3 bucket to another region and set up a lifecycle policy in the second region to immediately move this data to Amazon Glacier.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

41. A company has an existing on-premises three-tier web application. The Linux web servers serve content from a centralized file share on a NAS sever because the content is refreshed several times a day from various sources. The existing infrastructure is not optimized, and the company would like to move to AWS to gain the ability to scale resources up and down in response to load. On-premises and AWS resources are connected using AWS Direct Connect. How can the company migrate the web infrastructure to AWS without delaying the content refresh process?
    - [ ] A. Create a cluster of web server Amazon EC2 instances behind a Classic Load Balancer on AWS. Share an Amazon EBS volume among all instances for the content. Schedule a periodic synchronization of this volume and the NAS server.
    - [ ] B. Create an on-premises file gateway using AWS Storage Gateway to replace the NAS server and replicate content to AWS. On the AWS side, mount the same Storage Gateway bucket to each web server Amazon EC2 instance to serve the content.
    - [ ] C. Expose an Amazon EFS share to on-premises users to serve as the NAS serve. Mount the same EFS share to the web server Amazon EC2 instances to serve the content.
    - [ ] D. Create web server Amazon EC2 instances on AWS in an Auto Scaling group. Configure a nightly process where the web server instances are updated from the NAS server.

    <details>
       <summary>Answer</summary>

       - [ ] A. EBS无法作为文件共享
       - [ ] B. S3无法挂载到EC2实例上
       - [x] C. 正确
       - [ ] D. 数据不同步

    </details>

42. A company has multiple AWS accounts hosting IT applications. An Amazon CloudWatch Logs agent is installed on all Amazon EC2 instances. The company wants to aggregate all security events in a centralized AWS account dedicated to log storage. Security Administrators need to perform near-real-time gathering and correlating of events across multiple AWS accounts. Which solution satisfies these requirements?
    - [ ] A. Create a Log Audit IAM role in each application AWS account with permissions to view CloudWatch Logs, configure an AWS Lambda function to assume the Log Audit role, and perform an hourly export of CloudWatch Logs data to an Amazon S3 bucket in the logging AWS account.
    - [ ] B. Configure CloudWatch Logs streams in each application AWS account to forward events to CloudWatch Logs in the logging AWS account. In the logging AWS account, subscribe an Amazon Kinesis Data Firehose stream to Amazon CloudWatch Events, and use the stream to persist log data in Amazon S3.
    - [ ] C. Create Amazon Kinesis Data Streams in the logging account, subscribe the stream to CloudWatch Logs streams in each application AWS account, configure an Amazon Kinesis Data Firehose delivery stream with the Data Streams as its source, and persist the log data in an Amazon S3 bucket inside the logging AWS account.
    - [ ] D. Configure CloudWatch Logs agents to publish data to an Amazon Kinesis Data Firehose stream in the logging AWS account, use an AWS Lambda function to read messages from the stream and push messages to Data Firehose, and persist the data in Amazon S3.

    <details>
       <summary>Answer</summary>

       - [ ] A. 每个小时执行一此不是接近实时的了
       - [ ] B. CloudWatch不支持流式记录
       - [x] C. 正确 -> [ref](https://aws.amazon.com/cn/blogs/architecture/stream-amazon-cloudwatch-logs-to-a-centralized-account-for-audit-and-analysis/)
       - [ ] D. CloudWatch代理不推记录，需要通过别得方式拉取记录

       ![centrul-logging](img/centrul-logging.png)

    </details>

43. A company has a serverless application comprised of Amazon CloudFront, Amazon API Gateway, and AWS Lambda functions. The current deployment process of the application is to create a new version number of the Lambda function and run an AWS CLI script to update. If the new function version has errors, another CLI script reverts by deploying the previous working version of the function. The company would like to decrease the time to deploy new versions of the application logic provided by the Lambda functions, and reduce the time to detect and revert when errors are identified. How can this be accomplished?
    - [ ] A. Create and deploy nested AWS CloudFormation stacks with the parent stack consisting of the AWS CloudFront distribution and API Gateway, and the child stack containing the Lambda function. For changes to Lambda, create an AWS CloudFormation change set and deploy; if errors are triggered, revert the AWS CloudFormation change set to the previous version.
    - [ ] B. Use AWS SAM and built-in AWS CodeDeploy to deploy the new Lambda version, gradually shift traffic to the new version, and use pre-traffic and post-traffic test functions to verify. Rollback if Amazon CloudWatch alarms are triggered.
    - [ ] C. Refactor the AWS CLI scripts into a single script that deploys the new Lambda version. When deployment is completed, the script tests execute. If errors are detected, revert to the previous Lambda version.
    - [ ] D. Create and deploy an AWS CloudFormation stack that consists of a new API Gateway endpoint that references the new Lambda version. Change the CloudFront origin to the new API Gateway endpoint, monitor errors and if detected, change the AWS CloudFront origin to the previous API Gateway endpoint.

    <details>
       <summary>Answer</summary>

       - [ ] A. API Gateway也需要指向新的Lambda
       - [x] B. 正确 -> [ref](https://docs.aws.amazon.com/zh_cn/serverless-application-model/latest/developerguide/automating-updates-to-serverless-apps.html)
       - [ ] C. 同A
       - [ ] D. 不是自动的

    </details>

44. A company is running a .NET three-tier web application on AWS. The team currently uses XL storage optimized instances to store serve the website's image and video files on local instance storage. The company has encountered issues with data loss from replication and instance failures. The Solutions Architect has been asked to redesign this application to improve its reliability while keeping costs low. Which solution will meet these requirements?
    - [ ] A. Set up a new Amazon EFS share, move all image and video files to this share, and then attach this new drive as a mount point to all existing servers. Create an Elastic Load Balancer with Auto Scaling general purpose instances. Enable Amazon CloudFront to the Elastic Load Balancer. Enable Cost Explorer and use AWS Trusted advisor checks to continue monitoring the environment for future savings.
    - [ ] B. Implement Auto Scaling with general purpose instance types and an Elastic Load Balancer. Enable an Amazon CloudFront distribution to Amazon S3 and move images and video files to Amazon S3. Reserve general purpose instances to meet base performance requirements. Use Cost Explorer and AWS Trusted Advisor checks to continue monitoring the environment for future savings.
    - [ ] C. Move the entire website to Amazon S3 using the S3 website hosting feature. Remove all the web servers and have Amazon S3 communicate directly with the application servers in Amazon VPC.
    - [ ] D. Use AWS Elastic Beanstalk to deploy the .NET application. Move all images and video files to Amazon EFS. Create an Amazon CloudFront distribution that points to the EFS share. Reserve the m4.4xl instances needed to meet base performance requirements.

    <details>
       <summary>Answer</summary>

       - [ ] A. EFS比S3贵
       - [x] B. 正确
       - [ ] C. 服务器可以访问S3但是S3无法直接访问服务器
       - [ ] D. CloudFront无法直接指向NFS

    </details>

45. A company has developed a web application that runs on Amazon EC2 instances in one AWS Region. The company has taken on new business in other countries and must deploy its application into other to meet low-latency requirements for its users. The regions can be segregated, and an application running in one region does not need to communicate with instances in other regions. How should the company's Solutions Architect automate the deployment of the application so that it can be MOST efficiently deployed into multiple regions?
    - [ ] A. Write a bash script that uses the AWS CLI to query the current state in one region and output a JSON representation. Pass the JSON representation to the AWS CLI, specifying the --region parameter to deploy the application to other regions.
    - [ ] B. Write a bash script that uses the AWS CLI to query the current state in one region and output an AWS CloudFormation template. Create a CloudFormation stack from the template by using the AWS CLI, specifying the --region parameter to deploy the application to other regions.
    - [ ] C. Write a CloudFormation template describing the application's infrastructure in the resources section. Create a CloudFormation stack from the template by using the AWS CLI, specify multiple regions using the --regions to deploy the application.
    - [ ] D. Write a CloudFormation template describing the application's infrastructure in the Resources section. Use a CloudFormation stack set from an administrator account to launch stack instances that deploy the application to other regions.

    <details>
       <summary>Answer</summary>

       简单题，跨区域部署要使用堆栈集，答案D

    </details>

46. A media company has a 30-TB repository of digital news videos. These videos are stored on tape in an on-premises tape libraw and referenced by a Media Asset Management (MAM) system. The company wants to enrich the metadata for these videos in an automated fashion and put them into a searchable catalog by using a MAM feature. The company must be able to search based on information in the video, such as objects, scenery items, or people's faces. A catalog is available that contains faces of people who have appeared in the videos that include an image of each person. The company would like to migrate these videos to AWS. The company has a high-speed AWS Direct Connect connection with AWS and would like to move the MAM solution video content directly from its current file system. How can these requirements be met by using the LEAST amount of ongoing management overhead and causing MINIMAL dismption to the existing system?
    - [ ] A. Set up an AWS Storage Gateway, file gateway appliance on premises. Use the MAM solution to extract the videos from the current archive and push them into the file gateway. Use the catalog of faces to build a collection in Amazon Rekognition. Build an AWS Lambda function that invokes the Rekognition Javascript SDK to have Rekognition pull the video from the Amazon S3 files backing the file gateway, retrieve the required metadata, and push the metadata into the MAM solution.
    - [ ] B. Set up an AWS Storage Gateway, tape gateway appliance on-premises. Use the MAM solution to extract the videos from the current archive and push them into the tape gateway. Use the catalog of faces to build a collection in Amazon Rekognition. Build an AWS Lambda function that invokes the Rekognition Javascript SDK to have Amazon Rekognition process the video in the tape gateway, retrieve the required metadata, and push the metadata into the MAM solution.
    - [ ] C. Configure a video ingestion stream by using Amazon Kinesis Video Streams. Use the catalog of faces to build a collection in Amazon Rekognition. Stream the videos from the MAM solution into Kinesis Video Streams. Configure Amazon Rekognition to process the streamed videos. Then, use a stream consumer to retrieve the required metadata, and push the metadata into the MAM solution. Configure the stream to store the videos in Amazon S3.
    - [ ] D. Set up an Amazon EC2 instance that runs the OpenCV libraries. Copy the video, images, and face catalog from the on-premisesin into an Amazon EBS volume mounted on this EC2 instance. Process the video to retriev the required metadata, and push the metadata into the MAM solution while also copying the video files to an Amazon S3 bucket.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. 磁带需要在某处恢复后才可以被访问
       - [ ] C. 服务器可以访问S3但是S3无法直接访问服务器
       - [ ] D. 无法将流直接存储与S3

    </details>

47. A company is planning the migration of several lab environments used for software testing. An assortment of custom tooling is used to manage the test runs for each lab. The labs use immutable infrastructure for the software test runs, and the results are stored in a highly available SQL database cluster. Although completely rewriting the custom tooling is out of scope for the migration project, the company would like to optimize workloads during the migration. Which application migration strategy meets this requirement?
    - [ ] A. Re-host
    - [ ] B. Re-platform
    - [ ] C. Re-factor/re-architect
    - [ ] D. Retire

    <details>
       <summary>Answer</summary>

       概念题，答案B -> [ref](https://aws.amazon.com/jp/builders-flash/202007/awsgeek-migration-steps/?awsf.filter-name=*all)

    </details>

48. A company is implementing a multi-account strategy; however, the Management team has expressed concerns that services like DNS may become overly complex. The company needs a solution that allows private DNS to be shared among virtual private clouds (VPCs) in different accounts. The company will have approximately 50 accounts in total. What solution would create the LEAST complex DNS architecture and ensure that each VPC can resolve all AWS resources?
    - [ ] A. Create a shared services VPC in a central account, and create a VPC peering connection from the shared services VPC to each of the VPCs in the other accounts. Within Amazon Route 53, create a privately hosted zone in the shared services VPC and resource record sets for the domain and subdomains. Programmatically associate other VPCs with the hosted zone.
    - [ ] B. Create a VPC peering connection among the VPCs in all accounts. Set the VPC attributes enableDnsHostnames and enableDnsSupport to "true" for each VPC. Create an Amazon Route 53 private zone for each VPC. Create resource record sets for the domain and subdomains. Programmatically associate the hosted zones in each VPC with the other VPCs.
    - [ ] C. Create a shared services VPC in a central account. Create a VPC peering connection from the VPCs in other accounts to the shared services VPC. Create an Amazon Route 53 privately hosted zone in the shared services VPC with resource record sets for the domain and subdomains. Allow UDP and TCP port 53 over the VPC peering connections.
    - [ ] D. Set the VPC attributes enableDnsHostnames and enableDnsSupport to "false" in every VPC. Create an AWS Direct Connect connection with a private virtual interface. Allow UDP and TCP port 53 over the virtual interface. Use the on-premises DNS servers to resolve the IP addresses in each VPC on AWS.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. 为每一个VPC创建对等连接数量太大（N^2）
       - [ ] C. 没必要开53端口
       - [ ] D. 跟题目没啥关系，干扰项

    </details>

49. A company has released a new version of a website to target an audience in Asia and South America. The website's media assets are hosted on Amazon S3 and have an Amazon CloudFront distribution to improve end-user performance. However, users are having a poor login experience the authentication service is only available in the us-east-1 AWS Region. How can the Solutions Architect improve the login experience and maintain high security and performance with minimal management overhead?
    - [ ] A. Replicate the setup in each new geography and use Amazon Route 53 geo-based routing to route traffic to the AWS Region closest to the users.
    - [ ] B. Use an Amazon Route 53 weighted routing policy to route traffic to the CloudFront distribution. Use CloudFront cached HTTP methods to improve the user login experience.
    - [ ] C. Use Amazon Lambda@Edge attached to the CloudFront viewer request trigger to authenticate and authorize users by maintaining a secure cookie token with a session expiry to improve the user experience in multiple geographies.
    - [ ] D. Replicate the setup in each geography and use Network Load Balancers to route traffic to the authentication service running in the closest region to users.

    <details>
       <summary>Answer</summary>

       - [ ] A. 不是最佳体验
       - [ ] B. 登录无法被缓存
       - [x] C. 正确 -> [ref](https://docs.aws.amazon.com/zh_cn/AmazonCloudFront/latest/DeveloperGuide/lambda-generating-http-responses-in-requests.html)
       - [ ] D. NLB不是干这个的

    </details>

50. A company has a standard three-tier architecture using two Availability Zones. During the company's off season, users report that the website is not working. The Solutions Architect finds that no changes have been made to the environment recently, the website is reachable, and it is possible to log in. However, when the Solutions Architect selects the "find a store near you" function, the maps provided on the site by a third-party RESTful API call do not work about 50% of the time after refreshing the page. The outbound API calls are made through Amazon EC2 NAT instances. What is the MOST likely reason for this failure and how can it be mitigated in the future?
    - [ ] A. The network ACL for one subnet is blocking outbound web traffic. Open the networkACL and prevent administration from making future changes through IAM.
    - [ ] B. The fault is in the third-party environment. Contact the third party that provides the maps and request a fix that will provide better uptime.
    - [ ] C. One NAT instance has become overloaded. Replace both EC2 NAT instances with a larger-sized instance and make sure to account for growth when making the new instance size.
    - [ ] D. One of the NAT instances failed. Recommend replacing the EC2 NAT instances with a NAT gateway.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

51. A company is migrating to the cloud. It wants to evaluate the configurations of virtual machines in its existing data center environment to ensure that it can size new Amazon EC2 instances accurately. The company wants to collect metrics, such as CPU, memory, and disk utilization, and it needs an inventory of what processes are running on each instance. The company would also like to monitor network connections to map communications between servers. Which would enable the collection of this data MOST cost effectively?
    - [ ] A. Use AWS Application Discovery Service and deploy the data collection agent to each virtual machine in the data center.
    - [ ] B. Configure the Amazon CloudWatch agent on all servers within the local environment and publish metrics to Amazon CloudWatch Logs.
    - [ ] C. Use AWS Application Discovery Service and enable agentless discovery in the existing virtualization environment.
    - [ ] D. Enable AWS Application Discovery Service in the AWS Management Console and configure the corporate firewall to allow scans over a VPN.
  
    <details>
       <summary>Answer</summary>

       概念题，CloudWatch无法监控网络流，答案A

    </details>

52. A company will several AWS accounts is using AWS Organizations and service control policies (SCPs). An Administrator created the following SCP and has attached it to an organizational unit (OU) that contains AWS account 1111-1111-1111. Developers working in account 1111-1111-1111 complain that they cannot create Amazon S3 buckets. How should the Administrator address this problem?

    ```json
    {
        "Version": 2012-10-27,
        "Statement": [
            {
                "Sid": "AllowsAllActions",
                "Effect": "Allow",
                "Action": "*",
                "Resource" : "*"
            },
            {
                "Sid": "DenyCloudTrail",
                "Effect": "Deny",
                "Action": "cloudtrail",
                "Resource" : "*"
            }
        ]
    }
    ```

    - [ ] A. Add s3 :CreateBucket with "Allow" effect to the SCP.
    - [ ] B. Remove the account from the OU and attach the SCP directly to account 1111-1111-1111.
    - [ ] C. Instruct the Developers to add Amazon S3 permissions to their IAM entities.
    - [ ] D. Remove the SCP from account 1111-1111-1111.
  
    <details>
       <summary>Answer</summary>

       管理员仍然必须附加基于身份或基于资源的策略分配给 IAM 用户或角色，或者您账户中的资源，以实际授予权限。答案C -> [ref](https://docs.aws.amazon.com/zh_cn/organizations/latest/userguide/orgs_manage_policies_scps.html)

    </details>

53. A company that provides wireless services need a solution to store and analyze log files about user activities. Currently, log files are delivered daily to Amazon Linux on Amazon EC2 instance. A batch script is run once a day to aggregate data used for analysis by a third-party tool. The data pushed to the third-party tool is used to generate a visualization for end users. The batch script is cumbersome to maintain, and it takes several hours to deliver the ever-increasing data volumes to the third-party tool. The company wants to lower costs and is open to considering a new tool that minimizes development effort and lowers administrative overhead. The company wants to build a more agile solution that can store and perform the analysis in near-real time, with minimal overhead. The solution needs to be cost effective and scalable to meet the company's end-user base growth. Which solution meets the company's requirements?
    - [ ] A. Develop a Python script to failure the data from Amazon EC2 in real time and store the data in Amazon S3. Use a copy command to copy data from Amazon S3 to Amazon Redshift. Connect a business intelligence tool running on Amazon EC2 to Amazon Redshift and create the visualizations.
    - [ ] B. Use an Amazon Kinesis agent running on an EC2 instance in an Auto Scaling group to collect and send the data to an Amazon Kinesis Data Firehose delivery stream. The Kinesis Data Firehose delivery stream will deliver the data directly to Amazon ES. Use Kibana to visualize the data.
    - [ ] C. Use an in-memory caching application running on an Amazon EBS-optimized EC2 instance to capture the log data in near real-time. Install an Amazon ES cluster on the same EC2 instance to store the log files as they are delivered to Amazon EC2 in near real-time. Install a Kibana plugin to create the visualizations.
    - [ ] D. Use an Amazon Kinesis agent running on an EC2 instance to collect and send the data to an Amazon Kinesis Data Firehose delivery stream. The Kinesis Data Firehose delivery stream will deliver the data to Amazon S3. Use an AWS Lambda function to deliver the data from Amazon S3 to Amazon ES Use Kibana to visualize the data.

    <details>
       <summary>Answer</summary>

       - [ ] A. Python脚本变成了维护的一部分不符合客户要求
       - [x] B. 正确
       - [ ] C. io1很贵
       - [ ] D. Amazon Kinesis Data Firehose直接可以把数据发送到Amazon ES -> [ref](https://docs.aws.amazon.com/zh_cn/elasticsearch-service/latest/developerguide/es-aws-integrations.html)

    </details>

54. A company wants to move a web application to AWS. The application stores session information locally on each web server, which will make auto scaling difficult. As part of the migration, the application will be rewritten to decouple the session data from the web servers. The company requires low latency, scalability, and availability. Which service will meet the requirements for storing the session information in the MOST cost effective way?
    - [ ] A. Amazon ElastiCache with the Memcached engine
    - [ ] B. Amazon S3
    - [ ] C. Amazon RDS MySQL
    - [ ] D. Amazon ElastiCache with the Redis engine

    <details>
       <summary>Answer</summary>

       ElastiCache用于缓存会话信息，排除BC，Memcached是单AZ不满足高HA，答案D

    </details>

55. A company has an Amazon EC2 deployment that has the following architecture: -An application tier that contains 8 m4.xlarge instances -A Classic Load Balancer -Amazon S3 as a persistent data store. After one of the EC2 instances fails, users report very slow processing of their requests. A Solutions Architect must recommend design changes to maximize system reliability. The solution must minimize costs. What should the Solution Architect recommend?
    - [ ] A. Migrate the existing EC2 instances to a serverless deployment using AWS Lambda functions.
    - [ ] B. Change the Classic Load Balancer to an Application Load Balancer.
    - [ ] C. Replace the application tier with m4.large instances in an Auto Scaling group.
    - [ ] D. Replace the application tier with 4 m4.2xlarge instances.

    <details>
       <summary>Answer</summary>

       简单题，因为没有加Auto Scaling，答案C

    </details>

56. An on-premises application will be migrated to the cloud. The application consists of a single Elasticsearch virtual machine with data source feeds from local systems that will not be migrated, and a Java web application on Apache Tomcat running on three virtual machines. The Elasticsearch server currently uses 1TB of storage out of 16 TB available storage, and the web application is updated eve1Y 4 months. Multiple users access the web application from the Internet. There is a 10Gbit AWS Direct Connect connection established, and the application can be migrated over a schedules 48-hour change window. Which strategy will have the LEAST impact on the Operations staff after the migration?
    - [ ] A. Create an Elasticsearch server on Amazon EC2 right-sized with 2 TB of Amazon EBS and a public AWS Elastic Beanstalk environment for the web application. Pause the data sources, export the Elasticsearch index from on premises, and import into the EC2 Elasticsearch server. Move data source feeds to the new Elasticsearch server and move users to the web application.
    - [ ] B. Create an Amazon ES cluster for Elasticsearch and a public AWS Elastic Beanstalk environment for the web application. Use AWS DMS to replicate Elasticsearch data. When replication has finished, move data source feeds to the new Amazon ES cluster endpoint and move users to the new web application.
    - [ ] C. Use the AWS SMS to replicate the virtual machines into AWS. When the migration is complete, pause the data source feeds and start the migrated Elasticsearch and web application instances. Place the web application instances behind a public Elastic Load Balancer. Move the data source feeds to the new Elasticsearch server and move users to the new web Application Load Balancer.
    - [ ] D. Create an Amazon ES cluster for Elasticsearch and a public AWS Elastic Beanstalk environment for the web application. Pause the data source feeds, export the Elasticsearch index from on premises, and import into the Amazon ES cluster. Move the data source feeds to the new Amazon ES cluster endpoint and move users to the new web application.

    <details>
       <summary>Answer</summary>

       - [ ] A. 非高可用，不满足AWS最佳实践
       - [ ] B. DMS不干这事儿
       - [ ] C. 同A
       - [x] D. 正确

    </details>

57. A company's application is increasingly popular and experiencing latency because of high volume reads on the database server. The service has the following properties: -A highly available REST API hosted in one region using Application Load Balancer (ALB) with auto scaling. -A MySQL database hosted on an Amazon EC2 instance in a single Availability Zone. -The company wants to reduce latency, increase in-region database read performance, and have multi-region disaster recovery capabilities that can perform a live recover automatically without any data or performance loss (HADR). Which deployment strategy will meet these requirements?
    - [ ] A. Use AWS CloudFormation StackSets to deploy the API layer in two regions. Migrate the database to an Amazon Aurora with MySQL database cluster with multiple read replicas in one region and a read replica in a different region than the source database cluster. Use Amazon Route 53 health checks to trigger a DNS failover to the standby region if the health checks to the primary load balancer fail. In the event of Route 53 failover, promote the cross-region database replica to be the master and build out new read replicas in the standby region.
    - [ ] B. Use Amazon ElastiCache for Redis Multi-AZ with an automatic failover to cache the database read queries. Use AWS OpsWorks to deploy the API layer, cache layer, and existing database layer in two regions. In the event of failure, use Amazon Route 53 health checks on the database to trigger a DNS failover to the standby region if the health checks in the primary region fail. Back up the MySQL database frequently, and in the event of a failure in an active region, copy the backup to the standby region and restore the standby database.
    - [ ] C. Use AWS CloudFormation StackSets to deploy the API layer in two regions. Add the database to an Auto Scaling group. Add a read replica to the database in the second region. Use Amazon Route 53 health checks in the primaregion fail. Promote the cross-region database replica to be the master and build out new read replicas in the standby region.
    - [ ] D. Use Amazon ElastiCache for Redis Multi-AZ with an automatic failover to cache the database read queries. Use AWS OpsWorks to deploy the API layer, cache layer, and existing database layer in two regions. Use Amazon Route 53 health checks on the ALB to trigger a DNS failover to the standby region if the health checks in the primary region fail. Back up the MySQL database frequently, and in the event of a failure in an active region, copy the backup to the standby region and restore the standby database.

    <details>
       <summary>Answer</summary>

       别看他题目长，就是一简单题，答案A

    </details>

58. A company runs a three-tier application in AWS. Users report that the application performance can vary greatly depending on the time of day and functionality being accessed. The application includes the following components: -Eight t2.large front-end web servers that serve static content and proxy dynamic content from the application tier. -Four t2.large application servers. -One db.m4.large Amazon RDS MySQL Multi-AZ DB instance. Operations has determined that the web and application tiers are network constrained. Which of the following should cost effective improve application performance? (Choose two.)
    - [ ] A. Replace web and app tiers with t2.xlarge instances
    - [ ] B. Use AWS Auto Scaling and m4.large instances for the web and application tiers
    - [ ] C. Convert the MySQL RDS instance to a self-managed MySQL cluster on Amazon EC2
    - [ ] D. Create an Amazon CloudFront distribution to cache content
    - [ ] E. Increase the size of the Amazon RDS instance to db.m4.xlarge

    <details>
       <summary>Answer</summary>

       简单题，首先题目中没有使用Auto Scaling group，所以先选B，而且m4和t2的网络性能是一样的，CloudFront是标准操作，答案BD

    </details>

59. An online retailer needs to regularly process large product catalogs, which are handled in batches. These are sent out to be processed by people using the Amazon Mechanical Turk service, but the retailer has asked its Solutions Architect to design a workflow orchestration system that allows it to handle multiple concurrent Mechanical Turk operations, deal with the result assessment process, and reprocess failures. Which of the following options gives the retailer the ability to interrogate the state of every workflow with the LEAST amount of implementation effort?
    - [ ] A. Trigger Amazon CloudWatch alarms based upon message visibility in multiple Amazon SQS queues (one queue per workflow stage) and send messages via Amazon SNS to trigger AWS Lambda functions to process the next step. Use Amazon ES and Kibana to visualize Lambda processing logs to see the workflow states.
    - [ ] B. Hold workflow information in an Amazon RDS instance with AWS Lambda functions polling RDS for status changes. Worker Lambda functions then process the next workflow steps. Amazon QuickSight will visualize workflow states directly out of Amazon RDS.
    - [ ] C. Build the workflow in AWS Step Functions, using it to orchestrate multiple concurrent workflows. The status of each workflow can be visualized in the AWS Management Console, and historical data can be written to Amazon S3 and visualized using Amazon QuickSight.
    - [ ] D. Use Amazon SWF to create a workflow that handles a single batch of catalog records with multiple worker tasks to extract the data, transform it, and send it through Mechanical Turk. Use Amazon ES and Kibana to visualize AWS Lambda processing logs to see the workflow states.

    <details>
       <summary>Answer</summary>

       这题是固定套路，答案D -> [ref](https://aws.amazon.com/cn/swf/?nc1=h_ls)

    </details>

60. An organization has two Amazon EC2 instances: -The first is running an ordering application and an inventory application. -The second is running a queuing system. During certain times of the year, several thousand orders are placed per second. Some orders were lost when the queuing system was down. Also, the organization's inventory application has the incorrect quantity of products because some orders were processed twice. What should be done to ensure that the applications can handle the increasing number of orders?
    - [ ] A. Put the ordering and inventory applications into their own AWS Lambda functions. Have the ordering application write the messages into an Amazon SQS FIFO queue.
    - [ ] B. Put the ordering and inventory applications into their own Amazon ECS containers and create an Auto Scaling group for each application. Then, deploy the message queuing server in multiple Availability Zones.
    - [ ] C. Put the ordering and inventory applications into their own Amazon EC2 instances, and create an Auto Scaling group for each application. Use Amazon SQS standard queues for the incoming orders, and implement idempotency in the inventory application.
    - [ ] D. Put the ordering and inventory applications into their own Amazon EC2 instances. Write the incoming orders to an Amazon Kinesis data stream Configure AWS Lambda to poll the stream and update the inventory application.

    <details>
       <summary>Answer</summary>

       - [ ] A. 比较脏，Lambda并发最大1000，几千条肯定搞不定
       - [ ] B. ECS里没有AutoScaling
       - [x] C. 正确
       - [ ] D. Amazon Kinesis data stream里依然会出现丢失或者重复的消息

    </details>

61. A company is migrating its on-premises build artifact server to an AWS solution. The current system consists of an Apache HTTP server that serves artifacts to clients on the local network, restricted by the perimeter firewall. The artifact consumers are largely built automation scripts that download artifacts via anonymous HTTP, which the company will be unable to modify within its migration timetable. The company decides to move the solution to Amazon S3 static website hosting. The artifact consumers will be migrated to Amazon EC2 instances located within both public and private subnets in a virtual private cloud (VPC). Which solution will permit the artifact consumers to download artifacts without modifying the existing automation scripts?
    - [ ] A. Create a NAT gateway within a public subnet of the VPC. Add a default route pointing to the NAT gateway into the route table associated with the subnets containing consumers. Configure the bucket policy to allow the s3:ListBucket and s3:GetObject actions using the condition IpAddress and the condition key aws:SourceIp matching the elastic IP address if the NAT gateway.
    - [ ] B. Create a VPC endpoint and add it to the route table associated with subnets containing consumers. Configure the bucket policy to allow s3:ListBucket and s3:GetObject actions using the condition and the condition key aws:sourceVpce matching the identification of the VPC StringEquals endpoint.
    - [ ] C. Create an IAM role and instance profile for Amazon EC2 and attach it to the instances that consume build artifacts. Configure the bucket policy to allow the s3:ListBucket and s3:GetObjects actions for the principal matching the IAM role created.
    - [ ] D. Create a VPC endpoint and add it to the route table associated with subnets containing consumers. Configure the bucket policy to allow s3:ListBucket and s3:GetObject actions using the condition and the condition key aws:Sourcelp matching the VPC CIDR block.

    <details>
       <summary>Answer</summary>

       - [ ] A. 方法可行但是要经过公共网络，不算最佳实践
       - [x] B. 正确
       - [ ] C. 私网中的实例无法直接访问S3
       - [ ] D. S3 VPC端点不能直接使用IP地址

    </details>

62. A group of research institutions and hospitals are in a partnership to study 2 PBS of genomic data. The institute that owns the data stores it in an Amazon S3 bucket and updates it regularly. The institute would like to give all of the organizations in the partnership read access to the data. All members of the partnership are extremely cost-conscious, and the institute that owns the account with the S3 bucket is concerned about covering the costs for requests and data transfers from Amazon S3. Which solution allows for secure data sharing without causing the institute that owns the bucket to assume all the costs for S3 requests and data transfers?
    - [ ] A. Ensure that all organizations in the partnership have AWS accounts. In the account with the S3 bucket, create a cross-account role for each account in the partnership that allows read access to the data. Have the organizations assume and use that read role when accessing the data.
    - [ ] B. Ensure that all organizations in the partnership have AWS accounts. Create a bucket policy on the bucket that owns the data. The policy should allow the accounts in the partnership read access to the bucket. Enable Requester Pays on the bucket. Have the organizations use their AWS credentials when accessing the data.
    - [ ] C. Ensure that all organizations in the partnership have AWS accounts. Configure buckets in each of the accounts with a bucket policy that allows the institute that owns the data the ability to write to the bucket. Periodically sync the data from the institute's account to the other organizations. Have the organizations use their AWS credentials when accessing the data using their accounts.
    - [ ] D. Ensure that all organizations in the partnership have AWS accounts. In the account with the S3 bucket, create a cross-account role for each account in the partnership that allows read access to the data. Enable Requester Pays on the bucket. Have the organizations assume and use that read role when accessing the data.

    <details>
       <summary>Answer</summary>

       简单题，答案B

    </details>

63. A company currently uses a single 1 Gbps AWS Direct Connect connection to establish connectivity between an AWS Region and its data center. The company has five Amazon VPCs, all of which are connected to the data center using the same Direct Connect connection. The Network team is worried about the single point of failure and is interested in improving the redundancy of the connections to AWS while keeping costs to a minimum. Which solution would improve the redundancy of the connection to AWS while meeting the cost requirements?
    - [ ] A. Provision another 1 Gbps Direct Connect connection and create new VIFs to each of the VPCs. Configure the VIFs in a load balancing fashion using BGP.
    - [ ] B. Set up VPN tunnels from the data center to each VPC. Terminate each VPN tunnel at the virtual private gateway (VGW) of the respective VPC and set up BGP for route management.
    - [ ] C. Set up a new point-to-point Multiprotocol Label Switching (MPLS) connection to the AWS Region that's being used. Configure BGP to use this new circuit as passive, so that no traffic flows through this unless the AWS Direct Connect fails.
    - [ ] D. Create a public VIF on the Direct Connect connection and set up a VPN tunnel which will terminate on the virtual private gateway (VGW) of the respective VPC using the public VIF. Use BGP to handle the failover to the VPN connection.

    <details>
       <summary>Answer</summary>

       - [ ] A. VIF不能当VGW用
       - [x] B. 正确
       - [ ] C. 这么设置流量还是走DX，并没有什么卵用
       - [ ] D. 没必要设置公共VIF因为不需要访问公共AWS资源

    </details>

64. A company currently uses Amazon EBS and Amazon RDS for storage purposes. The company intends to use a pilot light approach for disaster recovery in a different AWS Region. The company has an RTO of 6 hours and an RPO of 24 hours. Which solution would achieve the requirements with MINIMAL cost?
    - [ ] A. Use AWS Lambda to create daily EBS and RDS snapshots, and copy them to the disaster recovery region. Use Amazon Route 53 with active-passive failover configuration. Use Amazon EC2 in an Auto Scaling group with the capacity set to 0 in the disaster recovery region.
    - [ ] B. Use AWS Lambda to create daily EBS and RDS snapshots, and copy them to the disaster recovery region. Use Amazon Route 53 with active-active failover configuration. Use Amazon EC2 in an Auto Scaling group configured in the same way as in the primary region.
    - [ ] C. Use Amazon ECS to handle long-running tasks to create daily EBS and RDS snapshots, and copy to the disaster recovery region. Use Amazon Route 53 with active-passive failover configuration. Use Amazon EC2 in an Auto Scaling group with the capacity set to 0 in the disaster recovery region.
    - [ ] D. Use EBS and RDS cross-region snapshot copy capability to create snapshots in the disaster recovery region. Use Amazon Route 53 with active-active failover configuration. Use Amazon EC2 in an Auto Scaling group with the capacity set to 0 in the disaster recovery region.

    <details>
       <summary>Answer</summary>

       EBS和RDS都有跨区域快照，所以Lambda不是最佳实践，排除AB，这里不涉及到容器所以C不对，答案D

    </details>

65. A company needs to cost-effectively persist small data records (up to 1 KiB) for up to 30 days. The data is read rarely. When reading the data, a 5-minute delay is acceptable. Which of the following solutions achieve this goal? (Choose two.)
    - [ ] A. Use Amazon S3 to collect multiple records in one S3 object. Use a lifecycle configuration to move data to Amazon Glacier immediately after write. Use expedited retrievals when reading the data.
    - [ ] B. Write the records to Amazon Kinesis Data Firehose and configure Kinesis Data Firehose to deliver the data to Amazon S3 after 5 minutes. Set an expiration action at 30 days on the S3 bucket.
    - [ ] C. Use an AWS Lambda function invoked via Amazon API Gateway to collect data for 5 minutes. Write data to Amazon S3 just before the Lambda execution stops.
    - [ ] D. Write the records to Amazon DynamoDB configured with a Time To Live (TTL) of 30 days. Read data using the Getltem or BatchGetItem call.
    - [ ] E. Write the records to an Amazon ElastiCache for Redis. Configure the Redis append-only file (AOF) persistence logs to write to Amazon S3. Recover from the log if the ElastiCache instance has failed.

    <details>
       <summary>Answer</summary>

       - [ ] A. 进入冰川后延迟可能高达数小时，不满足需求
       - [x] B. 正确
       - [ ] C. 数据可能缺失
       - [x] D. 正确
       - [ ] E. 价格昂贵不太合适

    </details>

66. A Development team is deploying new APIs as serverless applications within a company. The team is currently using the AWS Management Console to provision Amazon API Gateway, AWS Lambda, and Amazon DynamoDB resources. A Solutions Architect has been tasked with automating the future deployments of these serverless APIs. How can this be accomplished?
    - [ ] A. Use AWS CloudFormation with a Lambda-backed custom resource to provision API Gateway. Use the `AWS::DynamoDB::Table` and resources to create the Amazon DynamoDB table and Lambda functions. Write a script to automate the deployment of the CloudFormation template.
    - [ ] B. Use the AWS Serverless Application Model to define the resources. Upload a YAML template and application files to the repository. Use AWS CodePipeline to connect to the repository and to create an action to build using AWS CodeBuild. Use the AWS CloudFormation deployment provider in CodePipeline to deploy the solution.
    - [ ] C. Use AWS CloudFormation to define the serverless application. Implement versioning on the Lambda functions and create aliases to point to the versions. When deploying, configure weights to implement shifting traffic to the newest version, and gradually update the weights as traffic moves over.
    - [ ] D. Commit the application to the AWS CodeCommit repository. Use AWS CodePipeline and connect to the CodeCommit repository. Use AWS CodeBuild to build and deploy the Lambda functions using AWS CodeDeploy. Specify the deployment preference type in CodeDeploy to gradually shift traffic over to the new version.

    <details>
       <summary>Answer</summary>

       简单题，答案B
    </details>

67. The company Security team queries that all data uploaded into an Amazon S3 bucket must be encrypted. The encryption keys must be highly available, and the company must be able to control access on a per-user basis, with different users having access to different encryption keys. Which of the following architectures will meet these requirements? (Choose two.)
    - [ ] A. Use Amazon S3 server-side encryption with Amazon S3-managed keys. Allow Amazon S3 to generate an AWS/S3 master key and use IAM to control access to the data keys that are generated.
    - [ ] B. Use Amazon S3 server-side encryption with AWS KMS-managed keys, create multiple customer master keys, and use key policies to control access to them.
    - [ ] C. Use Amazon S3 server-side encryption with customer-managed keys, and use AWS CloudHSM to manage the keys. Use CloudHSM client software to control access to the keys that are generated.
    - [ ] D. Use Amazon S3 server-side encryption with customer-managed keys, and use two AWS CloudHSM instances configured in high-availability mode to manage the keys. Use the Cloud HSM client software to control access to the keys that are generated.
    - [ ] E. Use Amazon S3 server-side encryption with customer-managed keys, and use two AWS CloudHSM instances configured in high-availability mode to manage the keys. Use IAM to control access to the keys that are generated in CloudHSM.

    <details>
       <summary>Answer</summary>

       - [ ] A. 无法管理S3生成的key
       - [x] B. 正确
       - [ ] C. 一个CloudHSM的可用性满足不了需求
       - [x] D. 正确
       - [ ] E. CloudHSM不能够直接和AWS资源进行连接，必须使用客户端

    </details>

68. A company runs a public-facing application that uses a Java-based web service via a RESTful API. It is hosted on Apache Tomcat on a single server in a data center that runs consistently at 30% CPU utilization. Use of the API is expected to increase by 10 times with a new product launch. The business wants to migrate the application to AWS with no disruption and needs it to scale to meet demand. The company has already decided to use Amazon Route 53 and CNAME records to redirect traffic. How can these requirements be met with the LEAST amount of effort?
    - [ ] A. Use AWS Elastic Beanstalk to deploy the Java web service and enable Auto Scaling. Then switch the application to use the new web service.
    - [ ] B. Lift and shift the Apache server to the cloud using AWS SMS. Then switch the application to direct web service traffic to the new instance.
    - [ ] C. Create a Docker image and migrate the image to Amazon ECS. Then change the application to direct web service queries to the ECS container.
    - [ ] D. Modify the application to call the web service via Amazon API Gateway. Then create a new AWS Lambda Java function to run the Java web service. After testing, change API Gateway to use the Lambda function.

    <details>
       <summary>Answer</summary>

       简单题，答案A

    </details>

69. A company is using AWS for production and development workloads. Each business unit has its own AWS account for production, and a separate AWS account to develop and deploy its applications. The Information Security department has introduced new security policies that limit access for terminating certain Amazon EC2 instances in all accounts to a small group of individuals from the Security team. How can the Solutions Architect meet these requirements?
    - [ ] A. Create a new IAM policy that allows access to those EC2 instances only for the Security team. Apply this policy to the AWS Organizations master account.
    - [ ] B. Create a new tag-based IAM policy that allows access to these EC2 instances only for the Security team. Tag the instances appropriately and apply this policy in each account.
    - [ ] C. Create an organizational unit under AWS Organizations. Move all the accounts into this organizational unit and use SCP to apply a whitelist policy to allow access to these EC2 instances for the Security team only.
    - [ ] D. Set up SAML federation for all accounts in AWS. Configure SAML so that it checks for the service API call before authenticating the user. Block SAML from authenticating API calls if anyone other than the Security team accesses these instances.

    <details>
       <summary>Answer</summary>

       - [ ] A. 仅仅给Master账户权限是不够的，子权限没有被赋予权限
       - [x] B. 正确
       - [ ] C. 设置完SCP还必须设置实体IAM策略
       - [ ] D. SAML的应用场景是基于token的服务，而不适用于API

    </details>

70. A company is moving a business-critical, multi-tier application to AWS. The architecture consists of a desktop client application and server infrastructure. The server infrastructure resides in an on-premises data center that frequently fails to maintain the application uptime SLA of 99.95%. A Solutions Architect must re-architect the application to ensure that it can meet or exceed the SLA. The application contains a PostgreSQL database running on a single virtual machine. The business logic and presentation layers are load balanced between multiple virtual machines. Remote users complain about slow load times while using this latency-sensitive application. Which of the following will meet the availability requirements with little change to the application while improving user experience and minimizing costs?
    - [ ] A. Migrate the database to a PostgreSQL database in Amazon EC2. Host the application and presentation layers in automatically scaled Amazon ECS containers behind an Application Load Balancer. Allocate an Amazon WorkSpaces for each end user to impresser experience.
    - [ ] B. Migrate the database to an Amazon RDS Aurora PostgreSQL configuration. Host the application and presentation layers in an Auto Scaling configuration on Amazon EC2 instances behind an Application Load Balancer. Use Amazon AppStream 2.0 to improve the user experience.
    - [ ] C. Migrate the database to an Amazon RDS PostgreSQL Multi-AZ configuration. Host the application and presentation layers in automatically scaled AWS Fargate containers behind a Network Load Balancer. Use Amazon ElastiCache to improve the user experience.
    - [ ] D. Migrate the database to an Amazon Redshift cluster with at least two nodes. Combine and host the application and presentation layers in automatically scaled Amazon ECS containers behind an Application Load Balancer. Use Amazon CloudFront to improve the user experience.

    <details>
       <summary>Answer</summary>

       简单题，答案B

    </details>

71. An advisory firm is creating a secure data analytics solution for its regulated financial services users. Users will upload their raw data to an Amazon S3 bucket, where they have `PutObject` permissions only. Data will be analyzed by applications running on an Amazon EMR cluster launched in a VPC The firm requires that the environment be isolated from the internet. All data at rest must be encrypted using keys controlled by the firm. Which combination of actions should the Solutions Architect take to meet the user's security requirements? (Choose two.)
    - [ ] A. Launch the Amazon EMR cluster in a private subnet configured to use an AWS KMS CMK for at-rest encryption. Configure a gateway VPC endpoint for Amazon S3 and an interface VPC endpoint for AWS KMS.
    - [ ] B. Launch the Amazon EMR cluster in a private subnet configured to use an AWS KMS CMK for at-rest encryption. Configure a gateway VPC endpoint for Amazon S3 and a NAT gateway to access AWS KNIS.
    - [ ] C. Launch the Amazon EMR cluster in a private subnet configured to use an AWS CloudHSM appliance for at-rest encryption. Configure a gateway VPC endpoint for Amazon S3 and an interface VPC endpoint for CloudHSM.
    - [ ] D. Configure the S3 endpoint policies to permit access to the necessary data buckets only.
    - [ ] E. Configure the S3 bucket policies to permit access using an aws:sourceVpce condition to match the S3 endpoint ID.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. 使用NAT的话会流量会走公共网络，不符合需求
       - [ ] C. CloudHSM不能连接AWS的服务，必修使用客户端
       - [ ] D. 还需要配置桶策略而不是端点策略
       - [x] E. 正确

    </details>

72. A company is designing a new highly available web application on AWS. The application requires consistent and reliable connectivity from the application servers in AWS to a backend REST API hosted in the company's on-premises environment. The backend connection between AWS and on-premises will be routed over an AWS Direct Connect connection through a private virtual interface. Amazon Route 53 will be used to manage private DNS records for the application to resolve the IP address on the backend REST API. Which design would provide a reliable connection to the backend API?
    - [ ] A. Implement at least two backend endpoints for the backend REST API, and use Route 53 health checks to monitor the availability of each backend endpoint and perform DNS-level failover.
    - [ ] B. Install a second Direct Connect connection from a different network carrier and attach it to the same virtual private gateway as the first Direct Connect connection.
    - [ ] C. Install a second cross connect for the same Direct Connect connection from the same network carrier, and join both connections to the same link aggregation group (LAG) on the same private virtual interface.
    - [ ] D. Create an IPSec VPN connection routed over the public internet from the on-premises data center to AWS and attach it to the same virtual private gateway as the Direct Connect connection.

    <details>
       <summary>Answer</summary>

       B最简单粗暴

    </details>

73. A company has a data center that must be migrated to AWS as quickly as possible. The data center has a 500 Mbps AWS Direct Connect link and a separate, fully available 1 Gbps ISP connection. A Solutions Architect must transfer 20 TB of data from the data center to an Amazon S3 bucket. What is the FASTEST way transfer the data?
    - [ ] A. Upload the data to the S3 bucket using the existing DX link.
    - [ ] B. Send the data to AWS using the AWS Import/Export service.
    - [ ] C. Upload the data using an 80 TB AWS Snowball device.
    - [ ] D. Upload the data to the S3 bucket using S3 Transfer Acceleration.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

74. A bank is designing an online customer sewice portal where customers can chat with customer service agents. The portal is required to maintain a 15-minute RPO or RTO in case of a regional disaster. Banking regulations require that all customer sewice chat transcripts must be preserved on durable storage for at least 7 years, chat conversations must be encrypted in-flight, and transcripts must be encrypted at rest. The Data Lost Prevention team requires that data at rest must be encrypted using a key that the team controls, rotates, and revokes. Which design meets these requirements?
    - [ ] A. The chat application logs each chat message into Amazon CloudWatch Logs. A scheduled AWS Lambda function invokes a CloudWatch Logs. CreateExportTask every 5 minutes to export chat transcripts to Amazon S3. The S3 bucket is configured for cross-region replication to the backup region. Separate AWS KMS keys are specified for the CloudWatch Logs group and the S3 bucket.
    - [ ] B. The chat application logs each chat message into two different Amazon CloudWatch Logs groups in two different regions, with the same AWS KMS key applied. Both CloudWatch Logs groups are configured to export logs into an Amazon Glacier vault with a 7-year vault lock policy with a KMS key specified.
    - [ ] C. The chat application logs each chat message into Amazon CloudWatch Logs. A subscription filter on the CloudWatch Logs group feeds into an Amazon Kinesis Data Firehose which streams the chat messages into an Amazon S3 bucket in the backup region. Separate AWS KMS keys are specified for the CloudWatch Logs group and the Kinesis Data Firehose.
    - [ ] D. The chat application logs each chat message into Amazon CloudWatch Logs. The CloudWatch Logs group is configured to export logs into an Amazon Glacier vault with a 7-year vault lock policy. Glacier cross-region replication mirrors chat archives to the backup region. Separate AWS KMS keys are specified for the CloudWatch Logs group and the Amazon Glacier vault.

    <details>
       <summary>Answer</summary>

       - [ ] A. S3Export不支持KMS对象
       - [ ] B. 使用冰川无法实时提取数据
       - [x] C. 正确
       - [ ] D. 同B

    </details>

75. A company currently runs a secure application on Amazon EC2 that takes files from on-premises locations through AWS Direct Connect, processes them, and uploads them to a single Amazon S3 bucket. The application uses HTTPS for encryption in transit to Amazon S3, and S3 server-side encryption to encrypt at rest. Which of the following changes should the Solutions Architect recommend making this solution more secure without impeding application's performance?
    - [ ] A. Add a NAT gateway. Update the security groups on the EC2 instance to allow access to and from the S3 IP range only. Configure an S3 bucket policy that allows communication from the NAT gateway's Elastic IP address only.
    - [ ] B. Add a VPC endpoint. Configure endpoint policies on the VPC endpoint to allow access to the required Amazon S3 buckets only. Implement an S3 bucket policy that allows communication from the VPC's source IP range only.
    - [ ] C. Add a NAT gateway. Update the security groups on the EC2 instance to allow access to and from the S3 IP range only. Configure an S3 bucket policy that allows communication from the source public IP address of the on-premises network only.
    - [ ] D. Add a VPC endpoint. Configure endpoint policies on the VPC endpoint to allow access to the required S3 buckets only. Implement an S3 bucket policy that allows communication from the VPC endpoint only.

    <details>
       <summary>Answer</summary>

       - [ ] A. NAT会使用公共网络，不满足需求
       - [ ] B. VPC端点IAM策略中不能使用源IP
       - [ ] C. 同A
       - [x] D. 正确

    </details>

76. As a part of building large applications in the AWS Cloud, the Solutions Architect is required to implement the perimeter security protection. Applications running on AWS have the following endpoints: -Application Load Balancer. -Amazon API Gateway regional endpoint. -Elastic IP address-based EC2 instances. -Amazon S3 hosted websites. -Classic Load Balancer. The Solutions Architect must design a solution to protect all the listed web front ends and provide the following security capabilities: -DDoS protection. -SQL injection protection. -IP address whitelist/blacklist. -HTTP flood protection. -Bad bot scraper protection. How should the Solutions Architect design the solution?
    - [ ] A. Deploy AWS WAF and AWS Shield Advanced on all web endpoints. Add AWS WAF rules to enforce the company’s requirements.
    - [ ] B. Deploy Amazon CloudFront in front of all the endpoints. The CloudFront distribution provides perimeter protection. Add AWS Lambda-based automation to provide additional security.
    - [ ] C. Deploy Amazon CloudFront in front of all the endpoints. Deploy AWS WAF and AWS Shield Advanced. Add AWS WAF rules to enforce the company’s requirements. Use AWS Lambda to automate and enhance the security posture.
    - [ ] D. Secure the endpoints by using network ACLs and security groups and adding rules to enforce the company’s requirements. Use AWS Lambda to automatically update the rules.

    <details>
       <summary>Answer</summary>

       CloudFront和AWS Shield Advanced可以有效防御DDoS攻击，WAF能够防御SQL注入和Bad Bots，答案C

    </details>

77. A company has more than 100 AWS accounts, with one VPC per account, that need outbound HTTPS connectivity to the internet. The current design contains one NAT gateway per Availability Zone (AZ) in each VPC. To reduce costs and obtain information about outbound traffic, management has asked for a new architecture for internet access. Which solution will meet the current needs, and continue to grow as new accounts are provisioned, while reducing costs?
    - [ ] A. Create a transit VPC across two AZs using a third-party routing appliance. Create a VPN connection to each VPC. Default route internet traffic to the transit VPC.
    - [ ] B. Create multiple hosted-private AWS Direct Connect VIFs, one per account, each with a Direct Connect gateway. Default route internet traffic back to an on-premises router to route to the internet.
    - [ ] C. Create a central VPC for outbound internet traffic. Use VPC peering to default route to a set of redundant NAT gateway in the central VPC.
    - [ ] D. Create a proxy fleet in a central VPC account. Create an AWS PrivateLink endpoint service in the central VPC. Use PrivateLink interface for internet connectivity through the proxy fleet.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. 太昂贵了，非最佳体验
       - [ ] C. 对等连接无法流向NAT
       - [ ] D. PrivcateLink不走外网

    </details>

78. A company runs an e-commerce platform with front-end and e-commerce tiers. Both tiers run on LAMP stacks with the front-end instances running behind a load balancing appliance that has a virtual offering on AWS. Currently, the Operations team uses SSH to log in to the instances to maintain patches and address other concerns. The platform has recently been the target of multiple attacks, including a DDoS attack. An SQL injection attack Several successful dictionary attacks on SSH accounts on the web servers. The company wants to improve the security of the e-commerce platform by migrating to AWS. The company's Solutions Architects have decided to use the following approach:Code review the existing application and fix any SQL injection issues. Migrate the web application to AWS and leverage the latest AWS Linux AMI to address initial security patching high availability and minimizing risk?
    - [ ] A. Enable SSH access to the Amazon EC2 instances using a security group that limits access to specific IPs. Migrate on-premises MySQL to Amazon RDS Multi-AZ. Install the third-party load balancer from the AWS Marketplace and migrate the existing rules to the load balancer's AWS instances. Enable AWS Shield Standard for DDoS protection.
    - [ ] B. Disable SSH access to the Amazon EC2 instances. Migrate on-premises MySQL to Amazon RDS Multi-AZ. Leverage an Elastic Load Balancer to spread the load and enable AWS Shield Advanced for protection. Add an Amazon CloudFront distribution in front of the website. Enable AWS WAF on the distribution to manage the rules.
    - [ ] C. Enable SSH access to the Amazon EC2 instances through a bastion host secured by limiting access to specific IP addresses. Migrate on-premises MySQL to a self-managed EC2 instance. Leverage an AWS Elastic Load Balancer to spread the load and enable AWS Shield Standard for DDoS protection. Add an Amazon CloudFront distribution in front of the website.
    - [ ] D. Disable SSH access to the EC2 instances. Migrate on-premises MySQL to Amazon RDS SingleAZ. Leverage an AWS Elastic Load Balancer to spread the load. Add an Amazon CloudFront distribution in front of the website. Enable AWS WAF on the distribution to manage the rules.

    <details>
       <summary>Answer</summary>

       简单提，答案B

    </details>

79. A company has a High-Performance Computing (HPC) cluster in its on-premises data center which runs thousands of jobs in parallel for one week every month, processing petabytes of images. The images are stored on a network file server, which is replicated to a disaster recovery site. The on-premises data center has reached capacity and has started to spread the jobs out over the course of month to better utilize the cluster, causing a delay in the job completion. The company has asked its Solutions Architect to design a cost-effective solution on AWS to scale beyond the current capacity of 5,000 cores and 10 petabytes of data. The solution must require the least amount of management overhead and maintain the current level of durability. Which solution will meet the company's requirements?
    - [ ] A. Create a container in the Amazon Elastic Container Registry with the executable file for the job. Use Amazon ECS with Spot Fleet in Auto Scaling groups. Store the raw data in Amazon EBS SCI volumes and write the output to Amazon S3.
    - [ ] B. Create an Amazon EMR cluster with a combination of On Demand and Reserved Instance Task Nodes that will use Spark to pull data from Amazon S3. Use Amazon DynamoDB to maintain a list of jobs that need to be processed by the Amazon EMR cluster.
    - [ ] C. Store the raw data in Amazon S3, and use AWS Batch with Managed Compute Environments to create Spot Fleets. Submit jobs to AWS Batch Job Queues to pull down objects from Amazon S3 onto Amazon EBS volumes for temporary storage to be processed, and then write the results back to Amazon S3.
    - [ ] D. Submit the list of jobs to be processed to an Amazon SQS to queue the jobs that need to be processed. Create a diversified cluster of Amazon EC2 worker instances using Spot Fleet that will automatically scale based on the queue depth. Use Amazon EFS to store all the data sharing it across all instances in the cluster.

    <details>
       <summary>Answer</summary>

       - [ ] A. EBS最大16TB，太难维护了
       - [ ] B. DynamoDB不用于存放job项目
       - [x] C. 正确
       - [ ] D. 应该使用S3

    </details>

80. A large company has many business units. Each business unit has multiple AWS accounts for different purposes. The CIO of the company sees that each business unit has data that would be useful to share with other parts of the company in total, there are about 10 PB of data that needs to be shared with users in 1,000 AWS accounts. The data is proprietary, so some of it should only be available to users with specific job types. Some of the data is used for throughput of intensive workloads, such as simulations. The number of accounts changes frequently because of new initiatives, acquisitions, and divestitures. A Solutions Architect has been asked to design a system that will allow for sharing data for use in AWS with all the employees in the company. Which approach will allow for secure data sharing in scalable way?
    - [ ] A. Store the data in a single Amazon S3 bucket. Create an IAM role for every combination of job type and business unit that allows to appropriate read/write access based on object prefixes in the S3 bucket. The roles should have trust policies that allow the business unit's AWS accounts to assume their roles. Use IAM in each business unit's AWS account to prevent them from assuming roles for a different job type. Users get credentials to access the data by using `AssumeRole` from their business unit's AWS account. Users can then use those credentials with an S3 client.
    - [ ] B. Store the data in a single Amazon S3 bucket. Write a bucket policy that uses conditions to grant read and write access where appropriate, based on each user's business unit and job type. Determine the business unit with the AWS account accessing the bucket and the job type with a prefix in the IAM user's name. Users can access data by using IAM credentials from their business unit's AWS account with an S3 client.
    - [ ] C. Store the data in a series of Amazon S3 buckets. Create an application running in Amazon EC2 that is integrated with the company's identity provider (IdP) that authenticates users and allows them to download or upload data through the application. The application uses the business unit and job type information in the IdP to control what users can upload and download through the application. The users can access the data through the application's AP
    - [ ] D. Store the data in a series of Amazon S3 buckets. Create an AWS STS token vending machine that is integrated with the company's identity provider (IdP). When a user logs in, have the token vending machine attach an IAM policy that assumes the role that limits the user's access and/or upload only the data the user is authorized to access. Users can get credentials by authenticating to the token vending machine's website or API and then use those credentials with an S3 client.

    <details>
       <summary>Answer</summary>

       这里D是最佳解决方案，其他的工作量都挺大的

    </details>

81. A company wants to migrate its website from an on-premises data center onto AWS. At the same time, it wants to migrate the website to a containerized microservice-based architecture to improve the availability and cost efficiency. The company's security policy states that privileges and network permissions must be configured according to best practice, using least privilege. A Solutions Architect must create a containerized architecture that meets the security requirements and has deployed the application to an Amazon ECS cluster. What steps are required after the deployment to meet the requirements? (Choose two.)
    - [ ] A. Create tasks using the bridge network mode.
    - [ ] B. Create tasks using the AWS VPC network mode.
    - [ ] C. Apply security groups to Amazon EC2 instances, and use IAM roles for EC2 instances to access other resources.
    - [ ] D. Apply security groups to the tasks, and pass IAM credentials into the container at launch time to access other resources.
    - [ ] E. Apply security groups to the tasks, and use IAM roles for tasks to access other resources.

    <details>
       <summary>Answer</summary>

       - [ ] A. 在桥接模式下，所有容器共享一个安全组，这样就不得不开所有端口，不符合最小特权
       - [x] B. 正确
       - [ ] C. 因为不选A，随意C是不必要的
       - [ ] D. 传输IAM认证信息是不安全的
       - [x] E. 正确

    </details>

82. A company is migrating its marketing website and content management system from an on-premises data center to AWS. The company wants the AWS application to be developed in a VPC with Amazon EC2 instances used for the web servers and an Amazon RDS instance for the database. The company has a runbook document that describes the installation process of the on-premises system. The company would like to base the AWS system on the processes referenced in the runbook document. The runbook document describes the installation and configuration of the operating systems, network settings, the website, and content management system software on the servers. After the migration is complete, the company wants to be able to make changes quickly to take advantage of other AWS features. How can the application and environment be deployed and automated in AWS, while allowing for future changes?
    - [ ] A. Update the runbook to describe how to create the VPC, the EC2 instances, and the RDS instance for the application by using the AWS Console. Make sure that the rest of the steps in the runbook are updated to reflect any changes that may come from the AWS migration.
    - [ ] B. Write a Python script that uses the AWS API to create the VPC, the EC2 instances, and the RDS instance for the application. Write shell scripts that implement the rest of the steps in the runbook. Have the Python script copy and run the shell scripts on the newly created instances to complete the installation.
    - [ ] C. Write an AWS CloudFormation template that creates the VPC, the EC2 instances, and the RDS instance for the application. Ensure that the rest of the steps in the runbook are updated to reflect any changes that may come from the AWS migration.
    - [ ] D. Write an AWS CloudFormation template that creates the VPC, the EC2 instances, and the RDS instance for the application. Include EC2 user data in the AWS CloudFormation template to install and configure the software.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

83. A company is adding a new approved external vendor that only supports IPv6 connectivity. The company's backend systems sit in the private subnet of an Amazon VPC. The company uses a NAT gateway to allow these systems to communicate with external vendors over IPv4. Company policy requires systems that communicate with external vendors use a security group that limits access to only approved external vendors. The virtual private cloud (VPC) uses the default network ACL The Systems Operator successfully assigns IPv6 addresses to each of the backend systems. The Systems Operator also updates the outbound security group to include the IPv6 CIDR of the external vendor (destination). The systems within the VPC can ping one another successfully over IPv6. However, these systems are unable to communicate with the external vendor. What changes are required to enable communication with the external vendor?
    - [ ] A. Create an IPv6 NAT instance. Add a route for destination 0.0.0.0/0 pointing to the NAT instance.
    - [ ] B. Enable IPv6 on the NAT gateway. Add a route for destination::/0 pointing to the NAT gateway.
    - [ ] C. Enable IPv6 on the internet gateway. Add a route for destination 0.0.0.0/0 pointing to the IGW.
    - [ ] D. Create an egress-only internet gateway. Add a route for destination::/0 pointing to the gateway.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

84. A finance company is running its business-critical application on current-generation Linux EC2 instances. The application includes a self-managed MySQL database performing heavy I/O operations. The application is working fine to handle a moderate amount of traffic during the month. However, it slows down during the final three days of each month due to month-end reporting, even though the company is using Elastic Load Balancers and Auto Scaling within its infrastructure to meet the increased demand. Which of the following actions would allow the database to handle the month-end load with the LEAST impact on performance?
    - [ ] A. Pre-warming Elastic Load Balancers, using a bigger instance type, changing all Amazon EBS volumes to GP2 volumes.
    - [ ] B. Performing a one-time migration of the database cluster to Amazon RDS, and creating several additional read replicas to handle the load during end of month.
    - [ ] C. Using Amazon CloudWatch with AWS Lambda to change the type, size, or IOPS of Amazon EBS volumes in the cluster based on a specific CloudWatch metric.
    - [ ] D. Replacing all existing Amazon EBS volumes with new PIOPS volumes that have the maximum available storage size and I/O per second by taking snapshots before the end of the month and reverting back afterwards.

    <details>
       <summary>Answer</summary>

       简单题，答案D

    </details>

85. A Solutions Architect is designing the storage layer for a data warehousing application. The data files are large, but they have statically placed metadata at the beginning of each file that describes the size and placement of the file's index. The data files are read in by a fleet of Amazon EC2 instances that store the index size, index location, and other category information about the data file in a database. That database is used by Amazon EMR to group files together for deeper analysis. What would be the MOST cost-effective, high availability storage solution for this workflow?
    - [ ] A. Store the data files in Amazon S3 and use Range GET for each file's metadata, then index the relevant data.
    - [ ] B. Store the data files in Amazon EFS mounted by the EC2 fleet and EMR nodes.
    - [ ] C. Store the data files on Amazon EBS volumes and allow the EC2 fleet and EMR to mount and unmount the volumes where they are needed.
    - [ ] D. Store the content of the data files in Amazon DynamoDB tables with the metadata, index, and data as their own keys.

    <details>
       <summary>Answer</summary>

       简单题，答案A

    </details>

86. A company uses an Amazon EMR cluster to process data once a day. The raw data comes from Amazon S3, and the resulting processed data is also stored in Amazon S3. The processing must complete within 4 hours; currently, it only takes 3 hours. However, the processing time is taking 5 to 10 minutes. longer each week due to an increasing volume of raw data. The team is also concerned about rising costs as the compute capacity increases. The EMR cluster is currently running on three m3.xlarge instances (one master and two core nodes). Which of the following solutions will reduce costs related to the increasing compute needs?
    - [ ] A. Add additional task nodes, but have the team purchase an all-upfront convertible Reserved Instance for each additional node to offset the costs.
    - [ ] B. Add additional task nodes, but use instance fleets with the master node in on-Demand mode and a mix of On-Demand and Spot Instances for the core and task nodes. Purchase a scheduled Reserved Instances for the master node.
    - [ ] C. Add additional task nodes, but use instance fleets with the master node in Spot mode and a mix of On-Demand and Spot Instances for the core and task nodes. Purchase enough scheduled Reserved Instances to offset the cost of running any On-Demand instances.
    - [ ] D. Add additional task nodes, but use instance fleets with the master node in On-Demand mode and a mix of On-Demand and Spot Instances for the core and task nodes. Purchase a standard all-upfront Reserved Instance for the master node.

    <details>
       <summary>Answer</summary>

       - [ ] A. 额外节点使用预购实例会很贵
       - [x] B. 正确
       - [ ] C. 主节点不能够使用spot实例因为它会被终断
       - [ ] D. 同A

    </details>

87. A company is building an AWS landing zone and has asked a Solutions Architect to design a multi-account access strategy that will allow hundreds of users to use corporate credentials to access the AWS Console. The company is running a Microsoft Active Directory and users will use an AWS Direct Connect connection to connect to AWS. The company also wants to be able to federate to third-party services and providers, including custom applications. Which solution meets the requirements by using the LEAST amount of management overhead?
    - [ ] A. Connect the Active Directory to AWS by using single sign-on and an Active Directory Federation Services (AD FS) with SAML 2.0, and then configure the identity Provider (IdP) system to use form-based authentication. Build the AD FS portal page with corporate branding and integrate third-party applications that support SAML 2.0 as required.
    - [ ] B. Create a two-way Forest trust relationship between the on-premises Active Directory and the AWS Directory Service. Set up AWS Single Sign-On with AWS Organizations. Use single sign-on integrations for connections with third-party applications.
    - [ ] C. Configure single sign-on by connecting the on-premises Active Directory using the AWS Directory Service AD Connector. Enable federation to the AWS services and accounts by using the IAM applications and services linking function. Leverage third-party single sign-on as needed.
    - [ ] D. Connect the company's Active Directory to AWS by using AD FS and SAML 2.0. Configure the AD FS claim rule to leverage Regex third-party single sign-on as needed, and add it to the AD FS server.

    <details>
       <summary>Answer</summary>

       - [ ] A. 需要构筑额外的登录页面
       - [x] B. 正确
       - [ ] C. 非服务相关角色应用场景
       - [ ] D. 需要维护AD FS服务器

    </details>

88. A Solutions Architect is designing a network solution for a company that has applications running in a data center in Northern Virginia. The applications in the company's data center require predictable performance to applications running in a virtual private cloud (VPC) located in us-east-1, and a secondary VPC in us-west-2 within the same account. The company data center is collocated in an AWS Direct Connect facility that serves the us-east-1 region. The company has already ordered an AWS Direct Connect connection and a cross-connect has been established. Which solution will meet the requirements at the cost?
    - [ ] A. Provision a Direct Connect gateway and attach the virtual private (VGW) for the VPC in us-east-1 and the VGW for the VPC in us-west-2. Create a private VIF on the Direct Connect connection and associate it to the Direct Connect gateway.
    - [ ] B. Create private VIFs on the Direct Connect connection for each of the company's VPCs in the us-east-1 and us-west-2 regions. Configure the company's data center router to connect directly with the VPCs in those regions via the private VIFs.
    - [ ] C. Deploy a transit VPC solution using Amazon EC2-based router instances in the us-east-1 region. Establish IPsec VPN tunnels between the transit routers and virtual private gateways (VGWs) located in the us-east-1 and us-west-2 regions, which are attached to the company's VPCs in those regions. Create a public VIF on the Direct Connect connection and establish IPsec VPN tunnels over the public V IF between the transit routers and the company's data center router.
    - [ ] D. Order a second Direct Connect connection to a Direct Connect facility with connectivity to the us-west-2 region. Work with partner to establish a network extension link over dark fiber from the Direct Connect facility to the company's data center. Establish private VIFs on the Direct Connect connections for each of the company's VPCs in the respective regions. Configure the company's data center router to connect directly with the VPCs in those regions via the private VIFs.

    <details>
       <summary>Answer</summary>

       DX是全球资源，可以直接连接，答案A

    </details>

89. A company has a web service deployed in the following two AWS Regions:us-west-2 and us-east-1. Each AWS region runs an identical version of the web service. Amazon Route 53 is used to route customers to the AWS Region that has the lowest latency. The company wants to improve the availability of the web service in case an outage occurs in one of the two AWS Regions. A Solutions Architect has recommended that a Route 53 health check be performed. The health check must detect a specific text on an endpoint. What combination of conditions should the endpoint meet to pass the Route 53 health check? (Choose two.)
    - [ ] A. The endpoint must establish a TCP connection within 10 seconds.
    - [ ] B. The endpoint must return an HTTP 200 status.
    - [ ] C. The endpoint must return an HTTP 2xx or 3xx status.
    - [ ] D. The specific text string must appear within the first 5,120 bytes of the response.
    - [ ] E. The endpoint must respond to the request within the number of seconds specified when creating the health check.

    <details>
       <summary>Answer</summary>

       如果是匹配字符串，建立TCP连接的时间是4秒，且返回的是2xx或者是3xx状态码，排除A和B，C正确，Route 53 将在响应正文中搜索您指定的字符串。该字符串必须完全显示在响应正文的前 5120 个字节中，D正确，答案CD

    </details>

90. A company operating a website on AWS requires high levels of scalability, availability, and performance. The company is running a Ruby on Rails application on Amazon EC2. It has a data tier on MySQL 5.6 on Amazon EC2 using 16 TB of Amazon EBS storage Amazon CloudFront is used to cache application content. The Operations team is reporting continuous and unexpected growth of EBS volumes assigned to the MySQL database. The Solutions Architect has been asked to design a highly scalable, highly available, and high-performing solution. Which solution is the MOST cost-effective at scale?
    - [ ] A. Implement Multi-AZ and Auto Scaling for all EC2 instances in the current configuration. Ensure that all EC2 instances are purchased as reserved instances. Implement new elastic Amazon EBS volumes for the data tier.
    - [ ] B. Design and implement the Docker-based containerized solution for the application using Amazon ECS. Migrate to an Amazon Aurora MySQL Multi-AZ cluster. Implement storage checks for Aurora MySQL storage utilization and an AWS Lambda function to grow the Aurora MySQL storage, as necessary. Ensure that Multi-AZ architectures are implemented.
    - [ ] C. Ensure that EC2 instances are right-sized and behind an Elastic Load Balancing load balancer. Implement Auto Scaling with EC2 instances. Ensure that the reserved instances are purchased for fixed capacity and that Auto Scaling instances run on demand. Migrate to an Amazon Aurora MySQL Multi-AZ cluster. Ensure that Multi-AZ architectures are implemented.
    - [ ] D. Ensure that EC2 instances are right-sized and behind an Elastic Load Balancer. Implement Auto Scaling with EC2 instances. Ensure that Reserved instances are purchased for fixed capacity and that Auto Scaling instances run on demand. Migrate to an Amazon Aurora MySQL Multi-AZ cluster. Implement storage checks for Aurora MySQL storage utilization and an AWS Lambda function to grow Aurora MySQL storage, as necessary. Ensure Multi-AZ architectures are implemented.

    <details>
       <summary>Answer</summary>

       - [ ] A. 用EC2上的MySQL不是最佳实践
       - [ ] B. 极光的磁盘是可以自动扩张的
       - [x] C. 正确
       - [ ] D. 同B

    </details>

91. The Security team needs to provide a team of interns with an AWS environment so they can build the serverless video transcoding application. The project will use Amazon S3, AWS Lambda, Amazon API Gateway, Amazon Cognito, Amazon DynamoDB, and Amazon Elastic Transcoder. The interns should be able to create and configure the necessary resources, but they may not have access to create or modify AWS IAM roles. The Solutions Architect creates a policy and attaches it to the interns' group. How should the Security team configure the environment to ensure that the interns are self-sufficient?
    - [ ] A. Create a policy that allows creation of project-related resources only. Create roles with required service permissions, which are assumable by the services.
    - [ ] B. Create a policy that allows creation of all project-related resources, including roles that allow access only to specified resources.
    - [ ] C. Create roles with the required service permissions, which are assumable by the services. Have the interns create and use a bastion host to create the project resources in the project subnet only.
    - [ ] D. Create a policy that allows creation of project-related resources only. Require the interns to raise a request for roles to be created with the Security team. The interns will provide the requirements for the permissions to be set in the role.

    <details>
       <summary>Answer</summary>

       - [x] A. 正确
       - [ ] B. 不应该赋予操作Role的权限
       - [ ] C. 有一些资源是全球的，不在VPC内的
       - [ ] D. 这个不能自给自足

    </details>

92. A company is running a commercial Apache Hadoop cluster on Amazon EC2. This cluster is being used daily to query large files on Amazon S3. The data on Amazon S3 has been curated and does not require any additional transformations steps. The company is using a commerciALBusiness intelligence (BI) tool on Amazon EC2 to run queries against the Hadoop cluster and visualize the data. The company wants to reduce or eliminate the overhead costs associated with managing the Hadoop cluster and the BI tool. The company would like to remove to a more cost-effective solution with minimal effort. The visualization is simple and requires performing some basic aggregation steps only. Which option will meet the company's requirements?
    - [ ] A. Launch a transient Amazon EMR cluster daily and develop an Apache Hive script to analyze the files on Amazon S3. Shut down the Amazon ENIR cluster when the job is complete. The use the Amazon QuickSight to connect to Amazon EMR and perform the visualization.
    - [ ] B. Develop a stored procedure invoked from a MySQL database running on Amazon EC2 to analyze EC2 to analyze the files in Amazon S3. Then use a fast in-memory BL tool running on Amazon EC2 to visualize the data.
    - [ ] C. Develop a script that uses Amazon Athena to query and analyze the files on Amazon S3. Then use Amazon QuickSight to connect to Athena and perform the visualization.
    - [ ] D. Use a commercial extract, transform, load (ETL) tool that runs on Amazon EC2 to prepare the data for processing. Then switch to a faster and cheaper Bl tool that runs on Amazon EC2 to visualize the data from Amazon S3.

    <details>
       <summary>Answer</summary>

       简单题，答案C

    </details>

93. A large multinational company runs a timesheet application on AWS that is used by staff across the world. The application runs on Amazon EC2 instances in an Auto Scaling group behind an Elastic Load Balancing (ELB) load balancer, and stores in an Amazon RDS MySQL Multi-AZ database instance. The CFO is concerned about the impact on the business if the application is not available. The application must not be down for more than two hours, but he solution must be as cost-effective as possible. How should the Solutions Architect meet the CFO's requirements while minimizing data loss?
    - [ ] A. In another region, configure a read replica and create a copy of the infrastructure. When an issue occurs, promote the read replica, and configure as an Amazon RDS Multi-AZ database instance. Update the DNS to point to the other region's ELB.
    - [ ] B. Configure a 1-day window of 60-minute snapshots of the Amazon RDS Multi-AZ database instance. Create an AWS CloudFormation template of the application infrastructure that uses the latest snapshot. When an issue occurs, use the AWS CloudFormation template to create the environment in another region. Update the DNS record to point to the other region's ELB.
    - [ ] C. Configure a 1-day window of 60-minute snapshots of the Amazon RDS Multi-AZ database instance which is copied to another region. Crate an AWS CloudFormation template of the application infrastructure that uses the latest copied snapshot. When an issue occurs, use the AWS CloudFormation template to create the environment in another region. Update the DNS record to point to the other region's ELB.
    - [ ] D. Configure a read replica in another region. Create an AWS CloudFormation template of the application infrastructure. When an issue occurs, promote the read replica, and configure as an Amazon RDS Multi-AZ database instance and use the AWS CloudFormation template to create the environment in another region using the promoted Amazon RDS instance. Update the DNS record to point to the other region's ELB.

    <details>
       <summary>Answer</summary>

       - [ ] A. 没必要事先在两外一个区域架构一个网站，这样会很贵
       - [ ] B. 快照需要复制到另外一个区域
       - [ ] C. 会有一个小时的数据丢失
       - [x] D. 正确

    </details>

94. A development team has created a series ofAWS CloudFormation templates to help deploy services. They created a template for a network viltual private (VPC) stack, a database stack, a bastion host stack, and a web application-specific stack. Each service requires the deployment of at least:A network'VPC stackA bastion host stackA web application stack Each template has multiple input parameters that make it difficult to deploy the services individually from the AWS CloudFormation console. The input parameters from one stack are typically outputs from other stacks. For example, the VPC ID, subnet IDs, and security groups from the network stack may need to be used in the application stack or database stack Which actions will help reduce the operation burden and the number of parameters passed into a service deployment? (Choose two.)
    - [ ] A. Create a new AWS CloudFormation template for each service. After the existing templates to use cross-stack references to eliminate passing many parameters to each template. Call each required stack for the application as a nested stack from the new stack. Call the newly created service stack from the AWS CloudFormation console to deploy the specific service with a subset of the parameters previously required.
    - [ ] B. Create a new portfolio in AWS Service Catalog for each service. Create a product for each existing AWS CloudFormation template required to build the service. Add the products to the portfolio that represents that service in AWS Service Catalog. To deploy the service, select the specific service portfolio and launch the portfolio with the necessary parameters to deploy all templates.
    - [ ] C. Set up an AWS CodePipeline workflow for each service. For each existing template, choose AWS CloudFormation as a deployment action. Add the AWS CloudFormation template to the deployment action. Ensure that the deployment actions are processed to make sure that dependences are obeyed. Use configuration files and scripts to share parameters between the stacks. To launch the service, execute the specific template by choosing the name of the service and releasing a change.
    - [ ] D. Use AWS Step Functions to define a new service. Create a new AWS CloudFormation template for each service. After the existing templates to use cross-stack references to eliminate passing many parameters to each template. Call each required stack for the application as a nested stack from the new service template. Configure AWS Step Functions to call the service template directly. In the AWS Step Functions console, execute the step.
    - [ ] E. Create a new portfolio for the Services in AWS Service Catalog. Create a new AWS CloudFormation template for each service. After the existing templates to use cross-stack references to eliminate passing many parameters to each template. Call each required stack for the application as a nested stack from the new stack. Create a product for each application. Add the service template to the product. Add each new product to the portfolio. Deploy the product from the portfolio to deploy the service with the necessary parameters only to start the deployment.

    <details>
       <summary>Answer</summary>

       使用嵌套template，选A；使用Service Catalog选E，答案AE

    </details>

95. A company has an application behind a load balancer with enough Amazon EC2 instances to satisfy peak demand. Scripts and third-party deployment solutions are used to configure EC2 instances when demand increases, or an instance fails. The team must periodically evaluate the utilization of the instance types to ensure that the correct sizes are deployed. How can this workload be optimized to meet these requirements?
    - [ ] A. Use CloudFormation to create AWS CloudFormation stacks from the current resources. Deploy that stack by using AWS CloudFormation in the same region. Use Amazon CloudWatch alarms to send notifications about underutilized resources to provide cost-savings suggestions.
    - [ ] B. Create an Auto Scaling group to scale the instances and use AWS CodeDeploy to perform the configuration. Change from a load balancer to an Application Load Balancer. Purchase a third-party product that provides suggestions for cost savings on AWS resources.
    - [ ] C. Deploy the application by using AWS Elastic Beanstalk with default options. Register for an AWS Support Developer plan. Review the instance usage for the application by using Amazon CloudWatch and identify less expensive instances that can handle the load. Hold monthly meetings to review new instance types and determine whether Reserved instances should be purchased.
    - [ ] D. Deploy the application as a Docker image by using Amazon ECS. Set up Amazon EC2 Auto Scaling and Amazon ECS scaling. Register for AWS Business Support and use Trusted Advisor checks to provide suggestions on cost savings.

    <details>
       <summary>Answer</summary>

       - [ ] A. 不关CloudFormation什么事儿
       - [ ] B. CodeDeploy是部署应用的，不管基盘部署
       - [ ] C. 答非所问
       - [x] D. 正确

    </details>

96. A large global financial services company has multiple business units. The company wants to allow Developers to try new services, but there are multiple compliance requirements for different workloads. The Security team is concerned about the access strategy for on-premises and AWS implementations. They would like to enforce governance for AWS services used by business team for regulatory workloads, including Payment Card Industry (PCI) requirements. Which solution will address the Security team's concerns and allow the Developers to fry new services?
    - [ ] A. Implement a strong identity and access management model that includes users, groups, and roles in various AWS accounts. Ensure that centralized AWS CloudTrail logging is enabled to detect anomalies. Build automation with AWS Lambda to tear down unapproved AWS resources for governance.
    - [ ] B. Build a multi-account strategy based on business units, environments, and specific regulatory requirements. Implement SAML-based federation across all AWS accounts with an on-premises identity store. Use AWS Organizations and build organizational units (OUs) structure based on regulations and service governance. Implement service control policies across OUs.
    - [ ] C. Implement a multi-account strategy based on business units, environments, and specific regulatory requirements. Ensure that only PCI-compliant services are approved for use in the accounts. Build IAM policies to give access to only PCI-compliant services for governance.
    - [ ] D. Build one AWS account for the company for the strong security controls. Ensure that all the service limits are raised to meet company scalability requirements. Implement SAML federation with an on-premises identity store and ensure that only approved services are used in the account.

    <details>
       <summary>Answer</summary>

       使用AWS Organization是最佳实践，答案B

    </details>

97. A company had a tight deadline to migrate its on-premises environment to AWS. It moved over Microsoft SQL Servers and Microsoft Windows Servers using the virtual machine import/export service and rebuild other applications native to the cloud. The team created both Amazon EC2 databases and used Amazon RDS. Each team in the company was responsible for migrating their applications and would like suggestions on reducing its AWS spend. Which steps should a Solutions Architect take to reduce costs?
    - [ ] A. Enable AWS Business Support and review AWS Trusted Advisor's cost checks. Create Amazon EC2 Auto Scaling groups for applications that experience fluctuating demand. Save AWS Simple Monthly Calculator reports in Amazon S3 for trend analysis. Create a master account under Organizations and have teams join for consolidating billing.
    - [ ] B. Enable Cost Explorer and AWS Business Support Reserve Amazon EC2 and Amazon RDS DB instances. Use Amazon CloudWatch and AWS Trusted Advisor for monitoring and to receive cost-savings suggestions. Create a master account under Organizations and have teams join for consolidated billing.
    - [ ] C. Create an AWS Lambda function that changes the instance size based on Amazon CloudWatch alarms. Reserve instances based on AWS Simple Monthly Calculator suggestions. Have an AWS Well-Architected framework review and apply recommendations. Create a master account under Organizations and have teams join for consolidated billing.
    - [ ] D. Create a budget and monitor for costs exceeding the budget. Create Amazon EC2 Auto Scaling groups for applications that experience fluctuating demand. Create an AWS Lambda function that changes instance sizes based on Amazon CloudWatch alarms. Have each team upload their bill to an Amazon S3 bucket for analysis of team spending. Use Spot instances on nightly batch processing jobs.

    <details>
       <summary>Answer</summary>

       简单题，答案B

    </details>

98. A company wants to replace its call system with a solution built using AWS managed services. The company call center would like the solution to receive calls, create contact flows, and scale to handle growth projections. The call center would also like the solution to use deep learning capabilities to recognize the intent of the callers and handle basic tasks, reducing the need to speak an agent. The solution should also be able to query business applications and provide relevant information back to calls as requested. Which services should the Solution Architect use to build this solution? (Choose three.)
    - [ ] A. Amazon Rekognition to identity who is calling.
    - [ ] B. Amazon Connect to create a cloud-based contact center.
    - [ ] C. Amazon Alexa for Business to build conversational interface.
    - [ ] D. AWS Lambda to integrate with internal systems.
    - [ ] E. Amazon Lex to recognize the intent of the caller.
    - [ ] F. Amazon SQS to add incoming callers to a queue.

    <details>
       <summary>Answer</summary>

       - [ ] A. Amazon Rekognition用于图像识别
       - [x] B. Amazon Connect提供语言连接服务，符合场景，正确
       - [ ] C. Amazon Alexa是只能音响
       - [x] D. 通过Lambda进行内部统括，正确
       - [x] E. Amazon Lex提供对话业务，正确
       - [ ] F. SQS不干这些事儿

    </details>

99. A large company is migrating its entire IT portfolio to AWS. Each business in the company has a standalone AWS account that supports both development and test environments. New accounts to support production workloads will be needed soon. The Finance department requires a centralized method for payment but must maintain visibility into each group's spending to allocate costs. The Security team requires a centralized mechanism to control IAM usage in all the company's accounts. What combination of the following options meet the company's needs with LEAST effort? (Choose two.)
    - [ ] A. Use a collection of parameterized AWS CloudFormation templates defining common IAM permissions that are launched into each account. Require all new and existing accounts to launch the appropriate stacks to enforce the least privilege model.
    - [ ] B. Use AWS Organizations to create a new organization from a chosen payer account and define an organizational unit hierarchy. Invite the existing accounts to join the organization and create new accounts using Organizations.
    - [ ] C. Require each business unit to use its own AWS accounts. Tag each AWS account appropriately and enable Cost Explorer to administer chargebacks.
    - [ ] D. Enable all features of AWS Organizations and establish appropriate service control policies that filter IAM permissions for sub-accounts.
    - [ ] E. Consolidate all of the company's AWS accounts into a single AWS account. Use tags for billing purposes and IAM's Access Advice feature to enforce the least privilege model.

    <details>
       <summary>Answer</summary>

       - [ ] A. 影响范围太大了
       - [x] B. 正确
       - [ ] C. 无法满足统一支付
       - [x] D. 正确
       - [ ] E. 合并成一个账户不是最佳解决方案

    </details>

100. A company collects a steady stream of 10 million data records from 100,000 sources each day. These records are written to an Amazon RDS MySQL DB. A query must produce the daily average of a data source over the past 30 days. There are twice as many reads as writes. Queries to the collected data are for one source ID at a time. How can the Solutions Architect improve the reliability and cost effectiveness of this solution?
     - [ ] A. Use Amazon Aurora with MySQL in a Multi-AZ mode. Use four additional read replicas.
     - [ ] B. Use Amazon DynamoDB with the source ID as the partition key and the timestamp as the sort key. Use a Time to Live (TTL) to delete data after 30 days.
     - [ ] C. Use Amazon DynamoDB with the source ID as the partition key. Use a different table each day.
     - [ ] D. Ingest data into Amazon Kinesis using a retention period of 30 days. Use AWS Lambda to write data records to Amazon ElastiCache for read access.

     <details>
        <summary>Answer</summary>

        简单题，答案B

     </details>
