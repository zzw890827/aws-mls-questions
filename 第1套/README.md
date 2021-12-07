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
