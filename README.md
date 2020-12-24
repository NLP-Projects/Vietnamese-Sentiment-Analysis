# PROGRAMMING ASSIGNMENT 4: SENTIMENT ANALYSIS FOR VIETNAMESE LANGUAGE

## Sentiment Analysis for Vietnamese Language

### 1. Introduction

With the development of technology and the Internet, different types of social media such as social networks and forums have allowed people to not only share information but also to express their opinions and attitudes on products, services and other social issues. The Internet becomes a very valuable and important source of information. People nowadays use it as a reference to make their decisions on buying a product or using a service. Moreover, this kind of information also let the manufacturers and service providers receive feedback about limitations of their products and therefore should improving them to meet the customer needs better. Furthermore, it can also help authorities know the attitudes and opinions of their residents on social events so that they can make appropriate adjustments.

Since early 2000s, opinion mining and sentiment analysis have become a new and active research topic in Natural language processing and Data mining. The major tasks in this topic can be listed as follows:

#### (+) Subjective classification: 

This is the task to detect that whether a document contains personal opinions or not (only provides facts).

#### (+) Polarity classification (Sentiment classification): 

The objective of this task is to classify the opinion of a document into one of three types, which are “positive”, “negative” and “neutral”.

#### (+) Spam detection: 

The goal of this task is to detect fake reviews and reviewers.

#### (+) Rating:

Rating the documents having personal opinions from 1 star to 5 star (very negative to very positive).

### Besides these common tasks, recently there are some other important tasks:

#### (+) Aspect-based sentiment analysis: 

The goal is to identify the aspects of given target entities and the sentiment expressed for each aspect.

#### (+) Opinion mining in comparative sentences: 

This task focuses on mining opinions from comparative sentences, i.e., to identify entities to be compared and determine which entities are preferred by the author in a comparative sentence.

### 2. Task Description

##### This task is polarity classification, i.e., to evaluate the ability of classifying Vietnamese reviews/documents into one of three categories: “positive”, “negative”, or “neutral”.

##### Students can choose machine learning or deep learning approaches for this task. Encourage students to choose the Deep Learning approach to solve this task, using modern neural network architecture such as LSTM.

### 3. Data for Training and Testing
A review can be very complex with different sentiments on various objects. Therefore, we set some constraints on the dataset as follows:

*  The dataset only contains reviews having personal opinions.

*  The data are usually short comments, containing opinions on one object. There is no limitation on the number of the object's aspects mentioned in the comment.

*  Label (positive/negative/neutral) is the overall sentiment of the whole review.

*  The dataset contains only real data collected from social media, not artificially created by human.

Note: Normally, it is very difficult to rate a neutral comment because the opinions are always inclinable to be negative or positive. A review is rated to be neutral when we cannot decide whether it is positive or negative. The neutral label can be used for the situations in which a review contains both positive and negative opinions but when combining them, the comment becomes neutral.

#### Some examples of the dataset:

##### Label : Review

* Pos: Đẳng cấp Philips, máy đẹp, pin bền. Đóng và giao hàng rất chuyên nghiệp.

* Pos: Tốt Giá vừa túi tiền đẹp và sang.

* Pos: Rẻ hơn Samsung J1 nhưng cấu hình lại tốt hơn.

* Pos: lướt web nhanh,chụp hình rõ nét, âm thanh ngoài trung bình rất xứng đáng với giá bán hiện giờ. pin đang trãi nghiệm (do mới sạc lần đầu).

***
 
* Neg: Lâu lâu bị lỗi, màn hình cảm ứng không nhạy, chất lượng camera kém.

* Neg: pin nhanh tụt, chỉ được xài 1 ngày.

* Neg: Máy hay đơ màn hình, màn hình không nhạy dưới bên phải các phím M, N.

* Neg: Mình trả cách đây gần 1 tháng rồi.

***

* Neu: Pin khá hơn tí thì tốt nhỉ.

* Neu: Đẹp thật, tiếc là ram và pin chưa ngon.

* Neu: Vậy là không hỗ trợ thẻ nhớ. Một điểm hơi lăn tăn.

### 4. Model evaluation metrics

* Predicted: Outcome of the model on the validation set.

* Positive (P): Observation is positive.

* Negative (N): Observation is not positive. 

* True Positive (TP): Observation is positive, and is predicted correctly.

* False Negative (FN): Observation is positive, but predicted wrongly.

* True Negative (TN): Observation is negative, and predicted correctly.

* False Positive (FP): Observation is negative, but predicted wrongly.

##### Accuracy Score

A classification model’s accuracy is defined as the percentage of predictions it got right. However, it’s important to understand that it becomes less reliable when the probability of one outcome is significantly higher than the other one, making it less ideal as a stand-alone metric.

For example, if you have a dataset where 5% of all incoming emails are actually spam, we can adopt a less sophisticated model (predicting every email as non-spam) and get an impressive accuracy score of 95%. Unfortunately, most scenarios are significantly harder to predict.

The expression used to calculate accuracy is as follows:

Accuracy = TP + TN / TP + TN + FP + FN 

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#:~:text=Accuracy%20classification%20score.,set%20of%20labels%20in%20y_true.&text=Otherwise%2C%20return%20the%20fraction%20of%20correctly%20classified%20samples.


### 5. Approaches for sentiment analysis problem

#### Several typical approaches for the sentiment analysis problem are as follows: 

##### (+) Bayesian Networks Naive Bayes Classification Maximum Entropy Neural Networks Support Vector Machine.

 FEATURES/TECNIQUES: Term presence and frequency; Part of speech information; Negations; Opinion words and phrases.
 
 ADVANTAGES: the ability to adapt and create trained models for specific purposes and contexts.
 
 LIMITATIONS: the low applicability to new data because it is necessary the availability of labeled data that could be costly or even prohibitive

##### (+) Dictionary based approach Novel Machine Learning Approach Corpus based approach Ensemble Approaches. 

 FEATURES/TECNIQUES: Manual construction; Corpus-based; Dictionary-based. 
 
ADVANTAGES: wider term coverage.

LIMITATIONS: finite number of words in the lexicons and the assignation of a fixed sentiment orientation and score to words(+) Machine learning Lexicon based.

##### (+) Machine learning Lexicon based.

 FEATURES/TECNIQUES: Sentiment lexicon constructed using public resources for initial sentiment detection; Sentiment words as
features in machine learning method.

ADVANTAGES: lexicon/learning symbiosis, the detection and measurement of sentiment at the concept level and the lesser
sensitivity to changes in topic domain.

LIMITATIONS: noisy reviews.

##### Deep Learning-Based: 

Using models including LSTM, CNN, Multilayer Perceptron. Exploiting word embeddings models including Word2Vec, GloVe, fastText, Bert.

An overview of application of deep learning for sentiment analysis
![picture](https://github.com/NLP-Projects/Vietnamese-Sentiment-Analysis/blob/main/SA.png)

### 6. Fundamentals of Deep Learning

http://perso.ens-lyon.fr/jacques.jayez/Cours/Implicite/Fundamentals_of_Deep_Learning.pdf

http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf


#### Long Short-Term Memory with PyTorch

The structure of the Long Short-Term Memory
![picture](https://github.com/NLP-Projects/Vietnamese-Sentiment-Analysis/blob/main/LSTM.png)

https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/


#### Sentiment analysis using LSTM - PyTorch

https://www.kaggle.com/arunmohan003/sentiment-analysis-using-lstm-pytorch

### 7. Resources

*  pre-trained word embeddings: https://drive.google.com/open?id=1z1IDKaZuJXw5g7Yebr1GDq2UfvVR_aGx

*  Vietnamese SentiWordNet.

*  Code example (Code folder).

### 8. References 

[1]. https://viblo.asia/p/phan-tich-phan-hoi-khach-hang-hieu-qua-voi-machine-learningvietnamese-sentiment-analysis-Eb85opXOK2G

[2]. https://arxiv.org/ftp/arxiv/papers/1412/1412.8010.pdf

[3]. Reference folder.


