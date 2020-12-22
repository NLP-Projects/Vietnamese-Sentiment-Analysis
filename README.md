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

This task is polarity classification, i.e., to evaluate the ability of classifying Vietnamese reviews/documents into one of three categories: “positive”, “negative”, or “neutral”.  

### 3. Data for Training and Testing
A review can be very complex with different sentiments on various objects. Therefore, we set some constraints on the dataset as follows:

- The dataset only contains reviews having personal opinions.

- The data are usually short comments, containing opinions on one object. There is no limitation on the number of the object's aspects mentioned in the comment.

- Label (positive/negative/neutral) is the overall sentiment of the whole review.

- The dataset contains only real data collected from social media, not artificially created by human.

Note: Normally, it is very difficult to rate a neutral comment because the opinions are always inclinable to be negative or positive. A review is rated to be neutral when we cannot decide whether it is positive or negative. The neutral label can be used for the situations in which a review contains both positive and negative opinions but when combining them, the comment becomes neutral.

#### Some examples of the dataset:

##### Label : Review

Pos: Đẳng cấp Philips, máy đẹp, pin bền. Đóng và giao hàng rất chuyên nghiệp.

Pos: Tốt Giá vừa túi tiền đẹp và sang.

Pos: Rẻ hơn Samsung J1 nhưng cấu hình lại tốt hơn.

Pos: lướt web nhanh,chụp hình rõ nét, âm thanh ngoài trung bình rất xứng đáng với giá bán hiện giờ. pin đang trãi nghiệm (do mới sạc lần đầu).

 
Neg: Lâu lâu bị lỗi, màn hình cảm ứng không nhạy, chất lượng camera kém.

Neg: pin nhanh tụt, chỉ được xài 1 ngày.

Neg: Máy hay đơ màn hình, màn hình không nhạy dưới bên phải các phím M, N.

Neg: Mình trả cách đây gần 1 tháng rồi.


Neu: Pin khá hơn tí thì tốt nhỉ.

Neu: Đẹp thật, tiếc là ram và pin chưa ngon.

Neu: Vậy là không hỗ trợ thẻ nhớ. Một điểm hơi lăn tăn.

# Fundamentals of Deep Learning

http://perso.ens-lyon.fr/jacques.jayez/Cours/Implicite/Fundamentals_of_Deep_Learning.pdf

http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf


# Long Short-Term Memory with PyTorch
https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/


# Sentiment analysis using LSTM - PyTorch

https://www.kaggle.com/arunmohan003/sentiment-analysis-using-lstm-pytorch


# References

https://viblo.asia/p/phan-tich-phan-hoi-khach-hang-hieu-qua-voi-machine-learningvietnamese-sentiment-analysis-Eb85opXOK2G

https://arxiv.org/ftp/arxiv/papers/1412/1412.8010.pdf

Reference folder.

Code folder.
