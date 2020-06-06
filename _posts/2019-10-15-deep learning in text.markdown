---
layout: post
title:  "Assisting Customer Service Agents using deep learning techniques"
excerpt: "Implementing machine learning systems that suggest responses to agents 
when they are responding to an email or chat or social media conversation"
date:   2019-10-15 22:23:04 +0530
categories: jekyll update
---
## Introduction

Applications that assist Agents and supervisors when they respond to customers are very common. For example,
Agents are assisted with predefined responses when they answer email, chat or social media customer queries. 
The usefulness and accuracy of these suggestions can be improved by deep learning 
and transfer learning techniques.

#### Examples of Agent Assist Systems
When a question pops up on Agent's chat window, the potential responses/answers can appear on the Agent's screen
for him to either select one of the responses or build his own response based on the responses available. 

Similarly, email and social media systems can suggest top N responses for an incoming interaction from customer.

Further, sentiments and Intents of these interactions can be shown to the agents for appropriate responses.

Machine learning systems can produce these responses from historical interactions, documents, FAQs, 
internal web pages and knowledge articles. 

### Our Work

In the email scenario, we wanted to suggest top 5 email responses when an agent receives a new email 
to service. This will be by using the historical emails that have been answered before (by other agents) 

#### Feature based approach

We started with a simple bag-of-words approach with TF-IDF scores to find cosine similarity between emails and the top most 
ones that are closer to the un-answered email can be listed to the Agent. We built a very basic model 
with this approach (with lots of other features such as exact word count) and this works perfectly OK in few cases.

There are few challenges such as the model couldn’t take the language semantic into account. 
In production, we had to calculate cosine similarity with a huge set of interactions to find out 
the ones that are closer to a particular email. So, this doesn’t scale well for real time use cases.

A simple code snippet using [sklearn](https://scikit-learn.org/stable/) package for calculating cosine similarity.

{% highlight python %}

    from sklearn.feature_extraction.text import TfidfVectorizer
    m_tfidf = TfidfVectorizer(min_df=0, use_idf= True, tokenizer=tokenize, preprocessor=pre_process, norm='l2')
    tfidf = m_tfidf.fit_transform(all_docs)
    
    from sklearn.metrics.pairwise import linear_kernel
    cos_sim = linear_kernel(tfidf[-1], tfidf).flatten()
    doc_index = cos_sim.argsort()[:-5:-1]
    
{% endhighlight %}

#### BERT based model Approach
Recent advancements in text architectures such as Transformers and BERT have improved accuracy the
of Question-Answering, Text Classification, Sentiment Analysis and various other tasks.

We calculated BERT similarity between emails instead of TF-IDF to improve the results. 

BERT is trained on huge corpuses of text and understands the language semantics better instead of 
just using word counts and word importance. We have used BERT in Q&A and text classification. 
You can find more details [here](2020-01-04-BERT%20QA.markdown)

#### Embedding Approach

Transfer Learning/pre-training is a boon in machine learning especially deep learning. 
This helps in applying the “learning” (read weight matrix) from a different dataset in a similar domain to a new problem in the same or related domain.
We were inspired by the Quora question similarity problem [kaggle competition](https://www.kaggle.com/c/quora-question-pairs) and wanted 
to apply some of the techniques to solve our use case. 

To build our models, we used GloVe embedding, Tensorflow and Keras.
Word Embedding captures very powerful semantic relationships in text data. 
We wanted to leverage the already trained GloVe/Word2Vec models. The central idea is to convert the 
email text into word embedding first followed by a convolution/RNN (LSTM) deep layers on top of it 
with fully connected layers and finally a softmax layer for classification. This is similar to the 
other conventional deep network architectures except that the training will be disabled for the 
embedding layer. In Keras, it is as simple as setting trainable=false for the embedding layer.

## Summary
Contact centers handle lot of text data and deriving meaningful insights using machine learning techniques
can bring value by 
- reducing **agent handling time** allowing agent spend more time on important tasks
- improving **customer experience** by providing better resolutions
- **first call resolution** based on targeted, accurate responses.

Please reach out us if you need more details on this!


[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
