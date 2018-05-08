Original thread: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283

***Author: CPMP***

Summary of the solution:

***Feature Engineering***

Features were computed on the concatenation of train and test_supplement, ***sorted by click time then by original order***.

Only app, and os from the original features were kept. They were handled as categorical.

China days. ***Introduced 24 periods*** that start at 4 pm. These were used for lag features based on previous day(s) data.

User: ip, device, os triplets.

Aggregates on various feature groups: count, count of unique values, delta with previous value, delta with next value. Time to next click when grouped by user was important. ***delta with previous app***.

***Lag features***, based on previous China days values. ***Previous count by some grouping, and previous target mean by some grouping***. The latter was a weighted average with the overall target mean, the weights being such that groups with few rows in it had a value closer to the overall average. This is a standard normalization in target encoding.

***Ratios like number of clicks per ip, app to number of click per app.***

***Not last***. This was to capture the leak. It is one except for rows that are not the last of their group when grouped by user, app, and click time. I ignored channel as I think that clicks are attributed to the most recent click having same user and app as the download.

***Target***. This is to also capture the leak. I modified the target in train data by sorting is_attributed within group by user, app, and click time. The combination of both ways to capture the leak led to a boost between 0.0004 and 0.0005.

***Matrix factorization***. This was ***to capture the similarity between users and app***. I use several of them. They all start with the same approach; construct a matrix with log of click counts. I used: ip x app, user x app, and os x device x app. These matrices are extremely sparse (most values are 0). For the first two I used truncated svd from sklearn, which gives me latent vectors (embeddings) for ip and user. For the last one, given there are 3 factors, I implemented libfm in Keras and used the embeddings it computes. I used between 3 and 5 latent factors. All in all, these embeddings gave me a boost over 0.0010. I think this is what led me in top 10. I got some variety of models by varying which embeddings I was using.
