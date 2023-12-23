#### reproduce note
Since this method doesn't use any information from graph, the final results will be same with those in Twibot-22 benchmark.
So we simply use the given scores as the results of our reproduction.

**implement details in Twibot-22 benchmark**: 
   - We ignored the content-based features due to the lack of the information in our datasets 
   - We did not use the SMOTEENN method to balance the negative samples, since our datasets are well balanced and adding SMOTEENN did not improve the results
   - All datasets except Twibot-20 lack some of the core features in account based features such as default profile and geo-enabled, and some of the users’ data is missing ( i.e. no friends count t, follower count etc. ) and were set to 0 in our experiments


#### Results:

| dataset                   |      | acc    | precison | recall | f1     |
| ------------------------- | ---- | ------ | -------- | ------ | ------ |
| Twibot-22                 | mean | 0.7125 | 0.7990   | 0.5214 | 0.3709 |
| Twibot-20                 | mean | 0.8191 | 0.9656   | 0.7630 | 0.8524 |
| botometer-feedback-2019   | mean | 0.7597 | 0.3171   | 0.5909 | 0.4127 |
| cresci-rtbust-2019        | mean | 0.5000 | 0.5741   | 0.5124 | 0.5415 | 
| cresci-stock-2018         | mean | 0.8874 | 0.9989   | 0.8883 | 0.9403 | 
| midterm-2018              | mean | 0.8393 | 0.9991   | 0.8399 | 0.9126 |  
| cresci-2017               | mean | 0.9022 | 0.9156   | 0.9535 | 0.9342 |   
| gilani-2017               | mean | 0.4967 | 0.3517   | 0.4400 | 0.3910 |  
| cresci-2015               | mean | 0.8592 | 0.8570   | 0.9740 | 0.9118 |     