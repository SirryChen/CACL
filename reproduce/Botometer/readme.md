Botometer 101: Social bot practicum for computational social scientists
---
### reproduce note
Since this method doesn't use any information from graph, the final results will be same with those in Twibot-22 benchmark.
So we simply use the given scores as the results of our reproduction.

### note in Twibot-22 benchmark

This is an unofficial implementation Botometer. Coding by Herun Wan ([email address](wanherun@stu.xjtu.edu.cn))

- **authors**: Kai-Cheng Yang, Emilio Ferrara, Filippo Menczer
- **link**: [https://arxiv.org/abs/2201.01608](https://arxiv.org/abs/2201.01608)
- **introduction**: Botometer (formerly BotOrNot) is a public website to checks the activity of a Twitter account and gives it a score, where higher scores mean more bot-like activity. Botometer's classification system generates more 1,000 features using available meta-data and information extracted from interaction patterns and content.


- **implement details in Twibot-22 benchmark**:  
  
  We adopt the "english" ("english" or "universal") score returned by Botometer API to determine if a user is bot or not.  According to the research of [Lynnette Hui Xian Ng](https://www.sciencedirect.com/science/article/pii/S2468696422000027), we choose 0.75 as the threshold for judging bots. We remove the users which were unable to  get score from Botometer API because they are banned or they don't post any tweets.

### Results

|         dataset         | accuracy | f1-score | precision | recall |
| :---------------------: | :------: | :------: | :-------: | :----: |
| Botometer-feedback-2019 |  50.00   |  30.77   |   21.05   | 57.14  |
|       Cresci-2015       |  57.92   |  66.90   |   50.54   | 98.95  |
|       Cresci-2017       |  94.16   |  96.12   |   93.35   | 99.69  |
|   Cresci-rtbust-2019    |  69.23   |  78.95   |   65.22   | 100.0  |
|    Cresci-stock-2018    |  72.62   |  79.59   |   68.50   | 94.96  |
|       Gilani-2017       |  71.56   |  77.39   |   62.99   | 87.91  |
|      Midterm-2018       |  89.46   |  46.03   |   31.18   | 87.88  |
|        Twibot-20        |  53.09   |  53.13   |   55.67   | 50.82  |
|        Twibot-22        |  49.87   |  42.75   |   30.81   | 69.80  |
