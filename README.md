# halohoops
NCAA Tournament prediction competition. The goal is pretty simple, generate probability of winner for every possible matchup combination in the 2022 NCAA March Madness Men's and Women's tournament. Evaluation metric is log loss.

[Men's Tournament Challenge](https://www.kaggle.com/c/mens-march-mania-2022)  
[Women's Tournament Challenge](https://www.kaggle.com/c/womens-march-mania-2022)  

## Methodology
In years, past I took a lazy approach and just leveraged the Ken Pom team ranking and team seed in a logistic regression model to generate probabilities. This usually put me somewhere in the middle of the pack. Previous results:

* 2021 Mens (459/707) with log loss of .64637
* 2019 Mens (372/862) with log loss of .49544
* 2019 Womens (193/497) with log loss of .39872
* 2018 Mens (848/933) with log loss of 1.08633

The reason for the crash and burn in 2018 was that I got a little too cute and manually set all of the 1 vs 16 matchups to probability of 1. Lo and behold, UVA was upset and that result was like a log-loss of infinity. **Consider setting a probability confidence limit like 98% or something to combat the dangers of a similar upset.**  

## Updated Approach  
In addition to re-creating team-based efficiency stats like [Ken Pom](https://kenpom.com/), things like depth (average class ranking or games played per player weighted by playing time?) and injuries could also be figured into play.
  * Definitely add more regular season features (https://www.kaggle.com/toshimelonhead/ncaa-march-madness-sabermetric-spin/notebook)
  * Conference (categorical variable)
  * Quad 1 Wins
  * Quad 4 Losses
  * Manual Adjustments??
Also look at this solution:
https://www.kaggle.com/prashantkikani/ncaam-2021-diverse-model-ensemble

### Phase 1 Results Log
Very simple logistic regression for both Men's and Women's tournaments only using KenPom rankings and seeds for Men's and seeds for Women's.

men's score: 0.56336 (289/313)
  * Submission 2: 0.55792
women's score: 0.45374 (162/186)
  * Submission 2: 0.45000
