## learned
- `%matplotlib inline` => inline으로 figure 그려줌
- `%matplotlib qt5` or `%matplotlib tk` => qt5나 tk를 이용하여 새로운 창에 figure 그려줌
- `seaborn`
    - matplotlib와 같은 시각화 library
    - matplotlib와 같이 import 해놓으면 더 prettier한 그래프 그려줌
- type of feature
    - categorical feature
        - 어떤 기준에 따라 여러 category 중 하나로 분류할 수 있는 feature
        - 각 category는 서로 ordering 할 수 없음
        - ex) 우리의 feature 중 Sex, Embarked
    - ordinal feature
        - categorical feature와 유사하지만 ordering이 가능한 feature
        - ex) Height: Tall, Medium, Small은 Tall > Medium > Small로 ordering 가능
        - **categorical과 ordinal은 기준을 잡기 나름**
    - continuous feature
        - 연속값(숫자 등)을 갖는 feature, 어떤 두 feature의 사이값으로 feature가 존재할 수 있음
        - ex) 우리의 feature 중 Age
- pandas isnull(), isna() are alias for each other
- sns.regplot: data의 산점도와 linear regression 했을 때의 line을 그려줌


## About correlation
- Only numeric features can have correaltion(Alpahbet, String cannot have correaltion normarlly)
- If increase in feature A leads to increase in feature B, then they are positively correlated.
- Value 1 means perfect positive correaltion
- If increase in feature A leads to decrease in feature B, then they are negatively correlated.
- Value -1 means perfect negative correlation
- If correlation between A and B are about 1, then increase in A leads to increase in B. This means that A and B are very similar feature

## confusion matrix terminology
- Table that specifies classfication performance
- axis1: label, axis2: predicted
- terms
    - True Positive(TP): label=Positive and prediction=Negative
    - True Negative(TN): label=Negative and prediction=Positive
    - False Positive(FP): label=Negative and prediction=Positive // Type1 Error
    - False Negative(FN): label=Positive and prediction=Neagtive // Type2 Error
    - P = TP + FN
    - N = TN + FP
    - total = P + N = TP + TN + FP + FN
    - Accuracy
        - how often is the classifier correct
        - (TP+TN)/total
    - Misclassification Rate
        - == Error Rate
        - how often is the classifier incorrect
        - (FN+FP)/total = 1 - Accuracy
    - True Positive Rate(TPR)
        - == Sensitivity, Recall, Hit rate
        - When label=Positive, how often prediction=Positive?
        - TP/P = TP / (TP+FN)
    - False Positive Rate(FPR)
        - When label=Negative, how often prediction=Positive?
        - FP/N = FP / (TN+FP)
    - True Neagtive Rate(TNR)
        - == Specificity
        - When label=Negative, how often prediction=Negative?
        - TN/N = TN / (TN+FP)
    - False Negative Rate(FNR)
        - When label=Positive, how often prediction=Negative?
        - FN/P = FN / (TP+FN)
    - TPR + FNR = 1
    - FPR + TNR = 1
    - Precision
        - When prediction=True, how often label=True?
        - TP/(TP+FP)

## Ensembling
- in short, combine multiple `weak` models to make it `strong`
- how to ensemble
    - voting
        - collect each result of models and select the most voted one
    - bagging / pasting
        - both bagging and pasting use multiple similar models
        - both bagging and pasting resample from original data
        - bagging(bootstap aggregating): resample allowing duplicate
        - pasting: resample dis-allowing duplicate
        - train multiple model `parallel`
    - boosting(ada / gradient)
        - train multiple model `sequentially`
        - the next model focus on the `weak` point of the previous one
        - ada boost(adaptive boosting): more weight on previous `mis-predicted` sample
        - gradient boosting: learn `residual error` of previous model
- reference: https://excelsior-cjh.tistory.com/166

## scaling vs normalization
- 둘 다 numeric value에 대해서 적용
- scaling
    - range of data만 바꿔주고 shape of distribution은 안바꿔줌
- normalization
    - shape of distribution을 바꿔주는게 주요 목적
    - shape of distribution을 바꾸면서 range도 자연스레 바뀔 수 있음
    - 이름에서도 알 수 있듯이 normal distribution(정규분포)화 해주는 것
    - `normal distribution을 가정하는 ML method를 적용하기위해 normalization을 해줘야함`


## Conclusion
- 잘되는 model
    - XGBoost
    - lightgbm
    - sklearn.ensemble.GradientBoost series
- should analyze data before select/adapt model
    - As same with `design before coding`
- visualization exercise using seaborn
- regression의 경우 target의 skew(), kurt() 등을 이용하여 정규분포와 얼마나 차이나는지 확인
- categorical data => boxplot, numerical data => regplot, scatterplot 사용 고려
- categorical data => sklearn.LabelEncoder 고려
- 좋은 결과 나오면 반드시 코드 git에 commit하기