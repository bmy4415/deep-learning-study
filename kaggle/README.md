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

## About correlation
- Only numeric features can have correaltion(Alpahbet, String cannot have correaltion normarlly)
- If increase in feature A leads to increase in feature B, then they are positively correlated.
- Value 1 means perfect positive correaltion
- If increase in feature A leads to decrease in feature B, then they are negatively correlated.
- Value -1 means perfect negative correlation
- If correlation between A and B are about 1, then increase in A leads to increase in B. This means that A and B are very similar feature