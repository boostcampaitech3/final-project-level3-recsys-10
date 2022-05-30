# Movie Recommendation Baseline Code

영화 추천 대회를 위한 베이스라인 코드입니다. 다음 코드를 대회에 맞게 재구성 했습니다.

- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

## Installation

```
pip install -r requirements.txt
```

## How to run

1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```
