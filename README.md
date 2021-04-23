### How to use

**Auto Encoder**
```
>python train.py --objective ae --save_fn model.pth --input_size 27  --step 1 --btl_size 20 --n_epochs 500 --imp
```
- save_fn : 생성할 오토인코더 모델 파일명
- input_size : 변환할 데이터 차원
- step : 각 압축마다 줄일 차원(총 5번의 압축을 거침)
- btl_size : 최종 압축차원
- gpu_id : cuda가 사용가능하면 자동으로 지원, cpu를 사용하고 싶으면 -1로 지정
- batch_size : 배치 사이즈
- n_epochs : 에포크
- verbose : 출력할 메시지 양 조절 (0 ~ 2)
- imp : 결측값 대체 여부

```
>python encode.py --model_fn model.pth --save_fn encode_result.py
```

- model_fn : 사용할 오코인코더 모델
- save_fn : 저장할 출력파일 이름 (data 디렉토리)

**Classify**
```
>python train.py --objective clf --imp --model_name lgb --k 5
```
- model_name : 사용할 모델 이름 (lgb / xgb)
- save_fn : 저장할 출력파일 이름 (data 디렉토리)