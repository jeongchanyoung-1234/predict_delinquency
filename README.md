### How to use

**Auto Encoder**
* note1 : 'train_ratio' shoule be as smaller as possible ex) 0.99
* note2 :  The result accuracy will always be 0 because it isn't classification task.
```
>python train.py --model_fn model.pth --objective ae --input_size 27  --step 1 --btl_size 20 --n_epochs 500 --train_ratio 0.99


...
Epoch 500 Train - Accuracy: 0.0000 Loss: 0.4161
        Valid - Accuracy: 0.0000 Loss: 0.5825 Lowest_loss=0.5406
```

```
>python encode_result.py --model_fn model.pth
```
