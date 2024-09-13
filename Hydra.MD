# Train
```
# 100-epochs
python main_moco.py --dist-url 'tcp://localhost:29532' --multiprocessing-distributed --world-size 1 --rank 0  --moco-m-cos --crop-min=.2  ~/data/ImageNet -b 2048

# 300-epochs
python main_moco.py --dist-url 'tcp://localhost:29532' --multiprocessing-distributed --world-size 1 --rank 0  --lr=.3 --epochs=300 --moco-m-cos --crop-min=.2  ~/data/ImageNet -b 2048

```  