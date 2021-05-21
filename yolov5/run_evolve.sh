nohup python3 train.py --epochs 10 --data training/dataset.yaml --weights 'weights/best.pt' --img 384 --device 0 --adam --cache --evolve > nohup.out 2>&1
