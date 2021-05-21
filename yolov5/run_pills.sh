nohup python3 train.py --batch 15 --epochs 100 --data training/dataset.yaml --cfg training/yolov5m.yaml --weights '' --img 416 --device 0 --adam  > nohup.out 2>&1
