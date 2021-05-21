nohup python3 train.py --batch 15 --epochs 200 --data training/dataset.yaml --cfg training/yolov5m.yaml --weights '' --img 384 --device 0 --adam  > nohup.out 2>&1
