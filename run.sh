CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python pl_bolts/models/self_supervised/simsiam/simsiam_module.py --fp32 --dataset cifar10 --optimizer sgd --batch_size 512 --learning_rate 0.03 --max_epochs 800 --weight_decay 0.0005 --arch resnet18 --hidden_mlp 512 --online_ft --gpus 1
