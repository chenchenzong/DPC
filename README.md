# [AAAI24] Dirichlet-Based Prediction Calibration for Learning with Noisy Labels（DPC）
by **Chen-Chen Zong, Ye-Wen Wang, Ming-Kun Xie, Sheng-Jun Huang**
## Usage

Train the network on the Symmmetric Noise CIFAR dataset (noise rate = 0.2):

```
python Train_cifar.py --dataset cifar10 --data_path ./dataset/cifar-10 --noise_mode sym --noise_rate 0.2 --lambda_u 0
python Train_cifar_aug.py --dataset cifar10 --data_path ./dataset/cifar-10 --noise_mode sym --noise_rate 0.2 --lambda_u 0
python Train_cifar.py --dataset cifar100 --data_path ./dataset/cifar-100 --noise_mode sym --noise_rate 0.2 --lambda_u 25
python Train_cifar_aug.py --dataset cifar100 --data_path ./dataset/cifar-100 --noise_mode sym --noise_rate 0.2 --lambda_u 25
```

Train the network on the Asymmmetric Noise CIFAR dataset (noise rate = 0.1):

```
python Train_cifar.py --dataset cifar10 --data_path ./dataset/cifar-10 --noise_mode asym --noise_rate 0.1 --lambda_u 0
python Train_cifar_aug.py --dataset cifar10 --data_path ./dataset/cifar-10 --noise_mode asym --noise_rate 0.1 --lambda_u 0
python Train_cifar.py --dataset cifar100 --data_path ./dataset/cifar-100 --noise_mode asym --noise_rate 0.1 --lambda_u 25
python Train_cifar_aug.py --dataset cifar100 --data_path ./dataset/cifar-100 --noise_mode asym --noise_rate 0.1 --lambda_u 25
```

Train the network on CIFAR-N dataset:

```
python Train_cifarN.py --dataset cifar10 --data_path ./dataset/cifar-10 --noise_type aggre --lambda_u 0
python Train_cifarN.py --dataset cifar100 --data_path ./dataset/cifar-100 --noise_type noisy100 --lambda_u 25
```

Train the network on miniwebvision dataset:

```
python Train_webvision_parallel_edl.py --data_path ./dataset/webvision/
python Train_webvision_parallel_edl_aug.py --data_path ./dataset/webvision/
```


