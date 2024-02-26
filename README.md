# On Defining and Proving Security for Model Stealing Defenses

Implementation of the paper "On Defining and Proving Security for Model Stealing Defenses".

## Setup

1. ```
   pip install -r requirements.txt
   ```

2. git clone https://github.com/tribhuvanesh/knockoffnets.git" # Download KnockoffNets repository

3. export PYTHONPATH="$PYTHONPATH:<PATH>/knockoffnets:<PATH>/adaptivemisinformation" # Add KnockoffNets and AdaptiveMisinformation to PYTHONPATH; Replace <PATH> with the path containing knockoffnets/adaptivemisinformation dirs

## admis

### Train Benign Model

```
python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128
```

### Test Benign Model

```
python admis/benign_user/test.py MNIST models/defender/mnist
```

### Get Benign Model Output_tensor

```
python admis/benign_user/tensor.py MNIST models/defender/mnist
```

## Sampler

In `admis/utils/defense.py` 

MNIST:

```
self.generator.load_state_dict(torch.load('./sampler/res/mnist/1/generator130.pth'))
```

CIFAR10:

```
self.generator.load_state_dict(torch.load('./sampler/res/cifar/1/generator70.pth'))
```

...

## Detector

`admis/adv_user/train_jbda.py` Get JBDA OOD dataset: 

    X_sub = jacobian_augmentation(model_clone, X_sub, y_sub, nb_classes=num_classes)
    #if aug_round == 0:
    #np.save('./mnist/X_sub1500.npy', X_sub)



In `admis/utils/defense.py` 

MNIST:

```
self.MultiClassifier.load_state_dict(torch.load('./detector/mnist/attackmnist3.pth'))
        self.MultiClassifier.eval()
        
outputs = self.MultiClassifier(x)
```

CIFAR10 , GTSRB:

```
self.model.classifier[6] = torch.nn.Linear(4096, 3).to(self.device)        
self.model.load_state_dict(torch.load('./detector/cifar/attackcifar4.pth'))

outputs = self.model(x).to(self.device)
```

ImageNette:

```
num_ftrs = self.model1.fc.in_features  
self.model1.fc = nn.Linear(num_ftrs, 3).to(self.device)  # 3分类
self.model1.load_state_dict(torch.load('./detector/image/attackimage1.pth'))

outputs = self.model1(x).to(self.device)
```

## Train Defender Model

### BCG

```
python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=BCG --oe_lamb 1 -doe KMNIST
```

### SG

```
python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SG --oe_lamb 1 -doe KMNIST
```

### Selective Misinformation

```
python admis/defender/train.py MNIST lenet -o models/defender/mnist -e 20 --lr 0.1 --lr-step 10 --log-interval 200 -b 128 --defense=SM --oe_lamb 1 -doe KMNIST
```


## Evaluate Attacks

### KnockoffNets Attack

#### BCG

```
python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --batch-size 1 --defense BCG -d 3

python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense BCG
```

#### SG

delta = 1， 0.95， 0.9，0.85， 0.8

```
delta = 0.8    #TODO
```

```
python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SG -d 3 --batch-size 64

python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense SG 
```

#### SM

```
python admis/adv_user/transfer.py models/defender/mnist --out_dir models/adv_user/mnist --budget 50000 --queryset EMNISTLetters --defense SM --defense_levels 0.99

python ./admis/adv_user/train_knockoff.py models/adv_user/mnist lenet MNIST --budgets 50000 --batch-size 128 --log-interval 200 --epochs 20 --lr 0.1 --lr-step 10 --defense SM --defense_level 0.99
```

### JBDA Attack

#### BCG

```
python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --aug_rounds=6 --epochs=10 --substitute_init_size=150  --lr 0.01 --defense BCG --batch-size 1
```

#### SG

```
python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --aug_rounds=6 --epochs=10 --substitute_init_size=150  --lr 0.01 --defense SG --batch-size 64
```

#### SM

```
python admis/adv_user/train_jbda.py ./models/defender/mnist/ ./models/adv_user/mnist/ lenet MNIST --defense=SM --aug_rounds=6 --epochs=10 --substitute_init_size=150 --defense_level=0.99 --lr 0.01
```

Note:

1. '--defense_levels' refers to the values of tau in the context of Selective Misinformation.

2. Varying the value of --defense_levels can be used to obtain the defender accuracy vs clone accuracy trade-off curve

## Test D-DAE

### Defense

`online/victim/bb_BCG.py` The same as  `admis/utils/defense.py` 

### Training Shadow Models and Target Models

```
python offline/train_shadow.py --task mnist
```

### Defense Recovery

`offline/model_lib/defense_device.py`

```
elif defense == 'BCG':
            self.BCG_bb = bcGenerator_device(self.model, # comma separated values specifying defense levels: delta(SM)
                                   num_classes=10) #TODO
            y_mod = self.BCG_bb(x, y)
            #print(y_mod)
            y_mod = list(y_mod.values())[0]
            #print(y_mod)
            return y_mod
```

`offline/recovery.py` trains and evaluates the restorer regarding different defenses.

An example of training the BCG defense recovery on the MNIST task:

```
python offline/recovery.py --task mnist --defense BCG
```

### Victim Models

```
# Format:
$ python online/victim/train.py DS_NAME ARCH -d DEV_ID \
        -o models/victim/VIC_DIR -e EPOCHS --pretrained
# where DS_NAME = {MNIST, CIFAR10, GTSRB, ImageNette}, ARCH = {lenet, vgg16_bn, resnet34, ...}
python online/victim/train.py MNIST lenet -d 3 \
        -o models/victim/MNIST-lenet-train-BCG -e 10 --log-interval 25 --defense=BCG --oe_lamb 1 -doe KMNIST
```

### Attack Models

#### KnockoffNets

First, in `online/adversary/train.py`

```
model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, restored=True, task=args.testdataset,
                                optimizer=optimizer, **params)
```

Note: change `restored=False` to `restored=True`.

Second, in `online/utils/model.py` `Line 104 - Line 127`

```
    if task == 'MNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("mnist")
        generator_path = ${Generator}
        meta_path = ${Meta-classifer}
        
# Generator stored in /generator/${task}${defense}
# Meta-classifier stored in /meta/${defense}${task}
```

```
$ python online/adversary/transfer.py random ${vic_dir} ${strat} ${defense_args} \
    --out_dir ${out_dir} \
    --batch_size ${batch_size} \
    -d ${dev_id} \
    --queryset ${queryset} \
    --budget ${budget}
$ python online/adversary/train.py ${out_dir} ${f_v} ${p_v} \
    --budget 500,5000,20000,60000 \
    --log-interval 500 \
    --epochs 50 \
    -d ${dev_id}

python online/adversary/transfer.py random models/victim/MNIST-lenet-train-BCG bcg ,out_path:models/final_bb_dist/MNIST-lenet-bcg-EMNISTLetters-B50000-proxy_scratch-random \
    --out_dir models/final_bb_dist/MNIST-lenet-bcg-EMNISTLetters-B50000-proxy_scratch-random \
    --batch_size 1 \
    -d 3 \
    --queryset EMNISTLetters \
    --budget 50000 
   
python online/adversary/train.py models/final_bb_dist/MNIST-lenet-bcg-EMNISTLetters-B50000-proxy_scratch-random lenet MNIST \
    --budget 50000 \
    --log-interval 500 \
    --epochs 50 \
    -d 0   	
```

####  JBDA

First, in `online/adversary/jacobian.py`

```
model_adv = model_utils.train_model(model_adv, self.D, self.out_dir, num_workers=10,
                          checkpoint_suffix='.{}'.format(self.blackbox.call_count),
                          device=self.device, restored=False, task=self.testset_name,
                          epochs=self.final_train_epochs,
                          log_interval=500, lr=0.01, momentum=0.9, batch_size=self.batch_size,
                          lr_gamma=0.1, testset=self.testset_name,
                          criterion_train=model_utils.soft_cross_entropy)                         
```

Note: change `restored=False` to `restored=True`.

Second, in `online/utils/model.py`

```
  if task == 'MNIST':
        Model, input_size, class_num, inp_mean, inp_std, is_discrete = load_model_setting("mnist")
        generator_path = ${Generator}
        meta_path = ${Meta-classifer}
# Generator stored in /generator/${task}${defense}
# Meta-classifier stored in /meta/${defense}${task}
```

```
$ python online/adversary/jacobian.py jbda ${victim_dir} ${strat} ${defense_args} \
    --model_adv ${f_v} \
    --out_dir ${out_dir} 
    --testset ${p_v} \
    --budget 500,5000,20000,60000 \
    --queryset ${queryset} \
    -d 1
    
python online/adversary/jacobian.py jbda models/victim/MNIST-lenet-train-BCG bcg ,out_path:models/adversary/MNIST-lenet-jbda-bcg-EMNISTLetters-B50000 \
    --model_adv lenet \
    --out_dir models/adversary/MNIST-lenet-jbda-bcg-EMNISTLetters-B50000 \
    --testset MNIST \
    --budget 4800 \
    --queryset MNIST \
    -d 2 \
    --seedsize 150               
```



## Credits

Parts of this repository have been adapted from

https://github.com/sanjaykariyappa/adaptive_misinformation/

https://anonymous.4open.science/r/7bf97390fc6d3dc663ca8c9d657746/

