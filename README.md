# Flow Field for Edge Detection

# Train, using BSDS500 dataset as an example
python main.py --model ffed --config calv --sa --dil --only-bsds --resume --iter-size 24 -j 4 --gpu 0 --epochs 16 --lr 0.005 --lr-type multistep --lr-steps 10-16 --wd 1e-4 --datadir /path/to/BSDS500 --dataset BSDS

## Generating edge maps for your own images
```bash
python main.py --model --config calv --sa --dil -j 4 --gpu 0 --datadir /path/to/custom_images --dataset Custom --evaluate /path/to/checkpointxxx.pth 
```
## Reference
When building our code, we referenced the repositories as follow:
PidiNet:https://github.com/hellozhuo/pidinet
XYWNet:https://github.com/PXinTao/XYW-Net
