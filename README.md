# deepcluster-styletransfer
personal project for Undergraduate research program

train the encoder in /deep_cluster using: \
python mian.py dataroot \
--arch=vgg16 \
--nmb_cluster=600 \
--exp=feature \
--verbose \
--epoch=150 

train the decoder and discriminator in /reconstruction using: \
python train_vggrec.py --dataroot=... \
--dataroot_style=... \
--encoder=...pth.tar  \
--name=exp_name \
(save training result with order: --save_image --image_dir=... --serial_batches)
