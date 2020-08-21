# Distilling Image Dehazing With Heterogeneous Task Imitation
This repository contains code for reproducing the paper ["Distilling Image Dehazing With Heterogeneous Task Imitation"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hong_Distilling_Image_Dehazing_With_Heterogeneous_Task_Imitation_CVPR_2020_paper.pdf)

# Get Started


# Teacher Training

python src/train_teacher.py --model-name teacher --batch-size 4 --epochs 30 --teacher-lr 0.0001 --train-ratio 0.8 --validation-ratio 0.1 --gt-dir clear --hazy-dir Haze

# Student Training

python src/train.py --dataset OTS --model-name teacher student --batch-size 4 --epochs 200 --teacher-lr 0.0001 --train-ratio 0.8 --val-ratio 0.1 --gt-dir clear --hazy-dir Haze --load-best-teacher-model --best-teacher-model-path=model_path.pth