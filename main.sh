##### SingleLabel
DATASET=dtd
METHOD=UCAT-SingleLabel
GPU=7
ROOT=/datasets/

ATTACK=pgd
TARGETCON=0.07

#------TeCoA
TRAINEPS=1
TRAINSTEPS=2
TESTEPS=1
TESTSTEPS=100

# #------FARE
# TRAINEPS=2
# TRAINSTEPS=10
# TESTEPS=2
# TESTSTEPS=100

BATCHSIZE=256
SAVEDIR=./results/

################################# Adversarial FT
python -u ./main.py --testdata ${DATASET} \
            --Method ${METHOD} --gpu $GPU --root ${ROOT} \
            --attack ${ATTACK} --target_concentration ${TARGETCON} \
            --train_eps ${TRAINEPS} --train_numsteps ${TRAINSTEPS} \
            --test_eps ${TESTEPS} --test_numsteps ${TESTSTEPS} \
            --batch_size ${BATCHSIZE}  --save_dir ${SAVEDIR}


################################# Adversarial Inference (Single Label)
RESUMEDIR=/work1/wenjing/1_CLIP_Uncertainty/results/ceedl/ce-kladv2clean/model/ce-kladv2clean_tinyImageNet_clip_vit_b32_lr-0.0001_decay-0_bsz-256_warmup-1000_trial-1_Alpha-0.08_Beta-0.05_distance-l2_atten_methods-text-new/model_best.pth.tar

for DATASET in dtd fgvc_aircraft oxfordpet cifar10 flowers102 STL10 Caltech101 tinyImageNet cifar100 StanfordCars PCAM EuroSAT Food101 Caltech256 ImageNet SUN397 
do 
for ATTACK in pgd CW autoattack CAA a3
do
python -u ./main.py --testdata ${DATASET} \
            --Method ${METHOD} --gpu ${GPU} \
            --attack ${ATTACK} --target_concentration ${TARGETCON} \
            --test_eps ${TESTEPS} --test_numsteps ${TESTSTEPS} \
            --batch_size ${BATCHSIZE} --save_dir ${SAVEDIR} \
            --mode test --resume ${RESUMEDIR}
done
done

################################# Adversarial Inference (Single Label)
DATASET=coco2017
for ATTACK in CW pgd
do 
for TESTEPS in 1 2 4
do
python -u ./main-multilabel.py --testdata ${DATASET} \
            --Method ${METHOD} --gpu ${GPU} \
            --attack ${ATTACK} --target_concentration ${TARGETCON} \
            --test_eps ${TESTEPS} --test_numsteps ${TESTSTEPS} \
            --batch_size ${BATCHSIZE} --dataset tinyImageNet \
            --save_dir ${SAVEDIR} \
            --mode test --resume ${RESUMEDIR}
done
done
