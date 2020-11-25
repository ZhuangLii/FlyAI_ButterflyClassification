# echo learning rate cos
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.1 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.05 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 MODEL.DEVICE_ID '1'
# # best lr 0.008
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.005 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.0005 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.0001 MODEL.DEVICE_ID '1'
# echo learning warmup epoch
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 2 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 4 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 6 MODEL.DEVICE_ID '1'
# # best warm up 8 epoch
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 8 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 10 MODEL.DEVICE_ID '1'
# echo epoch

# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 40 MODEL.DEVICE_ID '1'
# # best 45 epcoh
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 50 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 60 MODEL.DEVICE_ID '1'
# echo center learning rate
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.1 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.2 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.3 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.4 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.5 MODEL.DEVICE_ID '1'
# echo learning rate warmup
# python main.py -c resnest.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.1 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.05 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.01 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.005 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.001 MODEL.DEVICE_ID '1'
# echo POOLING_METHOD
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 MODEL.POOLING_METHOD GeM MODEL.DEVICE_ID '1'
# echo AUG
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.4 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.3 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.2 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.1 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG simple MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.05 MODEL.DEVICE_ID '1'
# python main.py -c resnest.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.1 MODEL.DEVICE_ID '1'

# try gem auto simple hard
python main.py -c resnest.yaml SOLVER.BASE_LR 0.008 SOLVER.WARMUP_EPOCHS 8  SOLVER.MAX_EPOCHS 45 INPUT.GRAY_RPO 0.05 DATASETS.HARD_AUG simple