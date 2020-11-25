# echo expirement 1
# echo learning rate cos
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.1 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.05 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.005 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.0005 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.0001 SOLVER.IMS_PER_BATCH 64
# echo learning warmup epoch
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 2 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 4 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 6 SOLVER.IMS_PER_BATCH 64
# # best
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 8 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 10 SOLVER.IMS_PER_BATCH 64
# echo epoch
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 40 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 50 SOLVER.IMS_PER_BATCH 64
# # best
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 60 SOLVER.IMS_PER_BATCH 64
# echo center learning rate
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.1 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.2 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.3 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.4 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.5 SOLVER.IMS_PER_BATCH 64
# echo learning rate warmup
# python main.py -c se_resnext.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.1 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.05 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.01 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.005 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.001 SOLVER.IMS_PER_BATCH 64
# echo POOLING_METHOD
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 MODEL.POOLING_METHOD GeM SOLVER.IMS_PER_BATCH 64
# echo AUG
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.4 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.3 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.2 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.1 SOLVER.IMS_PER_BATCH 64
# # best
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG simple SOLVER.IMS_PER_BATCH 64
# # best
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.05 SOLVER.IMS_PER_BATCH 64
# python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.1 SOLVER.IMS_PER_BATCH 64

echo expirement 2 tmp best
python main.py -c se_resnext.yaml SOLVER.BASE_LR 0.04 SOLVER.IMS_PER_BATCH 64 SOLVER.WARMUP_EPOCHS 8 SOLVER.MAX_EPOCHS 60 DATASETS.HARD_AUG simple INPUT.GRAY_RPO 0.05