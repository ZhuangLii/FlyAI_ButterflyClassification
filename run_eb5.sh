echo learning rate cos
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.1 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.05 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.005 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.0005 
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.0001 
echo learning warmup epoch
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 2
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 4
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 6
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 8
# best
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.WARMUP_EPOCHS 10
echo epoch
# best
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 40
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 50
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.01 SOLVER.MAX_EPOCHS 60
echo center learning rate
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.1
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.2
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.3
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.4
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 CENTER_LR 0.5
echo learning rate warmup
python main.py -c efficientnetb5.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.1 
python main.py -c efficientnetb5.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.05 
python main.py -c efficientnetb5.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.01 
python main.py -c efficientnetb5.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.005 
python main.py -c efficientnetb5.yaml SOLVER.TYPE warmup SOLVER.BASE_LR 0.001 
echo POOLING_METHOD
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 MODEL.POOLING_METHOD GeM
echo AUG
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.4
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.3
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.2
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG hard INPUT.RE_PROB 0.1
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 DATASETS.HARD_AUG simple
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.05
python main.py -c efficientnetb5.yaml SOLVER.BASE_LR 0.001 INPUT.GRAY_RPO 0.1

