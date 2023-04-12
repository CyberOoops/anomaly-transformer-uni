python main.py --anormly_ratio 0.6 --num_epochs 10   --batch_size 256  --mode train --dataset Yahoo  --data_path data/Yahoo  --input_c 1 --output_c 1
python main.py --anormly_ratio 0.6 --num_epochs 10   --batch_size 256     --mode test    --dataset Yahoo   --data_path data/Yahoo    --input_c 1  --output_c 1   --pretrained_model 20

python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256  --mode train --dataset AIOPS  --data_path data/AIOPS  --input_c 1 --output_c 1
python main.py --anormly_ratio 1 --num_epochs 10   --batch_size 256     --mode test    --dataset AIOPS   --data_path data/AIOPS    --input_c 1  --output_c 1   --pretrained_model 20

python main.py --anormly_ratio 0.9 --num_epochs 10   --batch_size 256  --mode train --dataset WSD  --data_path data/WSD  --input_c 1 --output_c 1
python main.py --anormly_ratio 0.9 --num_epochs 10   --batch_size 256     --mode test    --dataset WSD   --data_path data/WSD    --input_c 1  --output_c 1   --pretrained_model 20

python main.py --anormly_ratio 0.9 --num_epochs 10   --batch_size 256  --mode train --dataset NAB  --data_path data/NAB  --input_c 1 --output_c 1
python main.py --anormly_ratio 0.9 --num_epochs 10   --batch_size 256     --mode test    --dataset NAB   --data_path data/NAB    --input_c 1  --output_c 1   --pretrained_model 20