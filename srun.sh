srun --gres=gpu:1 --mem=12G -p gpu --unbuffered  python main.py --dataset ring --cuda --train_iterations 800 --batch_size 16 --sampling_method random --log_name batch_size_1 --out_path random
