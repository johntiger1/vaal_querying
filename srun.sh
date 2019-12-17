srun --gres=gpu:1 --mem=16G -p gpu --unbuffered  python main.py --dataset ring --cuda --train_iterations 800 --batch_size 1 --sampling_method expected_error --log_name bs16 --out_path set2/test_1
