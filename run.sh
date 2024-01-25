python main.py --train_data ./data/imdb_synthetic.txt --test_data ./data/imdb_job-light.txt --db imdb --save_model_name imdb_synthetic --load_model_name imdb_synthetic --leaf_num 10
python main.py --test --load_model --test_data ./data/imdb_scale.txt --db imdb --load_model_name imdb_synthetic
