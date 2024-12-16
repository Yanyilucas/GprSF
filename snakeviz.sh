# 使用snakeviz对cProfile的结果进行可视化
python -m cProfile -o log.profile data_presenter.py
snakeviz log.profile