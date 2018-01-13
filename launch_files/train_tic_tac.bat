SET PATH=%PATH%;D:/Libraries/Caffe/install/bin

"../bin/Release/TicTacGame.exe" ^
--train=1 ^
--board_size=6 ^
--aligned_to_win=4 ^
--num_playouts=400 ^
--c_puct=3.0 ^
--solver_file=solver.prototxt ^
--alpha=0.3 ^
--epsilon=0.25 ^
--max_size_memory=10000 ^
--num_epoch=150 ^
--num_game_per_epoch=50 ^
--num_train_step=5 ^
--num_test=25 ^
--log_file=log.csv ^
--snapshot=