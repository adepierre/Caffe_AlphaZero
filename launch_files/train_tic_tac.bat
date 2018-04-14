SET PATH=%PATH%;D:/Libraries/Caffe/install/bin

"../bin/Release/TicTacGame.exe" ^
--train=1 ^
--board_size=6 ^
--aligned_to_win=4 ^
--num_playouts=400 ^
--c_puct=3.0 ^
--num_test=25 ^
--solver_file=solver.prototxt ^
--batch_size=128 ^
--alpha=0.3 ^
--max_size_memory=10000 ^
--num_game=10000 ^
--log_file=log.csv ^
--epsilon=0.25 ^
--snapshot=