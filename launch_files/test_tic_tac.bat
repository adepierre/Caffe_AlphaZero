SET PATH=%PATH%;D:/Libraries/Caffe/install/bin

"../bin/Release/TicTacGame.exe" ^
--train=0 ^
--board_size=6 ^
--aligned_to_win=4 ^
--num_playouts=400 ^
--c_puct=3.0 ^
--num_test=25 ^
--model_file=net_tic_tac_6_4.prototxt ^
--model_file2=net_tic_tac_6_4.prototxt ^
--trained_weights=net_tic_tac_6_4_24000.caffemodel ^
--trained_weights2=net_tic_tac_6_4_24000.caffemodel ^
--opponent_playouts=1000 ^
--test_mode=Compare ^
--display=0