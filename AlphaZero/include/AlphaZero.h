#pragma once

#include <caffe/caffe.hpp>

#include <unordered_map>
#include <deque>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>

#include "MCTS.h"

template <class State, typename Action>
class AlphaZero
{
public:
	/**
	* \brief Construct one AlphaZero object for training
	* \param solver_file Caffe solver file to define the NN parameters
	* \param snapshot Caffe snapshot file (*.solverstate) to restart an interrupted training
	* \param num_playouts_ Number of playouts used by the MCTS
	* \param c_puct_ Exploration constant used by the MCTS
	* \param alpha_ Dirichlet noise parameter
	* \param epsilon_ Dirichlet noise weight
	* \param max_buffer_size_ Size of the replay memory (older plays are discarded)
	* \param batch_size_ Size of the batch to use
	* \param log_file_ File into the losses are logged during training
	*/
	AlphaZero(const std::string &solver_file,
			  const std::string &snapshot,
			  const int num_playouts_,
			  const float c_puct_,
			  const float alpha_,
			  const float epsilon_,
			  const int max_buffer_size_,
			  const int batch_size_,
			  const std::string &log_file_);
	
	/**
	* \brief Construct one AlphaZero object for testing
	* \param net_model Model file for the net (*.prototxt)
	* \param trained_weights Weights learned during training to load in the net (*.caffemodel)
	* \param num_playouts_ Number of playouts used by MCTS before selecting an action
	* \param c_puct_ Exploration constant used by MCTS (Q + c_puct_ * P * U)
	*/
	AlphaZero(const std::string &net_model,
			  const std::string &trained_weights,
			  const int num_playouts_,
			  const float c_puct_);

	~AlphaZero();

	/**
	* \brief Perform one global training step : self play, train and then test the performance
	* \param num_game_to_play Number of self-played games performed before returning
	* \param num_test Number of game to play to evaluate the network performance
	*/
	void Train(const int num_game_to_play, const int num_test);

	/**
	* \brief Perform one game IA vs Human and return the winner
	* \param N Number of game
	* \return A vector with the winners, 1 for human, 0 for draw game, -1 for algorithm
	*/
	std::vector<int> HumanTest(const int N);
	
	/**
	* \brief Perform N games playing both players
	* \param N number of games
	* \param display Whether or not we want to display the intermediate states
	* \return A vector with the N winners (-1,0,1)
	*/
	std::vector<int> SelfTest(const int N, const bool display = false);

	/**
	* \brief Perform N games against a random player
	* \param N number of games
	* \param display Whether or not we want to display the intermediate states
	* \return A vector with the N winners, 1 for AlphaZero, 0 for a draw, -1 for the random player
	*/
	std::vector<int> RandomTest(const int N, const bool display = false);

	/**
	* \brief Perform N game between the current net and a MCTS algorithm with random playouts instead of NN evaluation
	* \param N Number of game played (N/2 games playing first, N/2 playing second)
	* \param opponent_num_playouts Number of playouts for the opponent
	* \param display Whether to display the intermediate states during testing
	* \param net The network to use (if not main net)
	* \return A vector with the N winners (1 for the net, -1 for the opponent, 0 for a draw)
	*/
	std::vector<int> MCTSTest(const int N, const int opponent_num_playouts, const bool display = false, boost::shared_ptr<caffe::Net<float> > net = boost::shared_ptr<caffe::Net<float> >());

	/**
	* \brief Perform N games between two networks to compare them
	* \param N Number of game played (N/2 games playing first, N/2 playing second)
	* \param net_model Model file to build the network (.prototxt)
	* \param trained_weights Caffemodel file to load into the opponent net
	* \param display Whether to display the intermediate states during testing
	* \return A vector with the N winners (1 for the main net, -1 for the opponent net, 0 for a draw)
	*/
	std::vector<int> CompareNets(const int N, const std::string &net_model, const std::string &trained_weights, bool display = false);

	/**
	* \brief Perform N games between two networks to compare them
	* \param N Number of game played (N/2 games playing first, N/2 playing second)
	* \param net_ Model to compare with the current net
	* \param display Whether to display the intermediate states during testing
	* \return A vector with the N winners (1 for the main net, -1 for the opponent net, 0 for a draw)
	*/
	std::vector<int> CompareNets(const int N, boost::shared_ptr<caffe::Net<float> > net_, const bool display = false);

	/**
	* \brief Return the prediction of the net given the input state
	* \param network A pointer of the used network for evaluation
	* \param s The state to evaluate
	* \return A pair with the evaluated value [-1, 1] of the current state from the player's perspective and the probability for each action from this state
	*/
	static std::pair<float, std::unordered_map<Action, float> > StateEvaluation(boost::shared_ptr<caffe::Net<float> > network, const State &s);

	/**
	* \brief Return a value from random play and uniform probabilities over the actions
	* \param s The state to evaluate
	* \return A pair with the evaluated value {-1, 0, 1} of the current state from the player's perspective and the probability for each action from this state
	*/
	static std::pair<float, std::unordered_map<Action, float> > StateEvaluationRandomRollout(const State &s);

private:

	/**
	* \Perform one snapshot of the current training state
	*/
	void Snapshot();

	/**
	* \brief Simulate one game
	* \param net The network to use for state evaluation
	* \return A vector of tuple for each successive moves : State of the game, map of action/probabilities, end winner
	*/
	std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > SelfPlay(boost::shared_ptr<caffe::Net<float> > net = boost::shared_ptr<caffe::Net<float> >());

	/**
	* \brief Play games to fill the replay buffer
	* \param N number of played games
	*/
	void FillBuffer(const int N);

	/**
	* \brief Test the current parameters against a pure MCTS with random playouts
	* \param num_game Number of games played for testing
	*/
	void Test(const int num_game);

	/**
	* \brief Train the network from self play samples
	*/
	void TrainNet();

	/**
	* \brief Reshape a network with a new batch size
	* \param net A shared pointer on the network
	* \param new_batch_size The new batch size we want to set
	*/
	void ReshapeNet(const boost::shared_ptr<caffe::Net<float> > net, const int new_batch_size);
	
private:
	boost::shared_ptr<caffe::Solver<float> > solver;
	boost::shared_ptr<caffe::Net<float> > main_net;


	boost::shared_ptr<caffe::Net<float> > test_net;
	int test_weights_iter;

	std::mt19937 random_engine;
	
	//The last saved training weights
	caffe::NetParameter last_weights;
	int last_weights_iter;

	//Parameters used for testing during training
	int opponent_playouts;
	int num_win;
	int num_draw;
	int num_opponent_win;

	//Used to save the past games
	std::deque<std::tuple<State, std::unordered_map<Action, float>, int> > replay_buffer;
	int max_buffer_size;
	int num_game_played;
	
	//Algorithm parameters
	float alpha;
	float epsilon;
	int num_playouts;
	float c_puct;
	int batch_size;

	//Log stuffs
	std::ofstream log_file;
	float loss_value;
	float loss_probas;
	float kl_div;
	int iter_since_log;
};

template<class State, typename Action>
AlphaZero<State, Action>::AlphaZero(const std::string &solver_file,
									const std::string &snapshot,
									const int num_playouts_,
									const float c_puct_,
									const float alpha_,
									const float epsilon_,
									const int max_buffer_size_,
									const int batch_size_,
									const std::string &log_file_)
{
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	num_playouts = num_playouts_;
	c_puct = c_puct_;
	alpha = alpha_;
	epsilon = epsilon_;

	batch_size = batch_size_;

	max_buffer_size = max_buffer_size_;
	num_game_played = 0;

	loss_value = 0.0f;
	loss_probas = 0.0f;
	kl_div = 0.0f;
	iter_since_log = 0;

	caffe::SolverParameter solver_param;
	caffe::ReadProtoFromTextFileOrDie(solver_file, &solver_param);

	solver.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	main_net = solver->net();

	ReshapeNet(main_net, batch_size);

	if (snapshot.empty())
	{
		if (!log_file_.empty())
		{
			log_file.open(log_file_);
			log_file << "Iter;Num Game played;Value loss;Probas loss;KL div;Opponent playouts;Win;Draw;Lost" << std::endl;
		}
	}
	else
	{
		solver->Restore(snapshot.c_str());
		
		if (!log_file_.empty())
		{
			log_file.open(log_file_, std::ofstream::app);
		}
	}


	main_net->ToProto(&last_weights, false);
	last_weights_iter = solver->iter();

	//Create the test network
	test_net.reset(new caffe::Net<float>(solver->param().net(), caffe::Phase::TEST));
	
	ReshapeNet(test_net, 1);
	test_weights_iter = last_weights_iter;
	test_net->CopyTrainedLayersFrom(last_weights);

	opponent_playouts = num_playouts_;
	num_win = 0;
	num_opponent_win = 0;
	num_draw = 0;

	random_engine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

template<class State, typename Action>
AlphaZero<State, Action>::AlphaZero(const std::string &net_model,
									const std::string &trained_weights,
									const int num_playouts_,
									const float c_puct_)
{
#ifdef CPU_ONLY
	caffe::Caffe::set_mode(caffe::Caffe::CPU);
#else
	caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

	num_playouts = num_playouts_;
	c_puct = c_puct_;
	epsilon = 0.0f;

	main_net.reset(new caffe::Net<float>(net_model, caffe::Phase::TEST));
	ReshapeNet(main_net, 1);
	
	if (!trained_weights.empty())
	{
		main_net->CopyTrainedLayersFromBinaryProto(trained_weights);
	}

	random_engine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

template<class State, typename Action>
AlphaZero<State, Action>::~AlphaZero()
{
}

template<class State, typename Action>
void AlphaZero<State, Action>::Train(const int num_game_to_play, const int num_test)
{
	int display_frequency = solver->param().display();
	int test_frequency = std::max(1, 2 * solver->param().snapshot() / num_test);
	int snapshot_frequency = solver->param().snapshot();

	//Train the network
	while (num_game_played < num_game_to_play)
	{
		//Play to fill the memory buffer
		if (solver->iter() % 10 == 0 || (5 * batch_size > replay_buffer.size() && replay_buffer.size() < max_buffer_size))
		{
			FillBuffer(1);
		}

		TrainNet();

		//Stop the training and test the network
		//against a pure MCTS algorithm
		if (solver->iter() % test_frequency == 0 && solver->iter() > 0)
		{
			ReshapeNet(main_net, 1);
			Test(2);
			ReshapeNet(main_net, batch_size);
		}

		//Every display iterations save the net weights
		if (solver->iter() % display_frequency == 0 && (solver->iter() > 0 || solver->iter() == 1))
		{
			main_net->ToProto(&last_weights, false);
			last_weights_iter = solver->iter();
		}

		if (solver->iter() % snapshot_frequency == 0 && solver->iter() > 0)
		{
			if (log_file.is_open())
			{
				log_file << solver->iter() << ";" << num_game_played << ";;;;" << opponent_playouts << ";" << num_win << ";" << num_draw << ";" << num_opponent_win << std::endl;
			}

			int total_played_period = num_win + num_draw + num_opponent_win;
			std::cout << "On the last period, current network is";
			if ((num_win + 0.5 * num_draw) / (float)(total_played_period) >= 0.55f)
			{
				std::cout << " better than ";
			}
			else if ((num_win + 0.5 * num_draw) / (float)(total_played_period) >= 0.45f)
			{
				std::cout << " as good as ";
			}
			else
			{
				std::cout << " weaker than ";
			}
			std::cout << "the opponent with " << opponent_playouts << " playouts (" << num_win << "/" << num_draw << "/" << num_opponent_win << ")" << std::endl;

			//If the net won almost every games, get a stronger opponent
			if ((num_win + 0.5 * num_draw) / (float)(total_played_period) >= 0.9f && opponent_playouts < 10 * num_playouts)
			{
				opponent_playouts += num_playouts;
				std::cout << "It was a bit too easy. Increasing the opponent strength with " << num_playouts << " supplementary playouts (total: " << opponent_playouts << ")" << std::endl;
			}

			num_win = 0;
			num_draw = 0;
			num_opponent_win = 0;
		}
	}

	Snapshot();
}

template<class State, typename Action>
void AlphaZero<State, Action>::Snapshot()
{
	if (solver)
	{
		solver->Snapshot();
	}
}

template<class State, typename Action>
std::pair<float, std::unordered_map<Action, float> > AlphaZero<State, Action>::StateEvaluation(boost::shared_ptr<caffe::Net<float> > network, const State &s)
{
	//Net prediction
	std::vector<float> nn_input = s.ToNNInput();
	caffe::caffe_copy(nn_input.size(), nn_input.data(), network->blob_by_name("input_data")->mutable_cpu_data());
	network->Forward();

	std::pair<float, std::unordered_map<Action, float> > output;

	//Get prediction on the game outcome from current player's perspective
	output.first = network->blob_by_name("output_value")->cpu_data()[0];

	//Get back probabilities predictions
	boost::shared_ptr<caffe::Blob<float> > output_proba = network->blob_by_name("output_probas");
	std::vector<float> predicted_proba = std::vector<float>(output_proba->count() / output_proba->num(), 0.0f);
	caffe::caffe_copy(predicted_proba.size(), output_proba->cpu_data(), predicted_proba.data());

	//Keep only the predictions for the possible actions
	std::vector<Action> possible_actions = s.GetPossibleActions();
	float sum_probas = 0.0f;

	for (int j = 0; j < possible_actions.size(); ++j)
	{
		output.second.insert(std::make_pair(possible_actions[j], predicted_proba[(int)(possible_actions[j])]));
		sum_probas += predicted_proba[(int)(possible_actions[j])];
	}

	//Re-normalize the probabilities so the sum is 1
	for (auto it = output.second.begin(); it != output.second.end(); ++it)
	{
		it->second /= sum_probas + 0.000001f;
	}

	return output;
}

template<class State, typename Action>
std::pair<float, std::unordered_map<Action, float> > AlphaZero<State, Action>::StateEvaluationRandomRollout(const State &s)
{
	std::pair<float, std::unordered_map<Action, float> > output;

	//Set uniform playing probabilities 
	std::vector<Action> possible_actions = s.GetPossibleActions();
	for (int i = 0; i < possible_actions.size(); ++i)
	{
		output.second.insert(std::make_pair(possible_actions[i], 1.0f / possible_actions.size()));
	}

	//Estimate the value with a random game from the current state
	State current_state = s;

	std::mt19937 randomizer(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	while (current_state.Winner() == -2)
	{
		possible_actions = current_state.GetPossibleActions();
		int index_action = std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(randomizer);

		current_state.Act(possible_actions[index_action]);
	}

	//Get the value from the current player's perspective
	output.first = current_state.Winner() * s.GetCurrentPlayer();

	return output;
}

template<class State, typename Action>
std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > AlphaZero<State, Action>::SelfPlay(boost::shared_ptr<caffe::Net<float> > net)
{
	if (!net)
	{
		net = main_net;
	}

	std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > output;
	
	State current_state;

	//If the net has never been trained before, play with a MCTS without net evaluation
	std::function<std::pair<float, std::unordered_map<Action, float> >(State)> function;
	if (solver && solver->iter() == 0)
	{
		function = std::bind(&AlphaZero<State, Action>::StateEvaluationRandomRollout, std::placeholders::_1);
	}
	else
	{
		function = std::bind(&AlphaZero<State, Action>::StateEvaluation, net, std::placeholders::_1);
	}
	MCTS<State, Action> play_tree(function, current_state, num_playouts, -1, c_puct, epsilon, alpha);
	
	float temperature = 1.0f;
	int winner = current_state.Winner();

	while (winner == -2)
	{
		//Simulate N games in the tree and get the visit count from the root state
		std::unordered_map<Action, int> visit_count = play_tree.GetActionVisitCount();

		//Transform the visit count into probabilities
		std::unordered_map<Action, float> probabilities;

		float max_proba = 0.0f;
		for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
		{
			float proba = 1.0f / temperature * std::log(it->second + 1e-6);//+epsilon because we don't want to have a 0 inside the log in case of unvisited node
			if (proba > max_proba)
			{
				max_proba = proba;
			}

			probabilities.insert(std::make_pair(it->first, proba));
		}

		float sum_proba = 0.0f;
		for (auto it = probabilities.begin(); it != probabilities.end(); ++it)
		{
			it->second = std::exp(it->second - max_proba);
			sum_proba += it->second;
		}

		for (auto it = probabilities.begin(); it != probabilities.end(); ++it)
		{
			it->second /= sum_proba;
		}

		std::vector<Action> actions;
		std::vector<float> probas;
		for (auto it = probabilities.begin(); it != probabilities.end(); ++it)
		{
			probas.push_back(it->second);
			actions.push_back(it->first);
		}

		//Select one action according to the probabilities distribution
#if(_MSC_VER != 1800)
		std::discrete_distribution<int> distrib(probas.begin(), probas.end());
		Action selected_action = actions[distrib(random_engine)];
#else
		//Bug with discrete distribution with visual 2013, so select manually
		//https://connect.microsoft.com/VisualStudio/feedback/details/976256/discrete-distribution-inputiterator-first-inputiterator-last-couldnt-compile
		int action_index = 0;

		float random_value = std::uniform_real_distribution<float>(0.0f, 1.0f)(random_engine);
		for (int k = 0; k < probas.size(); ++k)
		{
			if (random_value < probas[k])
			{
				action_index = k;
				break;
			}
			random_value -= probas[k];
		}
		Action selected_action = actions[action_index];
#endif

		play_tree.TreeStep(selected_action);

		output.push_back(std::make_tuple(current_state, probabilities, -2));
		current_state.Act(selected_action);
		winner = current_state.Winner();
	}

	//Add the winner to every saved tuple
	for (int i = 0; i < output.size(); ++i)
	{
		std::get<2>(output[i]) = winner;
	}

	return output;
}

template<class State, typename Action>
void AlphaZero<State, Action>::FillBuffer(const int N)
{
	//Load the weights if needed
	if (test_weights_iter != last_weights_iter)
	{
		test_net->CopyTrainedLayersFrom(last_weights);
		test_weights_iter = last_weights_iter;
	}

	//Play N games
	for (int n = 0; n < N; ++n)
	{
		std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > replay = SelfPlay(test_net);
		std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > augmented_replay;
		for (int j = 0; j < replay.size(); ++j)
		{
			std::vector<std::tuple<State, std::unordered_map<Action, float>, int> > local_symmetrized_replay = State::GetSymmetry(replay[j]);
			augmented_replay.insert(augmented_replay.end(), local_symmetrized_replay.begin(), local_symmetrized_replay.end());
		}

		for (int i = 0; i < augmented_replay.size(); ++i)
		{
			replay_buffer.push_back(augmented_replay[i]);
		}
		while (replay_buffer.size() > max_buffer_size)
		{
			replay_buffer.pop_front();
		}
		num_game_played++;
	}
}

template<class State, typename Action>
void AlphaZero<State, Action>::Test(const int N)
{
	//Perform the test
	std::vector<int> winners = MCTSTest(N, opponent_playouts);

	//Process the results
	for (int i = 0; i < winners.size(); ++i)
	{
		switch (winners[i])
		{
		case 1:
			num_win++;
			break;
		case 0:
			num_draw++;
			break;
		case -1:
			num_opponent_win++;
			break;
		default:
			break;
		}
	}
}

template<class State, typename Action>
void AlphaZero<State, Action>::TrainNet()
{
	//Do not train while the memory is not full enough
	if (5 * batch_size > replay_buffer.size() && replay_buffer.size() < max_buffer_size)
	{
		return;
	}

	boost::shared_ptr<caffe::Blob<float> > blob_state = main_net->blob_by_name("input_data");
	boost::shared_ptr<caffe::Blob<float> > blob_value = main_net->blob_by_name("label_value");
	boost::shared_ptr<caffe::Blob<float> > blob_probas = main_net->blob_by_name("label_probas");
	boost::shared_ptr<caffe::Blob<float> > blob_loss_value = main_net->blob_by_name("value_loss");
	boost::shared_ptr<caffe::Blob<float> > blob_loss_probas = main_net->blob_by_name("probas_loss");

	int number_of_actions = blob_probas->count() / batch_size;
	float batch_entropy = 0.0f;

	for (int i = 0; i < batch_size; ++i)
	{
		std::tuple<State, std::unordered_map<Action, float>, int> current_data = replay_buffer[std::uniform_int_distribution<int>(0, replay_buffer.size() - 1)(random_engine)];
		
		//State input
		State state = std::get<0>(current_data);
		std::vector<float> state_input = state.ToNNInput();
		caffe::caffe_copy(state_input.size(), state_input.data(), blob_state->mutable_cpu_data() + blob_state->offset(i));

		//Value label
		int winner = std::get<2>(current_data);
		float value_label = 0.0f;
		if (winner == 0)
		{
			value_label = 0.0f;
		}
		else
		{
			value_label = winner == state.GetCurrentPlayer() ? 1.0f : -1.0f;
		}
		*(blob_value->mutable_cpu_data() + blob_value->offset(i)) = value_label;

		//Probabilities
		std::vector<float> action_proba(number_of_actions, 0.0f);
		for (auto it = std::get<1>(current_data).begin(); it != std::get<1>(current_data).end(); ++it)
		{
			action_proba[(int)(it->first)] = it->second;
			batch_entropy += -it->second * std::log(it->second);
		}
		caffe::caffe_copy(action_proba.size(), action_proba.data(), blob_probas->mutable_cpu_data() + blob_probas->offset(i));
	}

	solver->Step(1);

	loss_value += blob_loss_value->cpu_data()[0];
	loss_probas += blob_loss_probas->cpu_data()[0];
	kl_div += blob_loss_probas->cpu_data()[0] - batch_entropy / batch_size;
	iter_since_log++;

	if (log_file.is_open() && solver->iter() % solver->param().display() == 0)
	{
		log_file << solver->iter() << ";" << num_game_played << ";" << loss_value / iter_since_log << ";" << loss_probas / iter_since_log << ";" << kl_div / iter_since_log << ";;;;" << std::endl;
		iter_since_log = 0;
		loss_value = 0.0f;
		loss_probas = 0.0f;
		kl_div = 0.0f;
	}

	//Perform a custom snapshot to be sure to save all useful variables
	if (solver->iter() % solver->param().snapshot() == 0)
	{
		Snapshot();
	}
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::MCTSTest(const int N, const int opponent_num_playouts, const bool display, boost::shared_ptr<caffe::Net<float> > net)
{
	if (!net)
	{
		net = main_net;
	}

	//Play N games
	std::vector<int> winners;

	for (int i = 0; i < N; ++i)
	{
		int opponent_index = 1;

		//Every player starts half of the games
		if (i % 2 == 0)
		{
			opponent_index = -1;
		}

		if (display)
		{
			std::cout << "AlphaZero is player " << -opponent_index << std::endl;
		}

		State current_state;

		int winner = current_state.Winner();
		int current_player = current_state.GetCurrentPlayer();

		MCTS<State, Action> play_tree_opponent(std::bind(&AlphaZero<State, Action>::StateEvaluationRandomRollout, std::placeholders::_1), current_state, opponent_num_playouts, -1, c_puct, 0.0f);
		MCTS<State, Action> play_tree_current(std::bind(&AlphaZero<State, Action>::StateEvaluation, net, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);

		//Play the first action randomly to have a bit of diversity in the games
		std::vector<Action> possible_actions = current_state.GetPossibleActions();
		Action random_action = possible_actions[std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(random_engine)];
		current_state.Act(random_action);
		play_tree_opponent.TreeStep(random_action);
		play_tree_current.TreeStep(random_action);

		if (display)
		{
			std::cout << current_state.ToString() << std::endl;
		}

		while (winner == -2)
		{
			std::unordered_map<Action, int> visit_count;

			//Simulate N games in the tree of the first player and get the visit count from the root state
			if (opponent_index == current_player)
			{
				visit_count = play_tree_opponent.GetActionVisitCount();
			}
			else
			{
				visit_count = play_tree_current.GetActionVisitCount();
			}

			//Select the action with the max visit count
			Action selected_action;
			int max_visit_count = 0;

			for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
			{
				if (it->second > max_visit_count)
				{
					max_visit_count = it->second;
					selected_action = it->first;
				}
			}
			current_state.Act(selected_action);

			winner = current_state.Winner();
			current_player = current_state.GetCurrentPlayer();

			play_tree_opponent.TreeStep(selected_action);
			play_tree_current.TreeStep(selected_action);

			if (display)
			{
				std::cout << current_state.ToString() << std::endl;
			}
		}

		if (display)
		{
			if (winner == 0)
			{
				std::cout << "Nobody wins..." << std::endl;
			}
			else if (opponent_index == winner)
			{
				std::cout << "It's a victory for the opponent" << std::endl;
			}
			else
			{
				std::cout << "AlphaZero wins" << std::endl;
			}
		}

		//Depending on who is player 1, add the winner (1 for alphazero, -1 for opponent)
		if (opponent_index == 1)
		{
			winners.push_back(-winner);
		}
		else
		{
			winners.push_back(winner);
		}
	}

	return winners;
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::HumanTest(const int N)
{
	std::vector<int> output;

	for (int i = 0; i < N; ++i)
	{
		int human_index = (i % 2) == 0 ? 1 : -1;

		State current_state;
		int winner = current_state.Winner();
		int current_player = current_state.GetCurrentPlayer();

		MCTS<State, Action> play_tree(std::bind(&AlphaZero<State, Action>::StateEvaluation, main_net, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);

		while (winner == -2)
		{
			Action selected_action;

			if (current_player == human_index)
			{
				selected_action = current_state.AskInput();
			}
			else
			{
				std::unordered_map<Action, int> visit_count = play_tree.GetActionVisitCount();
				int max_visit_count = 0;

				for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
				{
					if (it->second > max_visit_count)
					{
						max_visit_count = it->second;
						selected_action = it->first;
					}
				}
			}
			current_state.Act(selected_action);

			winner = current_state.Winner();
			current_player = current_state.GetCurrentPlayer();

			play_tree.TreeStep(selected_action);
		}

		if (human_index == winner)
		{
			std::cout << "Congratulations, you defeated the machine" << std::endl;
		}
		else if (winner == 0)
		{
			std::cout << "At least it can't beat you" << std::endl;
		}
		else
		{
			std::cout << "Maybe next time" << std::endl;
		}

		output.push_back(winner * human_index);
	}
	return output;
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::SelfTest(const int N, const bool display)
{
	std::vector<int> output;

	for (int i = 0; i < N; ++i)
	{
		State current_state;
		int winner = current_state.Winner();
		int current_player = current_state.GetCurrentPlayer();

		MCTS<State, Action> play_tree(std::bind(&AlphaZero<State, Action>::StateEvaluation, main_net, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);

		//Select first action randomly because the algorithm is sadly deterministic
		std::vector<Action> possible_actions = current_state.GetPossibleActions();
		int action_index = std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(random_engine);
		current_state.Act(possible_actions[action_index]);
		play_tree.TreeStep(possible_actions[action_index]);

		if (display)
		{
			std::cout << current_state.ToString() << std::endl;
		}

		while (winner == -2)
		{
			Action selected_action;
			std::unordered_map<Action, int> visit_count = play_tree.GetActionVisitCount();
			int max_visit_count = 0;

			for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
			{
				if (it->second > max_visit_count)
				{
					max_visit_count = it->second;
					selected_action = it->first;
				}
			}
			current_state.Act(selected_action);

			winner = current_state.Winner();
			current_player = current_state.GetCurrentPlayer();

			play_tree.TreeStep(selected_action);
			if (display)
			{
				std::cout << current_state.ToString() << std::endl;
			}
		}

		output.push_back(winner);

		if (display)
		{
			if (winner == 1)
			{
				std::cout << "Player 1 wins!" << std::endl;
			}
			else if (winner == -1)
			{
				std::cout << "Player 2 wins!" << std::endl;
			}
			else
			{
				std::cout << "No winner ..." << std::endl;
			}
		}
	}
	return output;
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::RandomTest(const int N, const bool display)
{
	//Play N games
	std::vector<int> winners;

	for (int i = 0; i < N; ++i)
	{
		int opponent_index = 1;

		//Every player starts half of the games
		if (i % 2 == 0)
		{
			opponent_index = -1;
		}

		if (display)
		{
			std::cout << "AlphaZero is player " << -opponent_index << std::endl;
		}

		State current_state;

		int winner = current_state.Winner();
		int current_player = current_state.GetCurrentPlayer();

		MCTS<State, Action> play_tree_current(std::bind(&AlphaZero<State, Action>::StateEvaluation, main_net, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);

		//Play the first action randomly to have a bit of diversity in the games
		std::vector<Action> possible_actions = current_state.GetPossibleActions();
		Action random_action = possible_actions[std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(random_engine)];
		current_state.Act(random_action);
		play_tree_current.TreeStep(random_action);

		if (display)
		{
			std::cout << current_state.ToString() << std::endl;
		}

		while (winner == -2)
		{
			Action selected_action;
			//Simulate N games in the tree of the first player and get the visit count from the root state
			if (opponent_index == current_player)
			{
				std::vector<Action> possible_actions = current_state.GetPossibleActions();
				selected_action = possible_actions[std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(random_engine)];
			}
			else
			{
				std::unordered_map<Action, int> visit_count;
				visit_count = play_tree_current.GetActionVisitCount();
				//Select the action with the max visit count

				int max_visit_count = 0;
				for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
				{
					if (it->second > max_visit_count)
					{
						max_visit_count = it->second;
						selected_action = it->first;
					}
				}
			}
			current_state.Act(selected_action);

			winner = current_state.Winner();
			current_player = current_state.GetCurrentPlayer();

			play_tree_current.TreeStep(selected_action);

			if (display)
			{
				std::cout << current_state.ToString() << std::endl;
			}
		}

		if (display)
		{
			if (winner == 0)
			{
				std::cout << "Nobody wins..." << std::endl;
			}
			else if (opponent_index == winner)
			{
				std::cout << "It's a victory for the opponent" << std::endl;
			}
			else
			{
				std::cout << "AlphaZero wins" << std::endl;
			}
		}

		//Depending on who is player 1, add the winner (1 for alphazero, -1 for opponent)
		if (opponent_index == 1)
		{
			winners.push_back(-winner);
		}
		else
		{
			winners.push_back(winner);
		}
	}

	return winners;
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::CompareNets(const int N, const std::string &net_model, const std::string &trained_weights, bool display)
{
	//Play N games
	std::vector<int> winners;

	boost::shared_ptr<caffe::Net<float> > opponent_net;
	opponent_net.reset(new caffe::Net<float>(net_model, caffe::Phase::TEST));

	if (!trained_weights.empty())
	{
		opponent_net->CopyTrainedLayersFromBinaryProto(trained_weights);
	}

	return CompareNets(N, opponent_net, display);
}

template<class State, typename Action>
std::vector<int> AlphaZero<State, Action>::CompareNets(const int N, boost::shared_ptr<caffe::Net<float> > net_, const bool display)
{
	//Play N games
	std::vector<int> winners;
	
	for (int i = 0; i < N; ++i)
	{
		int opponent_index = 1;

		//Every player starts half of the games
		if (i % 2 == 0)
		{
			opponent_index = -1;
		}

		if (display)
		{
			std::cout << "Main alpha zero is player " << -opponent_index << std::endl;
		}

		State current_state;

		int winner = current_state.Winner();
		int current_player = current_state.GetCurrentPlayer();

		MCTS<State, Action> play_tree_opponent(std::bind(&AlphaZero<State, Action>::StateEvaluation, net_, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);
		MCTS<State, Action> play_tree_current(std::bind(&AlphaZero<State, Action>::StateEvaluation, main_net, std::placeholders::_1), current_state, num_playouts, -1, c_puct, 0.0f);

		//Play the first action randomly to have a bit of diversity in the games
		std::vector<Action> possible_actions = current_state.GetPossibleActions();
		Action random_action = possible_actions[std::uniform_int_distribution<int>(0, possible_actions.size() - 1)(random_engine)];
		current_state.Act(random_action);
		play_tree_opponent.TreeStep(random_action);
		play_tree_current.TreeStep(random_action);

		if (display)
		{
			std::cout << current_state.ToString() << std::endl;
		}

		while (winner == -2)
		{
			std::unordered_map<Action, int> visit_count;

			//Simulate N games in the tree of the first player and get the visit count from the root state
			if (opponent_index == current_player)
			{
				visit_count = play_tree_opponent.GetActionVisitCount();
			}
			else
			{
				visit_count = play_tree_current.GetActionVisitCount();
			}

			//Select the action with the max visit count
			Action selected_action;
			int max_visit_count = 0;

			for (auto it = visit_count.begin(); it != visit_count.end(); ++it)
			{
				if (it->second > max_visit_count)
				{
					max_visit_count = it->second;
					selected_action = it->first;
				}
			}
			current_state.Act(selected_action);

			winner = current_state.Winner();
			current_player = current_state.GetCurrentPlayer();

			play_tree_opponent.TreeStep(selected_action);
			play_tree_current.TreeStep(selected_action);

			if (display)
			{
				std::cout << current_state.ToString() << std::endl;
			}
		}

		if (display)
		{
			if (winner == 0)
			{
				std::cout << "Nobody wins..." << std::endl;
			}
			else if (opponent_index == winner)
			{
				std::cout << "It's a victory for the opponent" << std::endl;
			}
			else
			{
				std::cout << "Main Alpha Zero wins" << std::endl;
			}
		}

		//Depending on who is player 1, add the winner (1 for main alphazero, -1 for opponent)
		if (opponent_index == 1)
		{
			winners.push_back(-winner);
		}
		else
		{
			winners.push_back(winner);
		}
	}

	return winners;
}

template<class State, typename Action>
void AlphaZero<State, Action>::ReshapeNet(const boost::shared_ptr<caffe::Net<float> > net, const int new_batch_size)
{
	if (!net)
	{
		return;
	}

	std::vector<int> shape;
	boost::shared_ptr<caffe::Blob<float> > blob = net->blob_by_name("input_data");
	if (blob)
	{
		shape = blob->shape();
		shape[0] = new_batch_size;
		blob->Reshape(shape);
	}

	blob = net->blob_by_name("label_value");
	if (blob)
	{
		shape = blob->shape();
		shape[0] = new_batch_size;
		blob->Reshape(shape);
	}

	blob = net->blob_by_name("label_probas");
	if (blob)
	{
		shape = blob->shape();
		shape[0] = new_batch_size;
		blob->Reshape(shape);
	}

	boost::shared_ptr<caffe::Layer<float> > layer = net->layer_by_name("Cross_Entropy_Scale");
	if (layer)
	{
		layer->blobs()[0]->mutable_cpu_data()[0] = 1.0f / new_batch_size;
	}

	net->Reshape();
}