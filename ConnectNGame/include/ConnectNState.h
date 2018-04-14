#pragma once

#include <unordered_map>
#include <functional>

template <typename Action>
class ConnectNState
{
public:
	/**
	* \brief Default constructor. Return a starting state waiting for player 1 action.
	* \return The default state
	*/
	ConnectNState();
	~ConnectNState();

	/**
	* \brief Check if the current state has a winner
	* \return The winner (-1 or 1/0 for a tie/-2 if there is no winner)
	*/
	int Winner() const;

	/**
	* \brief Apply the action on the current state and transform it
	* \param action The action to apply
	*/
	void Act(const Action &action);

	/**
	* \brief Return a vector of possible actions for the current player
	* \return A vector with all possible actions
	*/
	std::vector<Action> GetPossibleActions() const;

	/**
	* \brief Converts the state into a representation for the neural network
	* \return A vector of float representing the state
	*/
	std::vector<float> ToNNInput() const;

	/**
	* \brief Get the current player
	* \return The current player index (-1 or 1)
	*/
	int GetCurrentPlayer() const;

	/**
	* \brief Convert the state into a string for pretty display
	* \return A string representing the current state
	*/
	std::string ToString() const;

	/**
	* \brief Ask for the user an action to perform
	* \return A valid action
	*/
	Action AskInput() const;

	/**
	* \brief Apply all possible symmetries on the input tuples and return the resulting tuples
	* \param s The tuple with the state, the probabilities for each action and the value
	* \return All the tuples with the symmetrized states, probabilities for each action and values (at least the input)
	*/
	static std::vector<std::tuple<ConnectNState<Action>, std::unordered_map<Action, float>, int> > GetSymmetry(const std::tuple<ConnectNState<Action>, std::unordered_map<Action, float>, int> &s);

	/**
	* \brief Change the board_width parameter for the whole class, do not change in the middle of training
	* \param board_width_ The new board width
	*/
	static void SetBoardWidth(const int &board_width_);

	/**
	* \brief Change the board_height parameter for the whole class, do not change in the middle of training
	* \param board_height_ The new board width
	*/
	static void SetBoardHeight(const int &board_height_);

	/**
	* \brief Change the aligned_to_win parameter for the whole class, do not change in the middle of training
	* \param aligned_to_win_ The new aligned to win value
	*/
	static void SetAlignedToWin(const int &aligned_to_win_);

protected:
	std::vector<int> board;
	int current_player;

	static int board_height;
	static int board_width;
	static int aligned_to_win;
};

template <typename Action>
int ConnectNState<Action>::board_height = 6;

template <typename Action>
int ConnectNState<Action>::board_width = 7;

template <typename Action>
int ConnectNState<Action>::aligned_to_win = 4;

template <typename Action>
ConnectNState<Action>::ConnectNState()
{
	board = std::vector<int>(board_height * board_width, 0);
	current_player = 1;
}

template <typename Action>
ConnectNState<Action>::~ConnectNState()
{

}

template <typename Action>
int ConnectNState<Action>::Winner() const
{
	std::function<int (int)> Sign = [](int x){return (0 < x) - (x < 0); };
	
	//Loop through every rows
	//and check at the same time if 
	//the board is filled
	bool is_filled = true;
	for (int i = 0; i < board_height; ++i)
	{
		//Sum the values along the current row, reset every time we meet a 0 or the player change
		int partial_sum_row = 0;
		for (int j = 0; j < board_width; ++j)
		{
			if (board[board_width * i + j] == Sign(partial_sum_row))
			{
				partial_sum_row += board[board_width * i + j];
				if (std::abs(partial_sum_row) >= aligned_to_win)
				{
					return Sign(partial_sum_row);
				}
			}
			else
			{
				partial_sum_row = board[board_width * i + j];
			}

			//Filled
			if (board[board_width * i + j] == 0)
			{
				is_filled = false;
			}
		}
	}

	//Loop through every columns
	for (int i = 0; i < board_width; ++i)
	{
		//Sum the values along the current col, reset every time we meet a 0 or the player change
		int partial_sum_col = 0;
		for (int j = 0; j < board_height; ++j)
		{
			if (board[board_width * j + i] == Sign(partial_sum_col))
			{
				partial_sum_col += board[board_width * j + i];
				if (std::abs(partial_sum_col) >= aligned_to_win)
				{
					return Sign(partial_sum_col);
				}
			}
			else
			{
				partial_sum_col = board[board_width * j + i];
			}
		}
	}

	//For the diagonals it's a bit more tricky
	for (int i = -board_width + aligned_to_win; i <= board_height - aligned_to_win; ++i)
	{
		int partial_sum_diag = 0;
		int partial_sum_antidiag= 0;
		for (int j = std::max(0, -i); j < std::min(board_height, board_width - i); ++j)
		{
			if (board[board_width * j + j + i] == Sign(partial_sum_diag))
			{
				partial_sum_diag += board[board_width * j + j + i];
				if (std::abs(partial_sum_diag) >= aligned_to_win)
				{
					return Sign(partial_sum_diag);
				}
			}
			else
			{
				partial_sum_diag = board[board_width * j + j + i];
			}

			if (board[board_width * j + board_width - 1 - (j + i)] == Sign(partial_sum_antidiag))
			{
				partial_sum_antidiag += board[board_width * j + board_width - 1 - (j + i)];
				if (std::abs(partial_sum_antidiag) >= aligned_to_win)
				{
					return Sign(partial_sum_antidiag);
				}
			}
			else
			{
				partial_sum_antidiag = board[board_width * j + board_width - 1 - (j + i)];
			}
		}
	}

	if (is_filled)
	{
		return 0;
	}

	return -2;
}

template <typename Action>
void ConnectNState<Action>::Act(const Action &action)
{
	//Check the first line to know if the column is filled or not
	if (board[(int)(action)] != 0)
	{
		std::cerr << "Error, column (" << (int)(action) << ") is filled" << std::endl;
		return;
	}

	if ((int)(action) < 0 || (int)(action) > board_width - 1)
	{
		std::cerr << "Error, impossible action (" << (int)(action) << ")" << std::endl;
		return;
	}

	//Look for the last free position in the column
	int row = 1;
	while (row < board_height)
	{
		if (board[row * board_width + (int)(action)] != 0)
		{
			break;
		}
		row++;
	}
	board[(row - 1) * board_width + (int)(action)] = current_player;
	current_player = -current_player;
}

template <typename Action>
std::vector<Action> ConnectNState<Action>::GetPossibleActions() const
{
	std::vector<Action> possible_actions;
	possible_actions.reserve(board_width);

	//One action is chosing one column
	for (int i = 0; i < board_width; ++i)
	{
		if (board[i] == 0)
		{
			possible_actions.push_back(Action(i));
		}
	}

	return possible_actions;
}

template <typename Action>
std::vector<float> ConnectNState<Action>::ToNNInput() const
{
	std::vector<float> output;
	output.reserve(3 * board.size());

	//Player 1 channel
	for (int i = 0; i < board.size(); ++i)
	{
		if (board[i] == 1)
		{
			output.push_back(1.0f);
		}
		else
		{
			output.push_back(0.0f);
		}
	}

	//Player 2 channel
	for (int i = 0; i < board.size(); ++i)
	{
		if (board[i] == -1)
		{
			output.push_back(1.0f);
		}
		else
		{
			output.push_back(0.0f);
		}
	}

	//Current player channel
	for (int i = 0; i < board.size(); ++i)
	{
		if (current_player == 1)
		{
			output.push_back(1.0f);
		}
		else
		{
			output.push_back(0.0f);
		}
	}

	return output;
}

template <typename Action>
int ConnectNState<Action>::GetCurrentPlayer() const
{
	return current_player;
}

template <typename Action>
std::string ConnectNState<Action>::ToString() const
{
	std::string output;

	output += "-----\n";
	output += "| ";
	if (current_player == 1)
	{
		output += "x";
	}
	else
	{
		output += "o";
	}
	output += " |\n";

	for (int i = 0; i < board_width * 3 + 2; ++i)
	{
		output += "-";
	}
	output += "\n";

	for (int i = 0; i < board_height; ++i)
	{
		output += "|";
		for (int j = 0; j < board_width; ++j)
		{
			if (board[board_width * i + j] == -1)
			{
				output += " o ";
			}
			else if (board[board_width * i + j] == 1)
			{
				output += " x ";
			}
			else
			{
				output += "   ";
			}
		}
		output += "|\n";
	}
	for (int i = 0; i < board_width * 3 + 2; ++i)
	{
		output += "-";
	}
	output += "\n";

	return output;
}

template<typename Action>
std::vector<std::tuple<ConnectNState<Action>, std::unordered_map<Action, float>, int> > ConnectNState<Action>::GetSymmetry(const std::tuple<ConnectNState<Action>, std::unordered_map<Action, float>, int> &s)
{
	std::vector<std::tuple<ConnectNState<Action>, std::unordered_map<Action, float>, int> > output;

	//TODO implement symmetry
	output.push_back(s);

	return output;
}

template<typename Action>
void ConnectNState<Action>::SetBoardHeight(const int &board_height_)
{
	board_height = board_height_;
}

template<typename Action>
void ConnectNState<Action>::SetBoardWidth(const int &board_width_)
{
	board_width = board_width_;
}

template<typename Action>
void ConnectNState<Action>::SetAlignedToWin(const int &aligned_to_win_)
{
	aligned_to_win = aligned_to_win_;
}

template<typename Action>
Action ConnectNState<Action>::AskInput() const
{
	int col = 0;
	bool is_possible = false;
	Action selected_action;
	std::vector<Action> possible_actions = GetPossibleActions();
	while (!is_possible)
	{
		std::cout << ToString() << std::endl;
		std::cout << "Column? ";
		std::cin >> col;
		std::cout << std::endl;

		selected_action = Action(col);

		for (int i = 0; i < possible_actions.size(); ++i)
		{
			if (possible_actions[i] == selected_action)
			{
				is_possible = true;
				break;
			}
		}

		if (!is_possible)
		{
			std::cout << "You selected an impossible action... Try again" << std::endl;
		}
	}

	return selected_action;
}