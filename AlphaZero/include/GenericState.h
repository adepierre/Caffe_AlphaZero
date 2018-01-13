#include <vector>
#include <unordered_map>

//Action can be just an int or a custom
//struct/class following this example
struct ExampleAction
{
	ExampleAction()
	{
		action_index = 0;
	}

	ExampleAction(const int a)
	{
		action_index = a;
	}
	operator int() const { return action_index; }

	bool operator==(const ExampleAction &other) const
	{
		return (action_index == other.action_index);
	}

	int action_index;
};

namespace std
{
	template<>
	struct hash<ExampleAction>
	{
		std::size_t operator()(const ExampleAction &a) const
		{
			return std::hash<int>{}(a.action_index);
		}
	};
}

template <typename Action>
class GenericState
{
public:
	/**
	* \brief Default constructor. Return an empty state waiting for player 1 action.
	* \return The default state, with current player = 1
	*/
	GenericState();

	virtual ~GenericState() = 0;

	/**
	* \brief Check if the current state has a winner
	* \return The winner (-1 or 1/0 for a draw/-2 if there is no winner)
	*/
	virtual int Winner() const;

	/**
	* \brief Apply the action on the current state and transform it
	* \param action The action to apply
	*/
	virtual void Act(const Action &action);

	/**
	* \brief Return a vector of possible actions for the current player
	* \return A vector with all possible actions
	*/
	virtual std::vector<Action> GetPossibleActions() const;

	/**
	* \brief Converts the state into a representation for the neural network
	* \return A vector of float representing the state
	*/
	virtual std::vector<float> ToNNInput() const;

	/**
	* \brief Get the current player
	* \return The current player index (-1 or 1)
	*/
	virtual int GetCurrentPlayer() const;
	
	/**
	* \brief Apply all possible symmetries on the input tuples and return the resulting tuples
	* \param s The tuple with the state, the probabilities for each action and the value
	* \return All the tuples with the symmetrized states, probabilities for each action and values (at least the input)
	*/
	static std::vector<std::tuple<GenericState<Action>, std::unordered_map<Action, float>, int> > GetSymmetry(const std::tuple<IState<Action>, std::unordered_map<Action, float>, int> &s);

	/**
	* \brief Convert the state into a string for pretty display
	* \return A string representing the current state
	*/
	virtual std::string ToString() const;

	/**
	* \brief Ask for the user an action to perform
	* \return A valid action
	*/
	virtual Action AskInput() const;
};