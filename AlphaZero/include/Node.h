#pragma once

#include <vector>
#include <unordered_map>
#include <limits>
#include <cassert>

//State must, at least, implement GenericState functions
//Action(int) must exist, as well as (int)(Action) to transform an Action into a index in NN output
template <class State, typename Action>
class Node
{
public:
	/**
	* \brief Create a new node 
	* \param state_ The state of this node
	* \param proba The prior probability for the parent to choose this node
	* \param parent_ A pointer on the parent
	*/
	Node(const State &state_, const float &proba = 1.0f, Node<State, Action> *parent_ = nullptr);

	~Node();

	/**
	* \brief Expand the tree from this node by creating the corresponding children
	* \param action_and_probas Possible actions and their probabilities
	*/
	void Expand(const std::unordered_map<Action, float> &action_and_probas);

	/**
	* \brief Select the action among children to maximize Q + c * P * U
	* \param c Exploration constant
	* \param next_node Filled with the pointer to the next node
	* \return The selected action to perform
	*/
	Action Select(const float &c, Node<State, Action> **next_node);

	/**
	* \brief Update the values and apply it to its parent
	* \param value Reward (from player 1 perspective) to propagate
	*/
	void Update(const float &value);

	/**
	* \brief Check if this node is a leaf in the tree
	* \return True if the node has no children, false otherwise
	*/
	bool IsLeaf();

	/**
	* \brief Check if this node is the root of the tree
	* \return True if the node is the root, false otherwise
	*/
	bool IsRoot();

	/**
	* \brief Return the state corresponding to this node
	* \return The state of this node
	*/
	State const GetState() const;

	/**
	* \brief Get the visit count for each possible action from this node
	* \return A map of visit count for each possible action
	*/
	std::unordered_map<Action, int> GetActionVisitCount();

	/**
	* \brief Remove a child from the list. Make sure to save the pointer to the child before to avoid memory leak.
	* \param a The action we want to remove the child from
	*/
	void RemoveChild(const Action &a);

	/**
	* \brief Get a child from the list
	* \param a The action we want to get the child
	* \return A pointer on the child (or nullptr if it doesn't exist)
	*/
	Node<State, Action>* GetChild(const Action &a);

	/**
	* \brief Set the parent of this node
	* \param p The new parent of the node
	*/
	void SetParent(Node<State, Action> *p);

	/**
	* \brief Add noise to the children prior probabilities
	* \param noise The noise vector, should be at least of size children.size()
	* \param epsilon Noise weight (P = P * (1-eps) + eps * noise)
	*/
	void AddNoiseToChildren(const std::vector<float> &noise, const float &epsilon);

	/**
	* \brief Get the number of children of the node
	* \return The number of children
	*/
	int GetNumberChildren();

private:
	/**
	* \brief Return the exploration value of this node (P * U)
	* \return The exploration value of this node
	*/
	float GetExplorationValue();

	/**
	* \brief Return the exploitation value of this node (Q)
	* \return The exploitation value of this node
	*/
	float GetExploitationValue();

private:
	State state;
	Node<State, Action> *parent;

	int N;
	float W;
	float Q;
	float P;

	std::unordered_map<Action, Node<State, Action>* > children;
};

template<class State, typename Action>
Node<State, Action>::Node(const State &state_, const float &proba, Node<State, Action> *parent_)
{
	state = state_;
	parent = parent_;

	N = 0;
	W = 0.0f;
	Q = 0.0f;
	P = proba;
}

template<class State, typename Action>
Node<State, Action>::~Node()
{
	for (auto it = children.begin(); it != children.end(); ++it)
	{
		delete it->second;
	}
}

template<class State, typename Action>
void Node<State, Action>::Expand(const std::unordered_map<Action, float> &action_and_probas)
{
	for (auto it = action_and_probas.begin(); it != action_and_probas.end(); ++it)
	{
		if (children.find(it->first) == children.end())
		{
			State new_state = state;
			new_state.Act(it->first);
			children.insert(std::make_pair(it->first, new Node<State, Action>(new_state, it->second, this)));
		}
	}
}

template<class State, typename Action>
Action Node<State, Action>::Select(const float &c, Node<State, Action> **next_node)
{
	float max_value = -std::numeric_limits<float>::infinity();
	Action max_action;

	for (auto it = children.begin(); it != children.end(); ++it)
	{
		float current_exploration_value = it->second->GetExplorationValue();
		float current_exploitation_value = it->second->GetExploitationValue();
		
		//As we store the value from player 1 perspective, we have to invert it if -1 plays
		if (state.GetCurrentPlayer() != 1)
		{
			current_exploitation_value = -current_exploitation_value;
		}

		float current_value = current_exploitation_value + c * current_exploration_value;

		if (current_value > max_value)
		{
			max_value = current_value;
			max_action = it->first;
			*next_node = it->second;
		}
	}

	return max_action;
}

template<class State, typename Action>
void Node<State, Action>::Update(const float &value)
{
	if (!IsRoot())
	{
		parent->Update(value);
	}

	N += 1;
	W += value;
	Q = W / N;
}

template<class State, typename Action>
bool Node<State, Action>::IsLeaf()
{
	return (children.size() == 0);
}

template<class State, typename Action>
bool Node<State, Action>::IsRoot()
{
	return (parent == nullptr);
}

template<class State, typename Action>
State const Node<State, Action>::GetState() const
{
	return state;
}

template<class State, typename Action>
std::unordered_map<Action, int> Node<State, Action>::GetActionVisitCount()
{
	std::unordered_map<Action, int> output;
	for (auto it = children.begin(); it != children.end(); ++it)
	{
		Action action = it->first;
		int visit_count = it->second->N;
		output.insert(std::make_pair(action, visit_count));
	}
	return output;
}

template<class State, typename Action>
void Node<State, Action>::RemoveChild(const Action &a)
{
	children.erase(a);
}

template<class State, typename Action>
Node<State, Action>* Node<State, Action>::GetChild(const Action &a)
{
	auto it = children.find(a);

	if (it != children.end())
	{
		return it->second;
	}

	return nullptr;
}

template<class State, typename Action>
void Node<State, Action>::SetParent(Node<State, Action> *p)
{
	parent = p;
}

template<class State, typename Action>
float Node<State, Action>::GetExplorationValue()
{
	if (IsRoot())
	{
		return 0.0f;
	}

	return P * std::sqrtf(parent->N) / (1.0f + N);
}

template<class State, typename Action>
float Node<State, Action>::GetExploitationValue()
{
	return Q;
}

template<class State, typename Action>
void Node<State, Action>::AddNoiseToChildren(const std::vector<float> &noise, const float &epsilon)
{
	assert(noise.size() >= children.size());

	int counter = 0;

	for (auto it = children.begin(); it != children.end(); ++it)
	{
		it->second->P = it->second->P * (1.0f - epsilon) + noise[counter] * epsilon;
		counter++;
	}
}

template<class State, typename Action>
int Node<State, Action>::GetNumberChildren()
{
	return children.size();
}
