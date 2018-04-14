#pragma once

#include "Node.h"

#include <functional>
#include <vector>
#include <unordered_map>
#include <random>
#include <chrono>

//State must, at least, implement GenericState's functions
//Action(int) must exist, as well as (int)(Action) to transform an Action into a index in NN output
template <class State, typename Action>
class MCTS
{

public:
	/**
	* \brief Construct a MCTS object to perform evaluations
	* \param f Evaluation function : takes a state as input and returns both a score from the current player perspective and the estimated probabilities for every possible actions
	* \param init_sate Initial state of the root
	* \param n_simulation_ Number of iteration before action selection (-1 to use time limit)
	* \param n_milliseconds_ Number of milliseconds to think the next move (-1 to use iteration limit)
	* \param c_puct_ Exploration constant (Q + c_puct_ * P * U) 
	* \param epsilon_ Noise weight (P = P * (1 - eps) + noise * eps)
	* \param alpha_ Noise parameter (noise = Dirichlet(alpha))
	*/
	MCTS(const std::function<std::pair<float, std::unordered_map<Action, float> >(State)> &f,
		 const State &init_state,
		 const int n_simulation_,
		 const int n_milliseconds_,
		 const float c_puct_,
		 const float epsilon_,
		 const float alpha_ = 0.3f);

	~MCTS();

	/**
	* \brief Run n_simulation from the root and returns the probabilities for each possible action
	* \return A map with the visit count for each action
	*/
	std::unordered_map<Action, int> GetActionVisitCount();


	/**
	* \brief Perform one step in the tree executing the given action. The subtree of the new root is kept, the other are discarded
	* \param action Action to execute
	*/
	void TreeStep(const Action &a);

private:
	/**
	* \brief Run an iteration from the root. Find the first leaf, expand it, and backpropagate its value to the top.
	*/
	void Iteration();


	/**
	* \brief Add Dirichlet noise to the probabilities of the children of the root
	*/
	void AddDirichletToRoot();

	
private:
	int n_simulation;
	int n_milliseconds;
	float c_puct;
	float epsilon;
	float alpha;
	std::function<std::pair<float, std::unordered_map<Action, float> >(State)> EvaluationFunction;

	std::mt19937 random_engine;

	Node<State, Action> *root;
};


template<class State, typename Action>
MCTS<State, Action>::MCTS(const std::function<std::pair<float, std::unordered_map<Action, float> >(State)> &f,
						  const State &init_state,
						  const int n_simulation_,
						  const int n_milliseconds_,
						  const float c_puct_,
						  const float epsilon_,
						  const float alpha_)
{
	EvaluationFunction = f;
	n_simulation = n_simulation_;
	n_milliseconds = n_milliseconds_;
	c_puct = c_puct_;
	epsilon = epsilon_;
	alpha = alpha_;

	if (n_simulation < 0 && n_milliseconds < 0)
	{
		n_milliseconds = 1000;
	}
	
	random_engine = std::mt19937(std::chrono::high_resolution_clock::now().time_since_epoch().count());

	root = new Node<State, Action>(init_state, 1.0f, nullptr);

	//Estimate the value and the probabilities for all the possible actions
	std::pair<float, std::unordered_map<Action, float> > value_and_probas = EvaluationFunction(init_state);

	root->Expand(value_and_probas.second);
	AddDirichletToRoot();
}

template<class State, typename Action>
MCTS<State, Action>::~MCTS()
{
	if (root != nullptr)
	{
		delete root;
	}
}

template<class State, typename Action>
std::unordered_map<Action, int> MCTS<State, Action>::GetActionVisitCount()
{
	if (n_simulation >= 0)
	{
		//Perform n_simulation from the current root
		for (int i = 0; i < n_simulation; ++i)
		{
			Iteration();
		}
	}
	else
	{
		auto begin = std::chrono::high_resolution_clock::now();
		auto now = std::chrono::high_resolution_clock::now();
		while (std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() < n_milliseconds)
		{
			Iteration();
			now = std::chrono::high_resolution_clock::now();
		}
	}
	
	//Get the probabilities for each action from the visit count of the child
	std::unordered_map<Action, int> visit_count = root->GetActionVisitCount();

	return visit_count;
}

template<class State, typename Action>
void MCTS<State, Action>::TreeStep(const Action &a)
{
	if (root == nullptr)
	{
		return;
	}

	Node<State, Action> *new_root = root->GetChild(a);

	//If the root hasn't the corresponding child, create it
	if (new_root == nullptr)
	{
		State next_state = root->GetState();
		next_state.Act(a);
		new_root = new Node<State, Action>(next_state, 1.0f, nullptr);
	}

	new_root->SetParent(nullptr);
	root->RemoveChild(a);
	delete root;

	root = new_root;
	
	std::pair<float, std::unordered_map<Action, float> > value_and_probas = EvaluationFunction(root->GetState());
	root->Expand(value_and_probas.second);
	AddDirichletToRoot();
}

template<class State, typename Action>
void MCTS<State, Action>::Iteration()
{
	if (root == nullptr)
	{
		return;
	}

	Node<State, Action> *node = root;

	//Go to the first leaf
	while (!node->IsLeaf())
	{
		node->Select(c_puct, &node);
	}

	//Estimate the value and the probabilities for all the possible actions
	std::pair<float, std::unordered_map<Action, float> > value_and_probas = EvaluationFunction(node->GetState());

	int winner = node->GetState().Winner();

	//Not a terminal state
	if (winner == -2)
	{
		node->Expand(value_and_probas.second);
	}
	//Terminal state
	else
	{
		//Draw game
		if (winner == 0)
		{
			value_and_probas.first = 0.0f;
		}
		//We have a winner !
		//As the value is stored from player 1 perspective, it is just the player index
		else
		{
			value_and_probas.first = winner;
		}
	}

	//Backpropagate the value through the ancestors
	node->Update(value_and_probas.first);
}

template<class State, typename Action>
void MCTS<State, Action>::AddDirichletToRoot()
{
	std::vector<float> dirichlet_noise = std::vector<float>(root->GetNumberChildren(), 0.0f);

	float sum = 0.0f;
	std::gamma_distribution<float> distribution(alpha, 1.0f);

	for (int i = 0; i < dirichlet_noise.size(); ++i)
	{
		dirichlet_noise[i] = distribution(random_engine);
		sum += dirichlet_noise[i];
	}

	for (int i = 0; i < dirichlet_noise.size(); ++i)
	{
		dirichlet_noise[i] /= sum + 0.0000001f;
	}

	root->AddNoiseToChildren(dirichlet_noise, epsilon);
}