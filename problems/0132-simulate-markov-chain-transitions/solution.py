import numpy as np

def simulate_markov_chain(transition_matrix: np.ndarray, initial_state: int, num_steps: int) -> list[int]:
    """
    Simulate a Markov Chain and return the sequence of states.
    """
    # 1. Initialize the sequence with the starting state
    current_state = initial_state
    state_sequence = [current_state]
    
    # 2. Determine the total number of possible states (number of rows/cols)
    num_states = transition_matrix.shape[0]
    
    # 3. Step through the simulation num_steps times
    for _ in range(num_steps):
        # Extract the transition probabilities for the current state
        probabilities = transition_matrix[current_state]
        
        # 4. Randomly pick the next state based on those probabilities
        # np.random.choice(n) automatically chooses from [0, 1, ..., n-1]
        next_state = np.random.choice(num_states, p=probabilities)
        
        # 5. Record the new state and update our current position
        state_sequence.append