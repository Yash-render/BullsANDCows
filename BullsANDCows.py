# Implementing the Bulls and Cows game with a suggestion system based on information theory.
# The game generates a 4-digit secret number with unique digits, and the player has to guess it within a limited number of attempts.
# After each guess, the player receives feedback in terms of "bulls" and "cows" to help narrow down the possible secret numbers.
# The suggestion system uses information theory to suggest the optimal guess that maximizes the expected information gain.
# The game also displays statistics, remaining attempts, and a visualization of entropy and mutual information over the course of the game.



# Importing required libraries
import streamlit as st
import random
import itertools
import math
import pandas as pd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# --- Game Logic ---

# Generating 4 digit secret number with unique digits
def generate_secret_number() -> str: # Generate a 4-digit secret number with unique digits
    digits = list(range(10))
    random.shuffle(digits) # Shuffle the digits
    return "".join(map(str, digits[:4])) # Return the first 4 digits as a string


# Comparing the guess with the secret number and returning the number of bulls and cows
def compare_guess(secret_number: str, guess: str) -> Tuple[int, int]:
    bulls = sum(1 for i in range(4) if guess[i] == secret_number[i])
    cows = sum(1 for char in guess if char in secret_number) - bulls # Count the number of cows by subtracting the bulls because cows should not be counted twice
    return bulls, cows


# Checking if the guess is valid (4-digit number with unique digits)
def is_valid_guess(guess: str) -> bool:
    return guess.isdigit() and len(guess) == 4 and len(set(guess)) == 4
    # len(set(guess)) == 4 checks if all digits are unique as set() removes duplicates


# Calculating the entropy of a list of possible secrets: measure of uncertainty or randomness
def calculate_entropy(possible_secrets: List[str]) -> float:
    if not possible_secrets: # If no possible numbers, return 0.0
        return 0.0
    num_secrets = len(possible_secrets) # Number of possible secrets
    return -math.log2(1 / num_secrets) # Return the entropy value: formula: -Σ p(x) * log2(p(x))

    # Assuming uniform distribution, because all secrets are equally likely
    # so, the entropy formula -Σ p(x) * log2(p(x)) simplifies to -log2(1/n)
    # For n equally likely outcomes, each with probability 1/n:
    # = -(n * (1/n) * log2(1/n))
    # = -log2(1/n)


# Filtering the possible secrets based on the guess and the number of bulls and cows to narrow down the search space
# The possible secrets are filtered based on the number of bulls and cows obtained from comparing the guess with each secret number.
# This is done, so that only the secrets that match the feedback are retained.
def filter_possible_secrets(possible_secrets: List[str], guess: str, bulls: int, cows: int) -> List[str]:
    return [secret for secret in possible_secrets if compare_guess(secret, guess) == (bulls, cows)] # Return secrets that match the feedback


# Calculating the mutual information between the prior and posterior distributions
def calculate_mutual_information(prior_entropy: float, possible_secrets_after_guess: List[str]) -> float:
    posterior_entropy = calculate_entropy(possible_secrets_after_guess) # Calculate the entropy after the guess
    return prior_entropy - posterior_entropy # Return the mutual information: difference between prior and posterior entropy


# Calculating the expected information gain for each possible guess
# This function calculates the expected information gain for each possible guess based on the current set of possible secrets.
# This helps in determining the optimal guess that maximizes the expected information gain.
def calculate_expected_information_gain(possible_secrets: List[str], guess: str) -> float:
    expected_info_gain = 0
    possible_outcomes = set(compare_guess(secret, guess) for secret in possible_secrets) # Set of possible outcomes based on the guess
    prior_entropy = calculate_entropy(possible_secrets) # Calculate the entropy before the guess

    for outcome in possible_outcomes: # Iterate over possible outcomes
        filtered_secrets = filter_possible_secrets(possible_secrets, guess, *outcome) # Filter secrets based on the outcome
        probability = len(filtered_secrets) / len(possible_secrets) if possible_secrets else 0 # Calculate the probability of the outcome
        expected_info_gain += probability * calculate_mutual_information(prior_entropy, filtered_secrets) # Calculate the expected information gain, i.e., sum of probability * mutual information.

    return expected_info_gain



# Suggesting the optimal guess that maximizes the expected information gain
def suggest_guess(possible_secrets: List[str]) -> Optional[str]:
    if not possible_secrets: 
        return None

    best_guess = possible_secrets[0] # Initialize best guess as the first possible secret.
    max_info_gain = calculate_expected_information_gain(possible_secrets, best_guess) # Calculate the expected information gain for the initial guess.
    initial_entropy = calculate_entropy(possible_secrets)

    if max_info_gain == initial_entropy: # If the maximum information gain is equal to the initial entropy, return the best guess.
        return best_guess

    for guess in possible_secrets[1:]: # Iterate over the remaining possible secrets
        info_gain = calculate_expected_information_gain(possible_secrets, guess)
        if info_gain > max_info_gain: # Update the best guess if the information gain is higher
            max_info_gain = info_gain
            best_guess = guess
            if max_info_gain == initial_entropy: # If the maximum information gain is equal to the initial entropy, break the loop.
                break

    return best_guess



# --- Game Initialization and Statistics ---
def initialize_game() -> Tuple[str, List[str], Dict[str, List]]: # Initialize the game state
    secret_number = generate_secret_number() # Generate the secret number
    possible_secrets = [''.join(p) for p in itertools.permutations("0123456789", 4)] # Generate all possible 4-digit numbers with unique digits
    statistics = { # Initialize statistics dictionary
        'total_guesses': 0, # Initialize total guesses
        'information_gains': [], # Initialize information gains list
        'remaining_possibilities': [], # Initialize remaining possibilities list
        'entropy_values': [], # Initialize entropy values list
        'guess_list': [], # Initialize guess list
        'bulls': [],  # Initialize empty bulls list
        'cows': []    # Initialize empty cows list
    }
    return secret_number, possible_secrets, statistics



# Update statistics after each guess
def update_statistics(statistics: Dict[str, List], guesses: int, guess: str, bulls: int, cows: int, current_entropy: float, info_gain: float, possible_secrets: List[str]) -> None:
    statistics['total_guesses'] = guesses
    statistics['guess_list'].append(guess)
    statistics['bulls'].append(bulls)  
    statistics['cows'].append(cows)
    statistics['remaining_possibilities'].append(len(possible_secrets))
    statistics['entropy_values'].append(current_entropy)
    statistics['information_gains'].append(info_gain)



# Print statistics
def print_statistics(statistics: Dict[str, List], info_gain: float) -> None:
    st.write("### Statistics")
    st.write(f"**Total guesses:** {statistics['total_guesses']}")
    st.write(f"**Guesses made:** {', '.join(statistics['guess_list'])}")
    st.write(f"**Information gain:** {info_gain:.2f} bits")
    if statistics['information_gains']:
        avg_info_gain = sum(statistics['information_gains']) / len(statistics['information_gains'])
        st.write(f"**Average information gain:** {avg_info_gain:.2f} bits")
    st.write(f"**Initial entropy:** {statistics['entropy_values'][0]:.2f} bits")  # Print initial entropy, [0] is the first element
    st.write(f"**Current entropy:** {statistics['entropy_values'][-1]:.2f} bits") # Print initial entropy, [0] is the first element




# --- Streamlit App ---
def main():
    st.title("🐂🎯🐄 Bulls and Cows Game")
    MAX_ATTEMPTS = 20 # Maximum number of attempts allowed

    # restart button
    if st.button("Restart Game"):
        st.session_state.clear()
        st.rerun()

    # Initialize game state
    if 'secret_number' not in st.session_state: # Check if the game state is initialized
        st.session_state.secret_number, st.session_state.possible_secrets, st.session_state.statistics = initialize_game() # Initialize the game state
        st.session_state.guesses = 0 # Initialize the number of guesses
        st.session_state.remaining_attempts = MAX_ATTEMPTS # Initialize the remaining attempts
        st.session_state.show_suggestion = False # Initialize show suggestion state
        st.session_state.game_over = False # Initialize game over state
        st.session_state.suggested_guess = None  # Initialize suggested guess

    # Initialize entropies and mutual informations
    if 'entropies' not in st.session_state:
        st.session_state.entropies = []
    if 'mutual_informations' not in st.session_state:
        st.session_state.mutual_informations = []

    # --- Sidebar Instructions ---
    st.sidebar.header("Bulls and Cows Instructions")
    st.sidebar.info("""
    **Objective:**  
    Guess the 4-digit secret number with all unique digits within the allowed attempts.
    
    **How to Play:**  
    - Enter a 4-digit number with unique digits as your guess.
    - After each guess, you'll receive:
      - **Bulls:** Correct digits in the correct position.
      - **Cows:** Correct digits but in the wrong position.
    
    **Tips:**  
    Use the feedback to narrow down possible secret numbers. Good luck!
    """)

    # --- Main Game Interface ---
    st.write("I've generated a 4-digit secret number with no repeating digits. Try to guess it!")
    st.header("Make a Guess")
    guess_input = st.text_input("Enter your 4-digit guess with unique digits:", max_chars=4, key="guess_input") # Text input for the guess
    submit_button = st.button("Submit Guess", disabled=st.session_state.get('game_over', False), key="submit_button") # Submit button


    # Process the guess
    if submit_button and guess_input and not st.session_state.game_over: # Check if the submit button is clicked and the guess is valid
        if is_valid_guess(guess_input):
            # Recalculate dependent variables
            current_entropy = calculate_entropy(st.session_state.possible_secrets)
            bulls, cows = compare_guess(st.session_state.secret_number, guess_input)

            # Display results
            st.write(f"**Bulls:** <span style='color:green;'>{bulls} 🐂</span>, **Cows:** <span style='color:orange;'>{cows} 🐄</span>", unsafe_allow_html=True)
    

            # --- Update possible secrets and calculate mutual information ---
            new_possible_secrets = filter_possible_secrets(st.session_state.possible_secrets, guess_input, bulls, cows) # Filter possible secrets based on the guess
            info_gain = calculate_mutual_information(current_entropy, new_possible_secrets)
            st.session_state.possible_secrets = new_possible_secrets

            # Update statistics BEFORE checking game end
            update_statistics(
                st.session_state.statistics,
                st.session_state.guesses + 1, 
                guess_input,
                bulls,
                cows,
                current_entropy,
                info_gain,
                st.session_state.possible_secrets
            )
            st.session_state.entropies.append(current_entropy)
            st.session_state.mutual_informations.append(info_gain)

            # --- Update game state after processing guess ---
            st.session_state.remaining_attempts -= 1
            st.session_state.guesses += 1 

            # --- Generate and show suggestion based on updated possible secrets ---
            if st.session_state.remaining_attempts > 0 and not st.session_state.game_over: # Check if the game is not over
                if st.session_state.remaining_attempts <= 5: # Show suggestion when remaining attempts are less than or equal to 5
                    st.session_state.suggested_guess = suggest_guess(st.session_state.possible_secrets)
                    if st.session_state.suggested_guess: # Display suggested guess if available
                        st.info(f"**Suggested guess** ({st.session_state.remaining_attempts} attempts remaining): {st.session_state.suggested_guess}")
                    else:
                        st.warning("No more optimal guesses can be determined.")
                else:
                    st.session_state.suggested_guess = None  # Clear suggestion if not needed

            # Check for win or lose AFTER updating everything
            if bulls == 4: # Check if the guess is correct
                st.success(f"Congratulations! You've guessed the secret number {st.session_state.secret_number} in {st.session_state.guesses} guesses.")
                st.balloons()
                st.snow()
                st.session_state.game_over = True # Set game over state to True
            elif st.session_state.remaining_attempts <= 0: # Check if the remaining attempts are exhausted
                st.error(f"Game Over! You've reached the maximum {MAX_ATTEMPTS} attempts. The secret number was {st.session_state.secret_number}.")
                st.session_state.game_over = True
            elif not st.session_state.possible_secrets and bulls != 4: # Check if there are no possible secrets left
                    st.error("No possible secrets left! Something went wrong.")
                    st.session_state.game_over = True
            
            # Update remaining attempts display
            st.markdown(f"<h3 style='text-align: center; color: red;'>Remaining attempts: {st.session_state.remaining_attempts}</h3>", unsafe_allow_html=True)

        else:
            st.warning("Invalid input. Please enter a 4-digit number with unique digits (e.g., 1234).") # Display warning for invalid input



    # --- Statistics Table ---
    st.header("Statistics Table")
    if st.session_state.statistics['total_guesses'] > 0: # Check if there are any guesses
        data = []
        total_info_gain = 0
        for i in range(st.session_state.statistics['total_guesses']):
            info_gain = st.session_state.statistics['information_gains'][i] # Get information gain
            total_info_gain += info_gain
            data.append([
                i+1,  # Guess number
                st.session_state.statistics['guess_list'][i], # Guess
                st.session_state.statistics['bulls'][i], # Add bulls
                st.session_state.statistics['cows'][i],  # Add cows
                f"{st.session_state.statistics['entropy_values'][i]:.4f}", # Entropy
                f"{info_gain:.4f}",
                st.session_state.statistics['remaining_possibilities'][i]
            ])
        
        # Display average information gain
        avg_info_gain = total_info_gain / st.session_state.statistics['total_guesses'] if st.session_state.statistics['total_guesses'] else 0 
        st.write(f"**Average Information Gain:** {avg_info_gain:.4f} bits")

        # Create DataFrame for statistics to display in a table
        df = pd.DataFrame(
            data,
            columns=[
                'Guess #',
                'Guess',
                'Bulls',
                'Cows', 
                'Entropy',
                'Information Gain',
                'Remaining Possibilities'
            ]
        )
        st.dataframe(df)



    # --- Visualization ---
    st.header("Entropy and Mutual Information Plot")
    if st.session_state.entropies and st.session_state.mutual_informations: # Check if there are entropy and mutual information values
        fig, ax1 = plt.subplots(figsize=(10, 5)) # Create a plot with 2 y-axes

        ax1.set_xlabel("Guesses")
        ax1.set_ylabel("Entropy (bits)", color="tab:blue")
        ax1.plot(st.session_state.entropies, label="Entropy", color="tab:blue", marker="o")
        ax1.tick_params(axis="y", labelcolor="tab:blue") # Set y-axis color

        ax2 = ax1.twinx() # Create a second y-axis
        ax2.set_ylabel("Mutual Information (bits)", color="tab:orange") # Set y-axis label
        ax2.bar(range(1, len(st.session_state.mutual_informations) + 1), st.session_state.mutual_informations, alpha=0.6, color="tab:orange", label="Mutual Information") # Plot mutual information as a bar chart
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig.tight_layout() # Adjust layout
        st.pyplot(fig) # Display the plot. pyplot() is a Streamlit function to display Matplotlib plots

if __name__ == "__main__":
    main()