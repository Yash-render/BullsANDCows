import random
import itertools # For permutations
import math
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Optional # For type hints
 
def generate_secret_number() -> str:
    """Generates a 4-digit secret number with no repeating digits."""
    digits = list(range(10)) # 0-9
    random.shuffle(digits) 
    return "".join(map(str, digits[:4])) # Convert to string


def compare_guess(secret_number: str, guess: str) -> Tuple[int, int]:
    """Compares a guess to the secret number and returns the number of bulls and cows."""
    bulls = sum(1 for i in range(4) if guess[i] == secret_number[i]) # Count bulls
    cows = sum(1 for char in guess if char in secret_number) - bulls # Subtract bulls from cows because they are counted twice
    return bulls, cows


def is_valid_guess(guess: str) -> bool:
    """Validates that input is a 4-digit number with unique digits."""
    return guess.isdigit() and len(guess) == 4 and len(set(guess)) == 4
    # len(set(guess)) == 4 checks for unique digits, set() removes duplicates


def calculate_entropy(possible_secrets: List[str]) -> float: # List of possible secrets
    """Calculates the entropy of the possible secret numbers."""
    if not possible_secrets: 
        return 0.0 # No entropy if no possible secrets
    num_secrets = len(possible_secrets) 
    return -math.log2(1 / num_secrets) # Entropy formula: -Σ p(x) * log2(p(x))
"""
    Assuming uniform distribution, because all secrets are equally likely
    so, the entropy formula -Σ p(x) * log2(p(x)) simplifies to -log2(1/n)
    For n equally likely outcomes, each with probability 1/n:
    = -(n * (1/n) * log2(1/n))
    = -log2(1/n)"""



def filter_possible_secrets(possible_secrets: List[str], guess: str, bulls: int, cows: int) -> List[str]: 
    """Filters the list of possible secrets based on the guess and the number of bulls and cows."""
    return [secret for secret in possible_secrets if compare_guess(secret, guess) == (bulls, cows)]

"""
This function is a crucial part of the game's strategy for narrowing down possible secret numbers. Let me explain why:
After each guess, we receive feedback (bulls and cows)
We need to eliminate numbers that couldn't be the secret based on this feedback

For example:
possible_secrets = ["1234", "1243", "1324", "2134"]
guess = "1234"
bulls = 2  # Let's say we got 2 bulls, 1 cow
cows = 1

# Function will:
# - Compare "1234" with each possible secret
# - Keep only secrets that would give 2 bulls, 1 cow
# - If "1324" gives 2 bulls, 1 cow, it stays in list
# - If "2134" gives different feedback, it's removed

# We use this list to calculate mutual information and suggest the next guess
"""


def calculate_mutual_information(prior_entropy: float, possible_secrets_after_guess: List[str]) -> float: 
    """Calculates the mutual information gained from a guess."""
    posterior_entropy = calculate_entropy(possible_secrets_after_guess) # Calculate entropy after guess
    return prior_entropy - posterior_entropy # Mutual information is the difference between prior and posterior entropy


def visualize_entropy_and_mutual_information(entropies: List[float], mutual_informations: List[float]) -> None:
    """Visualizes entropy and mutual information on the same plot."""
    fig, ax1 = plt.subplots() # Create figure and axis

    ax1.set_xlabel("Guesses")
    ax1.set_ylabel("Entropy (bits)", color="tab:blue")
    ax1.plot(entropies, label="Entropy", color="tab:blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx() # Create a second y-axis that shares the same x-axis
    ax2.set_ylabel("Mutual Information (bits)", color="tab:orange")
    ax2.bar(range(len(mutual_informations)), mutual_informations, alpha=0.6, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout() # Adjust layout to fit labels
    plt.title("Entropy and Mutual Information During Game")
    plt.show()


# This fn calculates the expected information gain for a given guess, so we can suggest the best guess
# This is calculated as the sum of the probabilities of each possible outcome multiplied by the mutual information
def calculate_expected_information_gain(possible_secrets: List[str], guess: str) -> float: 
    """Calculates the expected information gain for a given guess."""
    expected_info_gain = 0 
    possible_outcomes = set(compare_guess(secret, guess) for secret in possible_secrets) # Set of unique outcomes
    prior_entropy = calculate_entropy(possible_secrets) # Entropy before the guess

    for outcome in possible_outcomes: 
        filtered_secrets = filter_possible_secrets(possible_secrets, guess, *outcome) # Filtered secrets for each outcome
        probability = len(filtered_secrets) / len(possible_secrets) # Probability of each outcome
        expected_info_gain += probability * calculate_mutual_information(prior_entropy, filtered_secrets) # Sum of probabilities * mutual information
    
    return expected_info_gain # Expected information gain for the guess


# This fn suggests the best guess based on the expected information gain
def suggest_guess(possible_secrets: List[str]) -> Optional[str]:
    if not possible_secrets: # If no possible secrets, 
        return None
    
    best_guess = possible_secrets[0]  # Initialize with first guess in list
    max_info_gain = calculate_expected_information_gain(possible_secrets, best_guess) # Calculate expected information gain
    initial_entropy = calculate_entropy(possible_secrets) # Calculate initial entropy

    # If first guess gives maximum possible information, return immediately
    if max_info_gain == initial_entropy:
        return best_guess

    for guess in possible_secrets[1:]: # Iterate over remaining guesses
        info_gain = calculate_expected_information_gain(possible_secrets, guess) # Calculate expected information gain
        if info_gain > max_info_gain: # If new guess has higher information gain
            max_info_gain = info_gain # Update max information gain
            best_guess = guess # Update best guess
            if max_info_gain == initial_entropy: # If maximum possible information gain is reached
                break # Exit loop
    
    return best_guess # Return the best guess, output is the guess with the highest expected information gain


def initialize_game() -> Tuple[str, List[str], Dict[str, List]]:
    secret_number = generate_secret_number() # Generate secret number
    possible_secrets = [''.join(p) for p in itertools.permutations("0123456789", 4)] # Generate all possible 4-digit numbers
    statistics = {
        'total_guesses': 0,
        'information_gains': [],
        'remaining_possibilities': [],
        'entropy_values': [],
        'time_per_guess': [],
        'guess_list': []
    }
    return secret_number, possible_secrets, statistics


def print_welcome_message() -> None: # Print welcome message
    print("Welcome to Bulls and Cows!")
    print("I've generated a 4-digit secret number with no repeating digits.")
    print("Try to guess it!")
    print("After each guess, I will tell you the number of bulls (correct digits in the correct position)")
    print("and cows (correct digits in the wrong position).")
    print("Let's start!")


def get_valid_guess() -> str: # Get valid guess from user
    while True:
        guess = input("Enter your 4-digit guess (unique digits): ")
        if is_valid_guess(guess):
            return guess
        print("Invalid input. Please enter a 4-digit number with unique digits (e.g., 1234).")


# This fn updates the statistics after each guess
def update_statistics(statistics: Dict[str, List], guesses: int, guess: str, current_entropy: float, info_gain: float, guess_start_time: float, guess_end_time: float, possible_secrets: List[str]) -> None:
    statistics['total_guesses'] = guesses # Update total guesses
    statistics['guess_list'].append(guess) # Append guess to list
    statistics['remaining_possibilities'].append(len(possible_secrets)) # Append remaining possibilities
    statistics['entropy_values'].append(current_entropy) # Append current entropy
    statistics['information_gains'].append(info_gain) # Append information gain
    statistics['time_per_guess'].append(guess_end_time - guess_start_time) # Append time taken for guess


# This fn prints the statistics after each guess
def print_statistics(statistics: Dict[str, List], info_gain: float) -> None:
    print("\nStatistics:")
    print(f"Total guesses: {statistics['total_guesses']}")
    print(f"Guesses made: {', '.join(statistics['guess_list'])}")
    print(f"Information gain: {info_gain:.2f} bits")
    if statistics['information_gains']: # If information gains exist
        avg_info_gain = sum(statistics['information_gains']) / len(statistics['information_gains']) # Calculate average information gain
        print(f"Average information gain: {avg_info_gain:.2f} bits") # Print average information gain
    print(f"Initial entropy: {statistics['entropy_values'][0]:.2f} bits") # Print initial entropy, [0] is the first element
    print(f"Current entropy: {statistics['entropy_values'][-1]:.2f} bits") # Print current entropy, [-1] is the last element
    print(f"Time taken for this guess: {statistics['time_per_guess'][-1]:.2f} seconds") # Print time taken for this guess, [-1] is the last element
    if len(statistics['time_per_guess']) > 1:   # If more than 1 guess has been made
        avg_guess_time = sum(statistics['time_per_guess']) / len(statistics['time_per_guess']) # Calculate average time per guess
        print(f"Average time per guess: {avg_guess_time:.2f} seconds") # Print average time per guess


# This fn prints the final statistics after the game ends
def print_final_statistics(statistics: Dict[str, List], total_time: float) -> None:
    print("\n--- Final Game Statistics ---")
    print(f"Total guesses: {statistics['total_guesses']}")
    print(f"Guesses made: {', '.join(statistics['guess_list'])}")
    if statistics['information_gains']:
        avg_info_gain = sum(statistics['information_gains']) / len(statistics['information_gains'])
        print(f"Average information gain: {avg_info_gain:.2f} bits")
    print(f"Initial entropy: {statistics['entropy_values'][0]:.2f} bits")
    if statistics['entropy_values']:
        print(f"Final entropy: {statistics['entropy_values'][-1]:.2f} bits")
    if statistics['time_per_guess']:
        avg_guess_time = sum(statistics['time_per_guess']) / len(statistics['time_per_guess'])
        print(f"Average time per guess: {avg_guess_time:.2f} seconds")
    print(f"Total time taken: {total_time:.2f} seconds")


# This fn plays the game
def play_game() -> None:
    secret_number, possible_secrets, statistics = initialize_game()
    entropies = []
    mutual_informations = []
    guesses = 0
    max_attempts = 20
    start_time = time.time()  # Capture start time

    print_welcome_message()

    while guesses < max_attempts:
        guess_start_time = time.time()  # Capture guess start time
        current_entropy = calculate_entropy(possible_secrets)
        entropies.append(current_entropy)

        print(f"\nRemaining possible numbers: {len(possible_secrets)}")
        print(f"Current entropy: {current_entropy:.2f} bits")
        print(f"Attempts left: {max_attempts - guesses}")

        if max_attempts - guesses <= 5: # If 5 or fewer attempts left, suggest a guess
            suggested_guess = suggest_guess(possible_secrets)
            if suggested_guess:
                print(f"\nSuggestion (5 or fewer attempts left): {suggested_guess}")
            else:
                print("\nNo more optimal moves can be determined.")

        guess = get_valid_guess() # Get valid guess from user
        guesses += 1

        bulls, cows = compare_guess(secret_number, guess) # Compare guess to secret number
        print(f"Bulls: {bulls}, Cows: {cows}")

        if bulls == 4: # If 4 bulls, correct guess
            print(f"Congratulations! You've guessed the secret number {secret_number} in {guesses} guesses.")
            mutual_informations.append(current_entropy)
            visualize_entropy_and_mutual_information(entropies, mutual_informations) # Visualize entropy and mutual information
            break

        new_possible_secrets = filter_possible_secrets(possible_secrets, guess, bulls, cows) # Filter possible secrets
        info_gain = calculate_mutual_information(current_entropy, new_possible_secrets) # Calculate mutual information
        mutual_informations.append(info_gain)

        possible_secrets = new_possible_secrets

        if not possible_secrets:
            print(f"No possible secrets left! Something went wrong.")
            break

        guess_end_time = time.time()  # Capture guess end time
        update_statistics(statistics, guesses, guess, current_entropy, info_gain, guess_start_time, guess_end_time, possible_secrets)
        print_statistics(statistics, info_gain)

        if guesses == max_attempts: # If max attempts reached, game over
            print(f"Game over! The secret number was {secret_number}.")

    end_time = time.time()  # Capture end time
    total_time = end_time - start_time
    statistics['total_time'] = total_time

    print_final_statistics(statistics, total_time)

if __name__ == "__main__":
    play_game()