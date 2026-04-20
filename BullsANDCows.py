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
def process_guess_callback():
    guess_input = st.session_state.guess_input_widget
    # Force strictly numeric behavior by stripping whitespace
    guess_input = guess_input.strip()
    
    if is_valid_guess(guess_input):
        st.session_state.invalid_input = False
        st.session_state.invalid_reason = ""
        current_entropy = calculate_entropy(st.session_state.possible_secrets)
        bulls, cows = compare_guess(st.session_state.secret_number, guess_input)

        new_possible_secrets = filter_possible_secrets(st.session_state.possible_secrets, guess_input, bulls, cows)
        info_gain = calculate_mutual_information(current_entropy, new_possible_secrets)
        st.session_state.possible_secrets = new_possible_secrets

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

        st.session_state.remaining_attempts -= 1
        st.session_state.guesses += 1 

        if st.session_state.remaining_attempts > 0 and not st.session_state.game_over:
            if st.session_state.remaining_attempts <= 5:
                st.session_state.suggested_guess = suggest_guess(st.session_state.possible_secrets)
            else:
                st.session_state.suggested_guess = None

        if bulls == 4:
            st.session_state.game_over = True
            st.session_state.is_victory = True
        elif st.session_state.remaining_attempts <= 0:
            st.session_state.game_over = True
            st.session_state.is_loss = True
        elif not st.session_state.possible_secrets and bulls != 4:
            st.session_state.game_over = True
            st.session_state.is_error = True
    else:
        st.session_state.invalid_input = True


def main():
    st.set_page_config(page_title="Bulls & Cows", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #050505 !important;
    color: #E0E0E0 !important;
}

header {
    background: transparent !important;
    border-bottom: none !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container { padding-top: 3rem !important; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #080808 !important;
    border-right: 1px solid rgba(212, 175, 55, 0.1);
}
[data-testid="stSidebar"] * {
    color: #AAA !important;
}

.stApp {
    background: radial-gradient(circle at 50% -20%, #1a1a14 0%, #050505 80%);
}

/* Professional Glass Panel for Streamlit Columns */
[data-testid="column"] > div {
    background: rgba(255, 255, 255, 0.015) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px !important;
    padding: 35px !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8) !important;
    transition: all 0.3s ease;
}
[data-testid="column"]:hover > div {
    border-color: rgba(212, 175, 55, 0.2) !important;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #FFF 0%, #D4AF37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
    margin-bottom: 30px;
    letter-spacing: -1px;
    width: 100%;
}

[data-testid="stTextInput"] input, 
div[data-baseweb="input"] input,
.stTextInput input {
    background: rgba(0, 0, 0, 0.8) !important;
    border: 2px solid rgba(212, 175, 55, 0.4) !important;
    color: #D4AF37 !important;
    font-size: 5rem !important;
    letter-spacing: 40px !important;
    text-align: center !important;
    border-radius: 24px !important;
    padding: 0 !important;
    height: 180px !important;
    line-height: 180px !important;
    font-family: 'Outfit', monospace !important;
    font-weight: 800 !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: inset 0 2px 40px rgba(0,0,0,0.9) !important;
    text-shadow: 0 0 20px rgba(212, 175, 55, 0.6) !important;
}

[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background: transparent !important;
    border: none !important;
    height: 180px !important;
}

[data-testid="stTextInput"] input:focus,
div[data-baseweb="input"] input:focus {
    border-color: #D4AF37 !important;
    background: rgba(10, 10, 5, 0.9) !important;
    box-shadow: inset 0 2px 40px rgba(0,0,0,0.9), 0 0 50px rgba(212, 175, 55, 0.2) !important;
    transform: scale(1.02);
}

div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #D4AF37 0%, #AA8C2C 100%) !important;
    color: #000 !important;
    border: none !important;
    padding: 18px 30px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    transition: all 0.3s ease !important;
    margin-top: 15px !important;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.1) !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(212, 175, 55, 0.3) !important;
}

div[data-testid="stButton"] button {
    background: transparent !important;
    color: #666 !important;
    border: 1px solid #333 !important;
    padding: 15px 20px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stButton"] button:hover {
    color: #D4AF37 !important;
    border-color: #D4AF37 !important;
    background: rgba(212, 175, 55, 0.05) !important;
}

.stat-box {
    text-align: center;
    padding: 10px;
    margin-bottom: 20px;
}
.stat-value {
    font-family: 'Outfit', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(to bottom, #FFF, #555);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-value.gold {
    background: linear-gradient(to bottom, #F3E5AB, #D4AF37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-label {
    font-size: 0.8rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 5px;
    font-weight: 600;
}

.guess-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 25px;
    background: rgba(255,255,255,0.015);
    border-radius: 16px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.03);
    transition: background 0.3s, transform 0.3s;
}
.guess-row:hover {
    transform: translateX(5px);
    background: rgba(255,255,255,0.03);
    border-color: rgba(212, 175, 55, 0.2);
}
.digit-container {
    display: flex;
    gap: 10px;
}
.digit-box {
    width: 50px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    border: 1px solid #222;
    border-radius: 10px;
    font-family: 'Outfit', monospace;
    font-size: 2rem;
    font-weight: 600;
    color: #FFF;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.8);
}
.visual-feedback {
    display: flex;
    gap: 8px;
    align-items: center;
}
.dot-bull { font-size: 1.5rem; }
.dot-cow { font-size: 1.5rem; }

.stTabs [data-baseweb="tab-list"] { gap: 30px; }
.stTabs [data-baseweb="tab"] { color: #555; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; }
.stTabs [aria-selected="true"] { color: #D4AF37 !important; }
.stTabs [data-baseweb="tab-border"] { background-color: #D4AF37 !important; }

@keyframes slideDown { 0% { opacity: 0; transform: translateY(-10px); } 100% { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

    MAX_ATTEMPTS = 20

    if 'secret_number' not in st.session_state:
        st.session_state.secret_number, st.session_state.possible_secrets, st.session_state.statistics = initialize_game()
        st.session_state.guesses = 0
        st.session_state.remaining_attempts = MAX_ATTEMPTS
        st.session_state.game_over = False
        st.session_state.is_victory = False
        st.session_state.is_loss = False
        st.session_state.is_error = False
        st.session_state.invalid_input = False
        st.session_state.suggested_guess = None
        st.session_state.entropies = []
        st.session_state.mutual_informations = []
        st.session_state.played_win_animation = False

    if st.session_state.is_victory and not st.session_state.played_win_animation:
        st.balloons()
        st.session_state.played_win_animation = True

    with st.sidebar:
        st.markdown("<h2 style='color:#D4AF37; font-family:Outfit;'>HOW TO PLAY</h2>", unsafe_allow_html=True)
        st.markdown("""
        **Objective**
        Crack the 4-digit code. All digits are unique (e.g., 1234).
        
        **Feedback Rules**
        - <span style='color:#D4AF37;'>●</span> **Bulls**: Correct digit in the correct spot.
        - <span style='color:#666;'>○</span> **Cows**: Correct digit in the wrong spot.
        
        **Attempts**
        You have 20 attempts to solve the sequence.
        """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>BULLS & COWS</div>", unsafe_allow_html=True)

    col1, spacer, col2 = st.columns([1.2, 0.1, 1.8])

    with col1:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"<div class='stat-box'><div class='stat-value gold'>{st.session_state.remaining_attempts}</div><div style='color:#666; font-size:0.75rem; letter-spacing:2px; font-weight:800; text-transform:uppercase;'>Attempts Left</div></div>", unsafe_allow_html=True)
        with s2:
            current_entropy_val = calculate_entropy(st.session_state.possible_secrets)
            st.markdown(f"<div class='stat-box'><div class='stat-value'>{current_entropy_val:.1f}</div><div style='color:#666; font-size:0.75rem; letter-spacing:2px; font-weight:800; text-transform:uppercase;'>Entropy Bits</div></div>", unsafe_allow_html=True)
        
        with st.form("guess_form", clear_on_submit=True):
            st.text_input("GUESS INPUT", max_chars=4, placeholder="0000", label_visibility="collapsed", key="guess_input_widget")
            st.form_submit_button("Initiate Sequence", width="stretch", disabled=st.session_state.game_over, on_click=process_guess_callback)
            
        if st.session_state.invalid_input:
            st.markdown("<p style='color: #E74C3C; text-align: center; margin-top: 15px; font-weight: 600; font-size:0.85rem; letter-spacing:1px;'>ERROR: SEQUENCE MUST CONTAIN 4 UNIQUE DIGITS.</p>", unsafe_allow_html=True)
            
        if st.session_state.remaining_attempts <= 5 and not st.session_state.game_over:
            if st.session_state.suggested_guess:
                st.markdown(f"<div style='margin-top: 30px; padding: 25px; border-left: 2px solid #D4AF37; background: rgba(212,175,55,0.05);'><div style='font-size: 0.75rem; color: #D4AF37; text-transform: uppercase; letter-spacing: 3px; font-weight: 800; margin-bottom: 8px;'>AI Optimal Protocol</div><div style='font-family: Outfit, monospace; font-size: 2.5rem; color: #FFF; letter-spacing: 8px; font-weight: 600;'>{st.session_state.suggested_guess}</div></div>", unsafe_allow_html=True)

        if st.session_state.game_over:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.is_victory:
                st.markdown(f"<div style='text-align:center; color:#D4AF37; font-size:1.8rem; font-weight:800; font-family: Outfit; letter-spacing: 2px;'>ACCESS GRANTED: {st.session_state.secret_number}</div>", unsafe_allow_html=True)
            elif st.session_state.is_loss:
                st.markdown(f"<div style='text-align:center; color:#E74C3C; font-size:1.8rem; font-weight:800; font-family: Outfit; letter-spacing: 2px;'>PROTOCOL OVERRIDE: {st.session_state.secret_number}</div>", unsafe_allow_html=True)
        
        if st.button("RESTART GAME", width="stretch"):
            st.session_state.clear()
            st.rerun()

    with col2:
        tab1, tab2 = st.tabs(["Sequence History", "Data Telemetry"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.session_state.statistics['total_guesses'] > 0:
                html_history = ""
                for i in range(st.session_state.statistics['total_guesses'] - 1, -1, -1):
                    g_num = st.session_state.statistics['guess_list'][i]
                    b_count = st.session_state.statistics['bulls'][i]
                    c_count = st.session_state.statistics['cows'][i]
                    
                    digits_html = "".join([f"<div class='digit-box'>{d}</div>" for d in g_num])
                    
                    bulls_dots = "".join(["<div class='dot-bull'></div>" for _ in range(b_count)])
                    cows_dots = "".join(["<div class='dot-cow'></div>" for _ in range(c_count)])
                    
                    html_history += f"""
                    <div class='guess-row' style='animation: slideDown 0.4s ease-out backwards; animation-delay: {((st.session_state.statistics['total_guesses'] - 1) - i) * 0.05}s;'>
                        <div style='color: #222; font-family: Outfit; font-weight: 800; font-size: 1.2rem; width: 30px;'>{i+1}</div>
                        <div style='display:flex; gap:10px; flex-grow:1; margin-left:15px;'>{digits_html}</div>
                        <div style='display:flex; gap:20px; background:rgba(0,0,0,0.3); padding:10px 25px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); align-items:center;'>
                            <div class='visual-feedback' style='font-size:1.5rem; gap:10px;'>{b_count} 🐂</div>
                            <div style='width:1px; height:25px; background:rgba(255,255,255,0.1);'></div>
                            <div class='visual-feedback' style='font-size:1.5rem; gap:10px;'>{c_count} 🐄</div>
                        </div>
                    </div>
                    """
                st.markdown(html_history, unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center; padding: 120px 20px; opacity:0.6;'><div style='font-family: Outfit; font-weight: 800; font-size: 2.5rem; margin-bottom: 10px; color: #D4AF37;'>AWAITING INPUT</div><div style='letter-spacing: 3px; font-size: 0.85rem; text-transform: uppercase; color: #E0E0E0;'>Telemetry will initialize upon first sequence</div></div>", unsafe_allow_html=True)
                
        with tab2:
            if st.session_state.statistics['total_guesses'] > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                fig, ax1 = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0)
                ax1.patch.set_alpha(0.0)
                ax1.set_xlabel("SEQUENCE", color="#555", fontweight="bold", fontsize=9)
                ax1.set_ylabel("ENTROPY", color="#D4AF37", fontweight="bold", fontsize=9)
                ax1.plot(st.session_state.entropies, color="#D4AF37", marker="o", linewidth=2.5, markersize=7, markerfacecolor="#050505")
                ax1.tick_params(colors="#333")
                ax1.grid(color='#222', linestyle='-', linewidth=1)
                ax2 = ax1.twinx()
                ax2.bar(range(1, len(st.session_state.mutual_informations) + 1), st.session_state.mutual_informations, alpha=0.15, color="#FFF")
                ax2.axis('off')
                for s in ax1.spines.values(): s.set_visible(False)
                st.pyplot(fig)
                with st.expander("RAW DIAGNOSTICS"):
                    data = []
                    for i in range(st.session_state.statistics['total_guesses']):
                        info_gain = st.session_state.statistics['information_gains'][i]
                        data.append([i+1, st.session_state.statistics['guess_list'][i], st.session_state.statistics['bulls'][i], st.session_state.statistics['cows'][i], f"{st.session_state.statistics['entropy_values'][i]:.4f}", f"{info_gain:.4f}", st.session_state.statistics['remaining_possibilities'][i]])
                    st.dataframe(pd.DataFrame(data, columns=['ID', 'Code', 'Bulls', 'Cows', 'Entropy', 'Info Gain', 'Remaining']), width="stretch", hide_index=True)
            else:
                st.markdown("<div style='text-align:center; padding: 120px 20px; opacity:0.6;'><div style='font-family: Outfit; font-weight: 800; font-size: 2.5rem; margin-bottom: 10px; color: #D4AF37;'>NO DATA</div><div style='letter-spacing: 3px; font-size: 0.85rem; text-transform: uppercase; color: #E0E0E0;'>Execute sequence to view telemetry</div></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

