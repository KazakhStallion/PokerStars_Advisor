import random
import eval7

# Ranks and Suits for deck construction
RANKS = "23456789TJQKA"
SUITS = "cdhs"


# Hand evaluation (hand type)

def evaluate_hero_hand(hero_cards, board_cards):
    """
    Print hero's current made hand (if any) using eval7.evaluate + eval7.handtype.

    hero_cards:  list of 2 strings, e.g. ["As", "Kd"]
    board_cards: list of 0–5 strings, e.g. ["7h", "8h", "9h"]
    """

    # Input validation
    if len(hero_cards) != 2:
        raise ValueError(f"Hero must have exactly 2 cards, got {len(hero_cards)}")

    if len(board_cards) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board_cards)}")

    all_str_cards = list(hero_cards) + list(board_cards)
    if len(all_str_cards) != len(set(all_str_cards)):
        raise ValueError("Duplicate cards detected between hero and board")

    # Convert to eval7 Card objects
    hero = [eval7.Card(c) for c in hero_cards]
    board = [eval7.Card(c) for c in board_cards]
    all_cards = hero + board

    # Calculate street name, can be obtained from GameState in the future
    street_name = {
        0: "Preflop",
        3: "Flop",
        4: "Turn",
        5: "River",
    }.get(len(board_cards), "Unknown street")

    print("=== Hand Evaluation ===")
    print(f"Street:       {street_name}")
    print(f"Hero cards:   {' '.join(hero_cards)}")
    print(f"Board cards:  {' '.join(board_cards) if board_cards else '(none yet)'}")

    # Need at least 5 cards to have a made 5-card hand
    if len(all_cards) < 5:
        print("Hand type:    (More cards needed to form a hand)")
        print("=======================\n")
        return None, None

    # Evaluate with eval7, hand type
    # (can delete this in future)
    score = eval7.evaluate(all_cards)
    hand_type = eval7.handtype(score)

    print(f"Hand type:    {hand_type}")
    print(f"Raw score:    {score}")
    print("=======================\n")

    return score, hand_type


# ---------------------------------------------------
# Equity vs random opponents at an N-handed table
# ---------------------------------------------------

def hero_equity_vs_random_hand(
    hero_cards,
    board_cards,
    num_players=6,
    iterations=100_000,
):
    """
    Estimate hero's equity at an N-handed table using Monte Carlo simulation.

    - Hero has fixed cards (hero_cards).
    - Board may be partially revealed (0–5 cards).
    - The remaining (num_players - 1) opponents all have completely random
      2-card hands, uniformly sampled from the remaining deck.
    - Pot is assumed to be split equally among all winners on ties.

    hero_cards:   list/tuple of 2 strings, e.g. ["As", "Kd"]
    board_cards:  list/tuple of 0–5 strings, e.g. ["7h", "8h", "9h"]
    num_players:  total players at the table (including hero).
                  For your case, use 6 (hero + 5 opponents).
    iterations:   number of Monte Carlo trials.

    Returns:
        equity as float between 0.0 and 1.0
        (hero's expected share of the pot)
    """

    # Input validation
    if num_players < 2:
        raise ValueError("num_players must be at least 2 (hero + at least one opponent).")

    if len(hero_cards) != 2:
        raise ValueError(f"Hero must have exactly 2 cards, got {len(hero_cards)}")

    if len(board_cards) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board_cards)}")

    all_str_cards = list(hero_cards) + list(board_cards)
    if len(all_str_cards) != len(set(all_str_cards)):
        raise ValueError("Duplicate cards detected between hero and board")

    # Check deck capacity: hero + board + villains' cards must fit in 52
    max_needed_cards = 2 * num_players + 5  # worst case: full board + all hands
    if max_needed_cards > 52:
        raise ValueError(
            f"Too many players ({num_players}) for a single deck game. "
            f"Max players with 2-card hands and 5-card board is 23."
        )

    # Precompute eval7.Card objects for hero and known board
    hero_eval = [eval7.Card(c) for c in hero_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    # Build a full 52-card deck as strings
    full_deck = [r + s for r in RANKS for s in SUITS]

    # Remove hero + board cards from the deck (by string)
    known_set = set(all_str_cards)
    remaining_deck = [c for c in full_deck if c not in known_set]

    # How many more community cards need to be dealt to complete the board?
    cards_needed_for_board = 5 - len(board_cards)
    if cards_needed_for_board < 0:
        raise ValueError("Board cannot have more than 5 cards")

    opponents = num_players - 1

    # Monte Carlo simulation
    hero_equity_sum = 0.0

    for _ in range(iterations):
        # For each trial:
        # - Deal 2 cards to each opponent
        # - Deal remaining community cards (if any)
        # All from remaining_deck.

        sample_size = 2 * opponents + max(cards_needed_for_board, 0)
        drawn = random.sample(remaining_deck, sample_size)

        idx = 0
        villain_hands_str = []
        for _op in range(opponents):
            villain_hands_str.append(drawn[idx:idx + 2])
            idx += 2

        future_board_str = drawn[idx:]
        future_board_eval = [eval7.Card(c) for c in future_board_str]

        full_board = board_eval + future_board_eval

        # Evaluate hero
        hero_full = hero_eval + full_board
        hero_score = eval7.evaluate(hero_full)

        # Evaluate all villains
        villain_scores = []
        for vh in villain_hands_str:
            v_eval = [eval7.Card(c) for c in vh]
            v_full = v_eval + full_board
            v_score = eval7.evaluate(v_full)
            villain_scores.append(v_score)

        # Determine winner(s) and hero's share of pot in this trial
        best_score = max([hero_score] + villain_scores)

        if hero_score < best_score:
            # Hero loses, gets 0 share
            continue

        # Hero has at least tied for best
        num_winners = 1  # hero
        for v_score in villain_scores:
            if v_score == best_score:
                num_winners += 1

        # Hero's share of pot is 1 / num_winners in case of ties
        hero_equity_sum += 1.0 / num_winners

    if iterations == 0:
        return 0.0

    equity = hero_equity_sum / iterations
    return equity


def print_hero_equity_vs_random_hand(
    hero_cards,
    board_cards,
    num_players=6,
    iterations=100_000,
):
    """
    Convenience wrapper: computes equity vs random opponents and prints a summary.

    num_players is total players at the table (including hero).
    """
    equity = hero_equity_vs_random_hand(
        hero_cards,
        board_cards,
        num_players=num_players,
        iterations=iterations,
    )

    street_name = {
        0: "Preflop",
        3: "Flop",
        4: "Turn",
        5: "River",
    }.get(len(board_cards), "Unknown street")

    print("=== Hero Equity vs Random Table ===")
    print(f"Street:        {street_name}")
    print(f"Hero hand:     {' '.join(hero_cards)}")
    print(f"Board cards:   {' '.join(board_cards) if board_cards else '(none yet)'}")
    print(f"Table size:    {num_players} players (hero + {num_players - 1} random opponents)")
    print("Villain range: Random Hands")
    print(f"Method:        Monte Carlo ({iterations:,} iterations)")
    print(f"Equity:        {equity:.4f}  ({equity * 100:.2f}%)")
    print("===================================\n")

    return equity


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    # Change these to test different spots

    # Hero's cards
    hero = ["As", "Kd"]

    # Board:
    #   []                              -> preflop
    #   ["7h", "8h", "9h"]              -> flop
    #   ["7h", "8h", "9h", "2c"]        -> turn
    #   ["7h", "8h", "9h", "2c", "3d"]  -> river
    board = ["Ah", "Ad", "9h"]
    # Total players at the table (including hero)
    num_players = 6  # hero + 5 random villains

    # 1) Print current made hand (if there is one)
    evaluate_hero_hand(hero, board)

    # 2) Print equity at a 6-handed random table
    print_hero_equity_vs_random_hand(
        hero,
        board,
        num_players=num_players,
        iterations=50_000,
    )
