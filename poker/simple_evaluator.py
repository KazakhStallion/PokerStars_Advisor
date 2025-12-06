import random
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def hero_equity_vs_random_hand(
    hero_cards,
    board_cards,
    num_players: int = 6,
    iterations: int = 100_000,
):
    """
    Estimate hero's equity vs random opponents at an N-handed table and
    return the current made hand type on the *known* board.

    Parameters
    ----------
    hero_cards : sequence[str]
        Hero's 2 hole cards, e.g. ["As", "Kd"].
    board_cards : sequence[str]
        0–5 community cards, e.g. ["7h", "8s", "9d"].
    num_players : int
        Total number of players at the table (hero + villains).
    iterations : int
        Number of Monte Carlo trials.

    Returns
    -------
    (equity, hand_type)
        equity    : float in [0.0, 1.0]
        hand_type : string such as "Pair", "Flush", etc.
                    If fewer than 5 total known cards (hero + board),
                    this will be "none".
    """
    hero_cards = list(hero_cards)
    board_cards = list(board_cards)

    # Basic validation
    if len(hero_cards) != 2:
        raise ValueError(f"Hero must have exactly 2 cards, got {len(hero_cards)}")

    if len(board_cards) > 5:
        raise ValueError(f"Board can have at most 5 cards, got {len(board_cards)}")

    if num_players < 2:
        raise ValueError("num_players must be at least 2 (hero + at least one villain).")

    opponents = num_players - 1

    all_str_cards = hero_cards + board_cards
    if len(set(all_str_cards)) != len(all_str_cards):
        raise ValueError("Duplicate cards detected between hero and board.")

    # Convert hero + board to eval7.Card
    hero_eval = [eval7.Card(c) for c in hero_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    # Build full deck and remove used cards
    full_deck = [eval7.Card(r + s) for r in RANKS for s in SUITS]
    used_set = {c for c in hero_eval + board_eval}
    remaining_deck = [c for c in full_deck if c not in used_set]

    # Number of community cards still to be dealt
    cards_needed_for_board = 5 - len(board_eval)
    if cards_needed_for_board < 0:
        raise ValueError("Board cannot have more than 5 cards.")

    hero_equity_sum = 0.0

    for _ in range(iterations):
        # Sample cards for villains and any missing board cards
        sample_size = 2 * opponents + cards_needed_for_board
        drawn = random.sample(remaining_deck, sample_size)

        idx = 0
        villain_hands = []
        for _op in range(opponents):
            villain_hand = drawn[idx:idx + 2]
            idx += 2
            villain_hands.append(villain_hand)

        # Remaining cards become the rest of the board
        extra_board = drawn[idx:idx + cards_needed_for_board]
        full_board = board_eval + extra_board

        # Evaluate hero
        hero_all = hero_eval + full_board
        hero_score = eval7.evaluate(hero_all)

        # Evaluate villains
        villain_scores = []
        best_score = hero_score
        for villain_hand in villain_hands:
            v_all = villain_hand + full_board
            v_score = eval7.evaluate(v_all)
            villain_scores.append(v_score)
            if v_score > best_score:
                best_score = v_score

        # Count winners
        num_winners = 0
        if hero_score == best_score:
            num_winners += 1
        num_winners += sum(1 for v_score in villain_scores if v_score == best_score)

        # Hero gets equal share if they are among the winners
        if hero_score == best_score:
            hero_equity_sum += 1.0 / num_winners

    equity = hero_equity_sum / iterations if iterations > 0 else 0.0

    # Find handtype on the *known* board (no future cards)
    all_cards = hero_eval + board_eval
    if len(all_cards) >= 5:
        score = eval7.evaluate(all_cards)
        hand_type = eval7.handtype(score)
    else:
        hand_type = "none"

    return equity, hand_type
