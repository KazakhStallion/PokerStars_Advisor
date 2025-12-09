import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from poker.simple_evaluator import hero_equity_vs_random_hand

def extract_state_from_json(
    json_path: str,
) -> Tuple[
    List[str],                 # hero_cards
    List[str],                 # board_cards
    List[Dict[str, Any]],      # seats
    List[Dict[str, Any]],      # active_seats
    str,                       # street_name
    Optional[int],             # button_seat
    Optional[float],           # pot_size (or None)
    List[Optional[float]],     # seat_bets (per seat, or None)
    List[Optional[str]],       # seat_statuses (per seat, or None)
]:
    """
    Read a GameState JSON file and extract:
      - hero_cards:   e.g. ["As", "Kd"]
      - board_cards:  e.g. ["7h", "8h", "9h"]
      - seats:        raw seat dicts from JSON
      - active_seats: subset where is_active == True
      - pot_size:     current pot size
      - seat_bets:    list of bet values per seat
      - seat_statuses:list of last_status values per seat
    """
    path = Path(json_path)

    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    hero_cards: List[str] = state.get("hero_cards", [])
    board_cards: List[str] = state.get("board_cards", [])
    seats: List[Dict[str, Any]] = state.get("seats", [])

    street_name: str = state.get("street", "unknown")
    button_seat: Optional[int] = state.get("button_seat")
    pot_size: Optional[float] = state.get("pot_size")

    seat_bets: List[Optional[float]] = [seat.get("bet") for seat in seats]
    active_seats = [s for s in seats if s.get("is_active", False)]
    seat_statuses: List[Optional[str]] = [seat.get("last_status") for seat in seats]

    return (
        hero_cards,
        board_cards,
        seats,
        active_seats,
        street_name,
        button_seat,
        pot_size,
        seat_bets,
        seat_statuses,
    )


def run_simple_evaluator_from_json(json_path: str, iterations: int = 50_000) -> None:
    """
    Load a GameState JSON snapshot, compute hero's equity vs random hands,
    and print a formatted summary of the game state.

    This uses simple_evaluator.hero_equity_vs_random_hand, which returns
    (equity, hand_type).
    """
    (
        hero_cards,
        board_cards,
        seats,
        active_seats,
        street_name,
        button_seat,
        pot_size,
        seat_bets,
        seat_statuses,
    ) = extract_state_from_json(json_path)


    num_players = len(active_seats)

    non_none_bets = [b for b in seat_bets if b is not None]
    max_bet = max(non_none_bets) if non_none_bets else None

    match button_seat:
        case 0:
            position = "BTN"
        case 1:
            position = "SB"
        case 2:
            position = "BB"     
        case 3:
            position = "UTG"
        case 4:
            position = "HJ"
        case 5:         
            position = "CO"
        case _:
            position = "unknown"

    pot_equity = max_bet / (pot_size + max_bet) if (pot_size is not None and max_bet is not None) else 0.0

    equity, hand_type = hero_equity_vs_random_hand(
        hero_cards,
        board_cards,
        num_players=num_players,
        iterations=iterations,
    )

    # Action Logic
    #num_bets = num of players whose "bet" value is not None and > 0 in addition to their last_status being "bet" or "raise"
    num_bets = sum(
        1
        for bet, status in zip(seat_bets, seat_statuses)
        if bet is not None and bet > 0 and status in ("bet", "raise", "allin")
    )

    num_calls = sum(
        1
        for bet, status in zip(seat_bets, seat_statuses)
        if bet is not None and bet > 0 and status == "call"
    )
    action = "Undecided"
    
    # Preflop:
    if street_name == "preflop":
        if position == "UTG":
            if equity > 0.20:
                action = "Raise"
            else:
                action = "Fold"
        elif position == "HJ":
            if (num_bets == 0):     # open betting
                if equity > 0.19:
                    action = "Raise"
                else:
                    action = "Fold"
            elif (num_bets >= 1): 
                if equity > 0.28:
                    action = "Raise"
                elif equity > 0.22:
                    action = "Call"
                else:
                    action = "Fold"
        elif position == "CO":
            if (num_bets == 0):     # open betting
                if equity > 0.18:
                    action = "Raise"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 0): # 5 players, 1 fold
                if equity > 0.30:
                    action = "Raise"
                elif equity > 0.26:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls > 0): # 6 players, no folds
                if equity > 0.26:
                    action = "Raise"
                elif equity > 0.21:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets > 1):  
                if equity > 0.37:
                    action = "Raise"
                elif equity > 0.28:
                    action = "Call"
                else:
                    action = "Fold"    
        elif position == "BTN":
            if (num_bets == 0):     # open betting
                if equity > 0.18:
                    action = "Raise"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 0): # 4 players
                if equity > 0.33:
                    action = "Raise"
                elif equity > 0.28:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 1): # 5 players
                if equity > 0.32:
                    action = "Raise"
                elif equity > 0.27:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls > 1): # 6 players
                if equity > 0.30:
                    action = "Raise"
                elif equity > 0.28:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets > 1):  
                if equity > 0.31:
                    action = "Raise"
                elif equity > 0.29:
                    action = "Call"
                else:
                    action = "Fold"
        elif position == "SB":
            if (num_bets == 0):     # open betting
                if equity > 0.20:
                    action = "Raise"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 0): # 3 players
                if equity > 0.43:
                    action = "Raise"
                elif equity > 0.38:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 1): # 4 players
                if equity > 0.35:
                    action = "Raise"
                elif equity > 0.32:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls > 1): # 5 players
                if equity > 0.32:
                    action = "Raise"
                elif equity > 0.30:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets > 1):  
                if equity > 0.40:
                    action = "Raise"
                elif equity > 0.30:
                    action = "Call"
                else:   
                    action = "Fold"      
        elif position == "BB":
            if (num_bets == 0):     # open betting
                if equity > 0.20:
                    action = "Raise"
                else:
                    action = "Check"
            elif (num_bets == 1 and num_calls == 0): # 2 players
                if equity > 0.58:
                    action = "Raise"
                elif equity > 0.49:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls == 1): # 3 players
                if equity > 0.42:
                    action = "Raise"
                elif equity > 0.31:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets == 1 and num_calls > 1): # 4 players
                if equity > 0.34:
                    action = "Raise"
                elif equity > 0.26:
                    action = "Call"
                else:
                    action = "Fold"
            elif (num_bets > 1):  
                if equity > 0.40:
                    action = "Raise"
                elif equity > 0.30:
                    action = "Call"
                else:   
                    action = "Fold"   
    
    else:   # Flop+ 
        if (num_bets == 0): #open betting
            if equity > 0.3:
                action = "Bet"
            else:
                action = "Check"
        else:
            relative_equity = equity - pot_equity
            #  Pot equity to hand equity comparison
            if relative_equity < 0:
                action = "Fold"     # fold immediately
            else:
                # React to other bets
                hidden_equity = equity
                if (num_bets == 1):
                    hidden_equity -= 0.05

                if (num_bets == 2):
                    hidden_equity -= 0.10

                # Equity buckets
                if hidden_equity > 0.35:
                    action = "Raise"
                elif hidden_equity > 0.20:
                    action = "Call" 
                else:
                    action = "Fold" 
    
    print("========= Hand Evaluation =========")
    print(f"Street:       {street_name}")
    print(f"Hero cards:   {' '.join(hero_cards) if hero_cards else 'none'}")
    print(f"Board cards:  {' '.join(board_cards) if board_cards else 'none'}")
    print(f"Hand type:    {hand_type}")
    print(f"Table size:   {num_players} players")
    print(f"Position:     {position}")
    print(f"Pot size:     {pot_size if pot_size is not None else 'unknown'}")
    print(f"Max bet:      {max_bet if max_bet is not None else 'unknown'}")
    print(f"Num bets:     {num_bets}\n")
    print(f"Pot equity:   {pot_equity:.4f}  ({pot_equity * 100:.2f}%)")
    print(f"Hero Equity:  {equity:.4f}  ({equity * 100:.2f}%)")
    print(f"Action:       {action}")
    print("===================================\n")

    # Equity buckets:
    # Best hands – Nutted hands we are prepared to stack off with
    # Good hands – Hands we want to value bet
    # Weak hands – Hands with some equity that we want to cheaply get to showdown with
    # Trash hands – Hands with very low equity that will only win the pot by bluffing