# models.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SeatState:
    seat_id: int
    name: str                    # "hero_bottom", etc.
    stack: Optional[float]       # chips behind
    bet: Optional[float]         # chips in front this street
    is_hero: bool
    has_cards: bool
    is_active: bool
    position: Optional[str]      # "BTN", "SB", "BB", "UTG", "MP", "CO"
    last_status: Optional[str] = None   # "fold", "check", "bet", "raise", "call"
    is_sitting_out: bool = False        

@dataclass
class TableState:
    street: str                        # "preflop", "flop", "turn", "river"
    hero_cards: List[str]
    board_cards: List[str]
    pot_size: float
    total_pot: float
    button_seat: Optional[int]         # 0..5 or None
    seats: List[SeatState]             # len = 6
