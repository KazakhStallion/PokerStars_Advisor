from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SeatState:
    seat_id: int              # 0..5 for 6-max
    is_occupied: bool
    stack: Optional[float]    # chips behind
    bet: Optional[float]      # chips in front this street
    has_cards: bool           # player dealt-in
    is_hero: bool
    is_active: bool           # current turn?
    position: Optional[str]   # "BTN", "SB", "BB", "UTG", "MP", "CO"

@dataclass
class FrameState:
    hero_cards: List[str]                # ["Ah", "Kd"]
    board_cards: List[str]               # ["7c", "Qs", "Th", "4d", "2s"]
    pot_size: float
    main_pot: float
    side_pots: List[float]
    seats: List[SeatState]               # len=6
    street: str                          # "preflop", "flop", "turn", "river"
    bb_size: float
    table_ok: bool = True                # false if occluded / not detected
