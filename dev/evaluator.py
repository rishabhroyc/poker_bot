import random
import time
from typing import Dict, List, Sequence, Tuple, Union
import numpy as np

# -------------------------------------------------------------------------
# Card Representations & Constants
# -------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"

# Card integer: (rank << 2) | suit
# rank in 0..12 (2..A), suit in 0..3 (s, h, d, c)
CARD_TO_INT = {r + s: (RANKS.index(r) << 2) | SUITS.index(s) for r in RANKS for s in SUITS}
INT_TO_CARD = {v: k for k, v in CARD_TO_INT.items()}

HAND_NAMES = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)

# A-2-3-4-5 wheel straight mask: Ace (bit 12) + Deuce..5 (bits 0, 1, 2, 3)
WHEEL_MASK = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)  # 0x100F


# -------------------------------------------------------------------------
# Hand Evaluation Data Structure
# -------------------------------------------------------------------------
class HandEvaluation:
    """
    Comparable hand evaluation result.
    
    Attributes:
        rank_tuple: Comparable tuple matching official engine format:
                    e.g. (6, 14, 13) for Aces full of Kings, (0, 14, 12, 10, 8, 4) for A-high.
        score: Packed 32-bit integer for fast comparisons (higher is strictly better).
        category: Hand category ID in 0..8 (0: High Card, ..., 8: Straight Flush).
        category_name: Human-readable hand type name.
    """

    __slots__ = ("rank_tuple", "score", "category", "category_name")

    def __init__(self, rank_tuple: Tuple[int, ...], score: int):
        self.rank_tuple = rank_tuple
        self.score = score
        self.category = rank_tuple[0]
        self.category_name = HAND_NAMES[self.category]

    def __lt__(self, other: "HandEvaluation") -> bool:
        return self.score < other.score

    def __gt__(self, other: "HandEvaluation") -> bool:
        return self.score > other.score

    def __le__(self, other: "HandEvaluation") -> bool:
        return self.score <= other.score

    def __ge__(self, other: "HandEvaluation") -> bool:
        return self.score >= other.score

    def __eq__(self, other: object) -> bool:
        if isinstance(other, HandEvaluation):
            return self.score == other.score
        return False

    def __repr__(self) -> str:
        return f"<HandEvaluation: {self.category_name} (score={self.score}, ranks={self.rank_tuple})>"

    def __iter__(self):
        # Support tuple unpacking: t, score = eval_showdown(...)
        return iter((self.rank_tuple, self.score))


# -------------------------------------------------------------------------
# Low-Level Bitwise Helper Functions
# -------------------------------------------------------------------------
def parse_card(c: Union[str, int]) -> int:
    """Convert 'As' -> 48, or validate integer card ID."""
    return CARD_TO_INT[c] if isinstance(c, str) else int(c)


def parse_cards(cards: Sequence[Union[str, int]]) -> List[int]:
    """Convert sequence of cards to list of integer card IDs."""
    return [parse_card(c) for c in cards]


def get_straight_high(mask: int) -> int:
    """
    Detect highest straight in a 13-bit rank mask.
    Returns the straight's high rank value (5..14), or 0 if no straight.
    """
    # 5 consecutive 1-bits: mask & (mask >> 1) & ... & (mask >> 4)
    s = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    if s > 0:
        high_idx = (s).bit_length() - 1 + 4
        return high_idx + 2  # Rank values: 2=2 .. A=14
    # Special case: A-2-3-4-5 wheel straight
    if (mask & WHEEL_MASK) == WHEEL_MASK:
        return 5
    return 0


def top_n_ranks(mask: int, n: int) -> Tuple[int, ...]:
    """Extract top n rank values (2..14) from a 13-bit rank mask."""
    res = []
    m = mask
    while m > 0 and len(res) < n:
        bit = m.bit_length() - 1
        res.append(bit + 2)
        m &= ~(1 << bit)
    return tuple(res)


def pack_score(rank_tuple: Tuple[int, ...]) -> int:
    """
    Pack rank tuple into an order-preserving 32-bit integer:
      Bits 20..23: Category (0..8)
      Bits 16..19: Rank 1 (0..14)
      Bits 12..15: Rank 2 (0..14)
      Bits  8..11: Rank 3 (0..14)
      Bits  4..7 : Rank 4 (0..14)
      Bits  0..3 : Rank 5 (0..14)
    """
    category = rank_tuple[0]
    ranks = rank_tuple[1:]
    val = category << 20
    for i, r in enumerate(ranks):
        val |= r << (16 - 4 * i)
    return val


# -------------------------------------------------------------------------
# Core Integer Bitmask Hand Evaluator (5, 6, or 7 cards)
# -------------------------------------------------------------------------
def eval_cards(cards: Sequence[Union[str, int]]) -> HandEvaluation:
    """
    Fast bitwise evaluator for any 5, 6, or 7 card hand.
    Eliminates all itertools.combinations and Counter overhead.
    """
    int_cards = parse_cards(cards)

    # 1. Tally suits and ranks using bitwise masks
    suit_masks = [0, 0, 0, 0]
    suit_counts = [0, 0, 0, 0]
    rank_counts = [0] * 13

    for c in int_cards:
        r = c >> 2
        s = c & 3
        suit_masks[s] |= 1 << r
        suit_counts[s] += 1
        rank_counts[r] += 1

    # 2. Flush check: At most one suit can have >= 5 cards in <= 7 cards
    flush_suit = -1
    for s in range(4):
        if suit_counts[s] >= 5:
            flush_suit = s
            break

    if flush_suit != -1:
        f_mask = suit_masks[flush_suit]
        str_high = get_straight_high(f_mask)
        if str_high > 0:
            t = (8, str_high)  # Straight Flush
        else:
            t = (5,) + top_n_ranks(f_mask, 5)  # Flush
        return HandEvaluation(t, pack_score(t))

    # 3. Non-flush hands: Analyze rank counts via bitmasks
    ranks_mask = 0
    pairs_mask = 0
    trips_mask = 0
    quads_mask = 0

    for r in range(13):
        cnt = rank_counts[r]
        if cnt >= 1:
            ranks_mask |= 1 << r
        if cnt >= 2:
            pairs_mask |= 1 << r
        if cnt >= 3:
            trips_mask |= 1 << r
        if cnt == 4:
            quads_mask |= 1 << r

    # Four of a Kind
    if quads_mask > 0:
        q_rank = quads_mask.bit_length() - 1
        kicker = (ranks_mask & ~(1 << q_rank)).bit_length() - 1
        t = (7, q_rank + 2, kicker + 2)

    # Full House (or Trips)
    elif trips_mask > 0:
        t_high = trips_mask.bit_length() - 1
        rem_trips = trips_mask & ~(1 << t_high)
        if rem_trips > 0:
            pair_rank = rem_trips.bit_length() - 1
            t = (6, t_high + 2, pair_rank + 2)
        else:
            rem_pairs = pairs_mask & ~(1 << t_high)
            if rem_pairs > 0:
                pair_rank = rem_pairs.bit_length() - 1
                t = (6, t_high + 2, pair_rank + 2)
            else:
                str_high = get_straight_high(ranks_mask)
                if str_high > 0:
                    t = (4, str_high)  # Straight
                else:
                    kickers = top_n_ranks(ranks_mask & ~(1 << t_high), 2)
                    t = (3, t_high + 2) + kickers  # Three of a Kind

    # Straight
    elif (str_high := get_straight_high(ranks_mask)) > 0:
        t = (4, str_high)

    # Two Pair
    elif pairs_mask > 0:
        p1 = pairs_mask.bit_length() - 1
        rem_p = pairs_mask & ~(1 << p1)
        if rem_p > 0:
            p2 = rem_p.bit_length() - 1
            kicker = (ranks_mask & ~((1 << p1) | (1 << p2))).bit_length() - 1
            t = (2, p1 + 2, p2 + 2, kicker + 2)
        else:
            kickers = top_n_ranks(ranks_mask & ~(1 << p1), 3)
            t = (1, p1 + 2) + kickers  # One Pair

    # High Card
    else:
        t = (0,) + top_n_ranks(ranks_mask, 5)

    return HandEvaluation(t, pack_score(t))


# -------------------------------------------------------------------------
# High-Level Evaluator Interface for Discard Hold'em
# -------------------------------------------------------------------------
def eval_showdown(
    hole_2: Sequence[Union[str, int]], board_5: Sequence[Union[str, int]]
) -> HandEvaluation:
    """
    Evaluates the showdown hand (2 hole cards + 5 board cards).
    Returns HandEvaluation representing the best 5-card hand.
    """
    all_7 = parse_cards(hole_2) + parse_cards(board_5)
    return eval_cards(all_7)


def eval_pre_discard(
    hole_3: Sequence[Union[str, int]], board_3: Sequence[Union[str, int]]
) -> Tuple[HandEvaluation, Union[str, int], Dict[Union[str, int], HandEvaluation]]:
    """
    Evaluates the 3-hole-card + 3-board-card flop state right before the discard.
    
    Evaluates the 3 possible choices of which hole card to discard (leaving 2 hole + 3 board).
    
    Returns:
        best_evaluation: HandEvaluation of the strongest resulting 5-card made hand.
        best_discard: The hole card (preserving input type) that yields the strongest made hand.
        discard_details: Dict mapping each discard candidate -> resulting HandEvaluation.
    """
    h = list(hole_3)
    b = parse_cards(board_3)

    best_eval: Optional[HandEvaluation] = None
    best_discard = h[0]
    discard_details: Dict[Union[str, int], HandEvaluation] = {}

    for i in range(3):
        discard_c = h[i]
        kept_hole = [parse_card(h[j]) for j in range(3) if j != i]
        candidate_5 = kept_hole + b
        res = eval_cards(candidate_5)
        discard_details[discard_c] = res

        if best_eval is None or res.score > best_eval.score:
            best_eval = res
            best_discard = discard_c

    return best_eval, best_discard, discard_details


# -------------------------------------------------------------------------
# Self-Test and Benchmarks
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("Evaluator Self-Test & Benchmark vs Naive Combinations")
    print("=" * 65)

    try:
        from engine import best_hand as engine_best_hand
        has_engine = True
    except ImportError:
        import sys
        sys.path.insert(0, ".")
        from engine import best_hand as engine_best_hand
        has_engine = True

    # 1. Correctness validation against official engine.py on 5,000 hands
    print("\n1. Validating correctness against official starter kit engine.py...")
    random.seed(42)
    mismatches = 0
    n_verify = 5000
    for _ in range(n_verify):
        deck = list(range(52))
        random.shuffle(deck)
        hand7 = deck[:7]
        h_hole = [INT_TO_CARD[c] for c in hand7[:2]]
        h_board = [INT_TO_CARD[c] for c in hand7[2:]]

        eng_res = engine_best_hand(h_hole, h_board)
        fast_res = eval_showdown(h_hole, h_board)

        if eng_res != fast_res.rank_tuple:
            print(f"Mismatch: hole={h_hole}, board={h_board}")
            print(f"Engine: {eng_res}, Fast: {fast_res.rank_tuple}")
            mismatches += 1
            break

    if mismatches == 0:
        print(f"   SUCCESS: {n_verify:,}/{n_verify:,} hands matched engine.py exactly!")

    # 2. Performance benchmark
    print("\n2. Benchmarking execution speed on 10,000 7-card hands...")
    hands_7_str = []
    hands_7_int = []
    for _ in range(10000):
        deck = list(range(52))
        random.shuffle(deck)
        hand = deck[:7]
        hands_7_int.append(hand)
        hands_7_str.append(([INT_TO_CARD[c] for c in hand[:2]], [INT_TO_CARD[c] for c in hand[2:]]))

    # Naive combinations benchmark (from engine.py)
    n_bench_naive = 2000
    t0 = time.perf_counter()
    for hole, board in hands_7_str[:n_bench_naive]:
        engine_best_hand(hole, board)
    t_naive = time.perf_counter() - t0
    us_naive = (t_naive / n_bench_naive) * 1e6

    # Fast bitwise benchmark
    n_bench_fast = 10000
    t0 = time.perf_counter()
    for hand in hands_7_int:
        eval_cards(hand)
    t_fast = time.perf_counter() - t0
    us_fast = (t_fast / n_bench_fast) * 1e6

    speedup = us_naive / us_fast
    hands_per_sec = n_bench_fast / t_fast

    print(f"   Naive combinations (engine.py): {us_naive:.2f} us/hand")
    print(f"   Fast Bitwise Evaluator:         {us_fast:.2f} us/hand ({hands_per_sec:,.0f} hands/sec)")
    print(f"   Speedup:                        {speedup:.1f}x FASTER!")

    # 3. Demonstration of eval_pre_discard
    print("\n3. Demonstrating eval_pre_discard() for Discard Hold'em:")
    sample_hole = ["As", "Kd", "7c"]
    sample_flop = ["Ah", "7d", "2s"]
    best_ev, best_disc, details = eval_pre_discard(sample_hole, sample_flop)

    print(f"   Hole: {sample_hole} | Flop: {sample_flop}")
    for card, ev in details.items():
        print(f"   - Discarding {card:<2}: Yields {ev.category_name:<16} (ranks={ev.rank_tuple})")
    print(f"   => Optimal immediate discard: {best_disc} (Yields {best_ev.category_name})")
    print("=" * 65)

