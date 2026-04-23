import io
import chess.pgn

pgn8 = """
1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 g6 6. Be3 Bg7 7. f3 O-O 8. Bc4
Nc6 9. Qd2 Nxd4 10. Bxd4 Be6 11. Bxe6 fxe6 12. O-O-O Qa5 13. h4 Rac8 14. Kb1 Nh5
15. Bxg7 Kxg7 16. Ne2 Qxd2 17. Rxd2 e5 18. b3 Nf4 19. Nxf4 exf4 20. Rhd1 Rc5 21.
Rd5 Rfc8 22. c4 Kf6 23. Kb2 a5 24. Rxc5 Rxc5 25. Rd5 Rxd5 26. exd5 g5 27. Kc3
Kf5 28. Kd4 g4 29. a3 b6 30. b4 axb4 31. axb4 h6 32. Kd3 gxf3 33. gxf3 Ke5 34.
b5 Kf5 35. Kc3 Ke5 36. Kd3 Kf5 37. Kc3 Ke5 38. Kd3 1/2-1/2
"""

pgn9 = """
1. e4 e6 2. d4 d5 3. e5 c5 4. c3 Qb6 5. Nf3 Bd7 6. a3 a5 7. Be2 Bb5 8. O-O Bxe2
9. Qxe2 c4 10. a4 Nd7 11. Nbd2 Qc6 12. Re1 Nb6 13. Qd1 Ne7 14. b3 cxb3 15. Qxb3
Nc4 16. Rb1 Rb8 17. Qb5 Qxb5 18. Rxb5 Nxd2 19. Bxd2 Nc6 20. Reb1 Ba3 21. Rxb7
Rxb7 22. Rxb7 O-O 23. Kf1 h6 24. Ke2 Rc8 25. Ne1 Kf8 26. Nd3 g5 27. h4 Be7 28.
h5 Nd8 29. Rb5 Nc6 30. f3 Rc7 31. Nf2 Kg7 32. Ng4 Na7 33. Rxa5 Nc6 34. Ra8 Rb7
35. Kd3 Rb1 36. Ra6 Nd8 37. Kc2 Rg1 38. Ne3 Rh1 39. g4 Rh2 40. Rb6 1-0
"""


def game_to_uci_list(pgn: str) -> list[str]:
    g = chess.pgn.read_game(io.StringIO(pgn))
    out: list[str] = []
    for node in g.mainline():
        out.append(node.move.uci())
    return out


def print_pairs(moves: list[str], name: str) -> None:
    print(f"{name}, total {len(moves)} halfmoves")
    for i in range(0, len(moves), 2):
        w = moves[i]
        if i + 1 < len(moves):
            print(f'        ("{w}", "{moves[i+1]}"),')
        else:
            print(f'        # ORPHAN: ("{w}",)')


if __name__ == "__main__":
    import chess

    m8 = game_to_uci_list(pgn8)
    m9 = game_to_uci_list(pgn9)
    print_pairs(m8, "8")
    print()
    print_pairs(m9, "9")

    def emit(name: str, mlist: list[str]) -> None:
        n = len(mlist) // 2
        orphan = mlist[2 * n :]
        print()
        print(f"--- EMIT {name} ---")
        for i in range(n):
            w, b_ = mlist[2 * i], mlist[2 * i + 1]
            print(f'        ("{w}", "{b_}"),')
        if orphan:
            print(f"    # final half-move: {orphan!r}")

    emit("8", m8)
    emit("9", m9)

    for name, mlist in [("8", m8), ("9", m9)]:
        b = chess.Board()
        for u in mlist:
            mv = chess.Move.from_uci(u)
            if mv not in b.legal_moves:
                print(f"VERIFY FAIL {name} at {u} FEN {b.fen()}")
                break
            b.push(mv)
        else:
            print(f"VERIFY OK {name}, final: {b.fen()[:60]}...")
