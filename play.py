from chess.chessboard import Chessboard
from mainNode import MainNode
import mainNode

from config import *
from node import Node


def play(fen:str):
    state = Chessboard(fen)
    mcts_parent = MainNode(state=state)

    game_counter = 0
    while game_counter < games_per_iteration:
        print(f"game_counter:{game_counter}")
        state = mcts_parent.state = Chessboard(fen)
        mcts_parent_current = MainNode(state)
        [mcts_parent_current.children.append(c) for c in mcts_parent.children]
        #mcts_parent_current.children = mcts_parent.children
        while state.get_game_status() == 3:
            print(state)
            mcts_child = Node(
                state=mcts_parent.state,
                main_node=mcts_parent_current)
            mcts_child.run()
            mcts_child.print_selectively(2)
            best_move = max(mcts_child.children, key=lambda c: c.visits).move
            mcts_parent_current=mcts_child.update_tree(best_move)
        game_counter+=1
        print("Game Complete!")
    print("Done!")
    print(f"c_hit:{mainNode.c_hit}, c_miss:{mainNode.c_miss}")
    
    


def main():
    fen = "k7/8/8/8/8/8/8/7R w - 0 1"
    play(fen)

if __name__=="__main__":
    main()
