import time
from chess.chessboard import Chessboard
from mainNode import MainNode
import mainNode

from config import *
from nn_architecture import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork
from node import Node


def play(fen:str):
    model_config = NeuralNetwork(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE)
    model = model_config.build_nn()
    try:
        model.load_weights(checkpoint_path)
    except Exception as e:
        print('EXCEPTION, couldnt load weights ', e)
    
    state = Chessboard(fen)
    mcts_parent = MainNode(state=state, model=model)

    game_counter = 0
    time_start = time.time()
    while game_counter < games_per_iteration:
        print(f"game_counter:{game_counter}")
        state = mcts_parent.state = Chessboard(fen)
        mcts_parent_current = MainNode(state, model=model)
        [mcts_parent_current.children.append(c) for c in mcts_parent.children]
        while state.get_game_status() == 3:
            print(state)
            mcts_child = Node(
                state=mcts_parent.state,
                main_node=mcts_parent_current)
            mcts_child.run()
            best_move = max(mcts_child.children, key=lambda c: c.visits).move
            mcts_parent_current=mcts_child.update_tree(best_move)
        game_counter+=1
        print("Game Complete!")
        print(f"c_hit:{mainNode.c_hit}, c_miss:{mainNode.c_miss}, hitrate:{100*mainNode.c_hit/(mainNode.c_hit+mainNode.c_miss)}%")
    time_end = time.time()
    time_total = time_end-time_start
    print("Done!")
    print(f"time:{time_total}")
    
    


def main():
    fen = "k7/8/8/8/8/8/8/7R w - 0 1"
    play(fen)

if __name__=="__main__":
    main()
