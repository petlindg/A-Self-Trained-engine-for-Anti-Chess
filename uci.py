import sys
from chess import Chessboard, algebraic_to_move, move_to_algebraic
from keras import Model

import tensorflow
from multiprocessing import Queue
from node import Node
from nn_architecture import INPUT_SHAPE, OUTPUT_SHAPE, NeuralNetwork
from Game.EnginePlayer import EnginePlayer



"""
Implement the UCI protocol as publiced by Stefan-Meyer Kahlen (ShredderChess)

See uci-protocol.txt. Summary below.

GUI to engine:
    [DONE] uci
    [SKIP] debug [ on | off ]
    isready
    [SKIP] setoption name  [value ]
    [SKIP] register
    [DONE] ucinewgame
    [DONE] position [fen  | startpos ]  moves  ....
    go ....
    [SKIP] stop
    [SKIP] ponderhit
    [DONE] quit

Engine to GUI:
    [DONE] id name author
    [DONE] uciok
    [DONE] readyok
    [DONE] bestmove  [ ponder  ]
    [SKIP] copyprotection
    [SKIP] registration
    [SKIP] info ....
    [SKIP] option ....
"""



class EngineOptions:
    def __init__(self):
        self.options = {
            "Protocol type": "uci",
            "Debug Log File": "",
            "Threads": 1,
            "Hash": 16,  # in MB
            "Ponder": False,
            "MultiPV": 1,
            "Skill Level": 20,
            "Move Overhead": 10,
            "Slow Mover": 100,
            "nodestime": 0,
            "UCI_Chess960": False,
            "UCI_Variant": "var antichess",
            "VariantPath": " "
        }

    def set_option(self, name, value):
        if name in self.options:
            self.options[name] = value
            print(f"Option {name} set to {value}")
        else:
            print(f"Unknown option: {name}")

    def get_option(self, name):
        return self.options.get(name, None)

# Initialize the options object
options = EngineOptions()




def uci_command(command: str, board: Chessboard, ai: EnginePlayer, options: EngineOptions):
    tokens = command.split()
    match tokens[0]:
        case "uci":
            print("id name G80")
            print("id author G80")
            # List all options
            for name, value in options.options.items():
                if isinstance(value, bool):
                    print(f"option name {name} type check default {str(value).lower()}")
                elif isinstance(value, int):
                    # Handle ranges like Threads, Hash, etc.
                    if name == "Threads":
                        print(f"option name {name} type spin default {value} min 1 max 512")
                    elif name == "Hash":
                        print(f"option name {name} type spin default {value} min 1 max 33554432")
                    else:
                        print(f"option name {name} type spin default {value}")
                elif isinstance(value, str):
                    print(f"option name {name} type string default {value}")
            print("uciok")

        case "setoption":
            if "name" in tokens and "value" in tokens:
                name_index = tokens.index("name") + 1
                value_index = tokens.index("value") + 1
                option_name = " ".join(tokens[name_index:tokens.index("value")])
                option_value = " ".join(tokens[value_index:])
                options.set_option(option_name, option_value)

        case "ucinewgame":
            board.__init__()
        
        case "isready":
            print("readyok")

        case "position":
            
            moves_index = tokens.index("moves") if "moves" in tokens else len(tokens)
            moves = tokens[moves_index + 1:] if "moves" in tokens else []
            if "startpos" in tokens:
                board.__init__()
                
                
            elif "fen" in tokens:
                fen_index = tokens.index("fen") + 1
                
                
                
                fen = ' '.join(tokens[fen_index:moves_index])
                
            else:
                board.__init__()
                

            for move_str in moves:
                move = algebraic_to_move(move_str)
                if move:
                    board.move(move)
                    
                    print(f"Applied move: {move_str}")
                else:
                    print(f"Invalid move detected: {move_str}")
                    break
            print(board)

        case "go":
            handle_go_command
            best_move = ai.get_move()
            if board.is_valid_move(best_move):
                board.move(best_move)
                ai.update(best_move)
                print(f"bestmove {best_move}")
            else:
                print("invalid move")
            #print(board)

        case "quit":
            import sys
            sys.exit()

        case _:
            print(f"Unknown command: {tokens[0]}")
def handle_go_command(command):
    parts = command.split()
    # List of parameters to remove
    parameters_to_remove = ['depth', 'movetime']

    
    for param in parameters_to_remove:
        while param in parts:
            try:
                index = parts.index(param)
                
                if index + 1 < len(parts) and parts[index + 1].isdigit():
                    
                    del parts[index:index+2]
                else:
                   
                    del parts[index]
            except ValueError:
                print(f"Error: '{param}' not followed by a number")

 
    new_command = ' '.join(parts)
    return new_command


# Main function for initializing and using the engine
if __name__ == "__main__":
    chessboard = Chessboard()
    options = EngineOptions()

    try:
        model = tensorflow.keras.models.load_model(r'C:\Users\abdul\OneDrive\Documents\GitHub\A-Self-Trained-engine-for-Anti-Chess\saved_model\model_320_it.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")

    model.compile()

    while True:
        command = input().strip()
        engine_player = EnginePlayer(chessboard, model)
        uci_command(command, chessboard, engine_player, options)
