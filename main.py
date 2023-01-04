import numpy as np
import math
import traceback
import sys


class Graph:

    def __init__(self, start_state: np.ndarray, finish_state: np.ndarray, dimensions: dict[str:int]):
        self.dimensions = dimensions
        self.combinations = math.factorial(dimensions['n']+dimensions['p']-1) / math.factorial(dimensions['p']-1)
        self.visited: list[np.ndarray] = []
        self.start = self.Node(graph=self, state=start_state, past_state={})
        self.finish = finish_state

    def future_reveal(self, n: int):
        self.start.future_reveal(n)

    def print(self):
        self.start.travel()

    def get_discovered(self) -> int:
        return len(self.visited)

    def get_combiations(self) -> int:
        return self.combinations

    def depth_first_search(self):
        return self.start.DFS()

    class Node:
    
        def __init__(self, graph: 'Graph', state: np.ndarray, past_state: dict[tuple:any]):
            self.graph = graph
            self.graph.visited.append(state)
            self.state: np.ndarray = state
            self.possible_future_paths: list[tuple] = [(p, r) for r in range(self.graph.dimensions['p']) for p in range(self.graph.dimensions['p'])]
            self.past = past_state
            self.future = None


        def valid_move(self, p: int, r: int) -> bool:
            if p >= self.graph.dimensions['p'] or p < 0 or r >= self.graph.dimensions['p'] or r < 0:
                return False
            return True


        def move(self, p: int, r: int) -> np.ndarray:

            if not self.valid_move(p, r): raise Exception("Invalid move. Index p and/or r out of bounds of matrix.")


            substitute_column_ndarray: np.ndarray = self.state[:, p]
            substitute_column: list = substitute_column_ndarray.tolist()
            try:
                # get the block at position p
                sub_index = substitute_column.index(next(x for x in substitute_column if x != ' '))
                sub_element = substitute_column[sub_index]
                substitute_column[sub_index] = ' '


                # get the location for the block to move to AND move it
                place_column_ndarray: np.ndarray = self.state[:, r]
                place_column: list = place_column_ndarray.tolist()
                place_index = len(place_column) - place_column[::-1].index(' ') - 1
                place_column[place_index] = sub_element

                # create a future instance of present state of node and return
                state_copy: np.ndarray = np.copy(self.state)
                state_copy[:, p] = substitute_column
                state_copy[:, r] = place_column

                return state_copy
            except Exception as e:
                tb = traceback.extract_tb(sys.exc_info()[2])
                print(f'Line {tb[-1][1]}, in {tb[-1][2]}')
                print(e)



        # The BREADTH part of BFS
        def future_step(self):

            # remove moves that create a ciclic connection OR
            # performe a move on an empty box OR 
            # form a connection to the past OR 
            # future state has already been visited
            legal_future_paths: list[tuple] = [
                (p, r) for p, r in self.possible_future_paths
                if  self.valid_move(p, r) and
                    p!=r and
                    not all(x == ' ' for x in self.state[:, p]) and any([x == ' ' for x in self.state[:, r]]) and
                    all([not np.array_equal(self.move(p, r), visited) for visited in self.graph.visited])
            ]

            

            # print(legal_future_paths)
            # create future scenarios created by all the legal moves 
            # 1st argument: next (future) Nodes
            # 2nd argument: make a reference to parent(this) node
            # 3rd argument: pass existing dimensions(DNA hehe)
            self.future = {(p, r): self.graph.Node(self.graph, self.move(p, r), {(p, r): self}) for p, r in legal_future_paths}


        def travel(self):

            print(self.state,'\n')
            if (self.future == None): return

            for possible_future in self.future.values():
                possible_future.travel()


        
        def future_reveal(self, future_depth: int = 2):

            if future_depth == 0: return

            if self.future == None: self.future_step()
            for possible_future in self.future.values():
                possible_future.future_reveal(future_depth-1)


        def DFS(self, max_depth = 1000000):

            if max_depth == 0: return ["zmanjka globine"]

            if np.array_equal(self.state, self.graph.finish):
                return self.reconstruct_path()
            
            if self.future == None: self.future_step()
            for possible_future in self.future.values():
                value = possible_future.DFS(max_depth-1)
                if value: return value
            


        def reconstruct_path(self):
            
            if (self.past == {}):
                return []

            past_node_pair = next(iter(self.past.items()))
            move = past_node_pair[0]

            node: self.graph.Node = past_node_pair[1]

            return node.reconstruct_path() + [move]


        def BFS(self, max_depth = 10):
            
            if max_depth == 0: return ["zmanjka globine"]

            if self.future == None: self.future_step()
            for possible_future in self.future.values():
                if np.array_equal(possible_future.state, self.graph.finish):
                    return possible_future.reconstruct_path()
            for possible_future in self.future.values():
                possible_future.BFS(max_depth-1)


            
            


def get_dimensions(state: np.ndarray) -> dict[str:int]:
    p: int = len(state[0])
    n: int = len(np.where(state != ' ')[0])
    return {'n': n, 'p': p}


def show_space_complexity(start_array: np.ndarray, final_array: np.ndarray, n: int = 100):
    test_graph = Graph(start_array, final_array, get_dimensions(start_array))
    i = 0
    while i < n:
        test_graph.future_reveal(i)
        discovered = test_graph.get_discovered()
        combinations = test_graph.get_combiations()
        percetage_found = round(discovered/combinations, 4)*100
        print(
            f"""Depth of search: {i},
            positions discovered: {discovered},
            possible_combinations: {combinations},
            disovered(%): {percetage_found} %\n"""
        )
        if (discovered == combinations): break
        i+=1
    else:
        print("Runtime exited. Didn't finish in {n} iterations.")


def read_examples(dir_location: str = "./primeri/") -> list[np.ndarray, np.ndarray]:
    
    prefix = dir_location+"primer"
    final_suffix = "_koncna.txt"
    start_suffix = "_zacetna.txt"

    examples = []

    for i in range(1, 6):
        try:
            file_name = prefix+str(i)+start_suffix
            start_position = read_example(file_name)
        except:
            raise Exception(f'Trouble handling {file_name}')
        try:
            file_name = prefix+str(i)+final_suffix
            final_position = read_example(file_name)
        except:
            raise Exception(f'Trouble handling {file_name}')
        examples.append((start_position, final_position))

    return examples


def read_example(file_name: str) -> np.ndarray:
    with open(file_name) as file:
        string_lines: list[str] = file.readlines()
        lines = [[element[1:-1] for element in line.strip().split(',')] for line in string_lines]
        state = np.array(lines)
    file.close()
    return state



examples = read_examples()

index = 0
for example in examples:
    index+=1
    print(str(index)+" primer")
    #print("start:\n"+ str(example[0]) + ",\n\n end:\n"+ str(example[1]))
    test = Graph(start_state=example[0], finish_state=example[1], dimensions=get_dimensions(example[0]))
    print(test.depth_first_search())
    print()