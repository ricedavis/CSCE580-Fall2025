from collections import deque


class MCAgent:

    def __init__(self):
        pass

    def solve(self, initial_missionaries, initial_cannibals):
    # My changes are noted throughout the code
        class States:
            def __init__(self, left_missionaries, left_cannibals,
                         right_missionaries, right_cannibals, boat_position):
                self.left_missionaries = left_missionaries
                self.left_cannibals = left_cannibals
                self.right_missionaries = right_missionaries
                self.right_cannibals = right_cannibals
                self.boat_position = boat_position
                self.parent = None

            def __eq__(self, other):
                return (self.left_missionaries == other.left_missionaries and
                        self.left_cannibals == other.left_cannibals and
                        self.right_missionaries == other.right_missionaries and
                        self.right_cannibals == other.right_cannibals and
                        self.boat_position == other.boat_position)

            def goal_state(self):
                # Goal: all missionaries and cannibals are on the right
                if (self.left_missionaries == 0 and self.left_cannibals == 0 and
                    self.right_missionaries == initial_missionaries and
                    self.right_cannibals == initial_cannibals and
                        self.boat_position == "right"):
                    return True
                else:
                    return False

            def valid_state(self):
                # Check that missionaries are never outnumbered
                if (self.left_missionaries != 0 and
                        self.left_cannibals > self.left_missionaries) \
                        or (self.right_missionaries != 0 and
                            self.right_cannibals > self.right_missionaries) \
                        or self.left_missionaries < 0 or self.left_cannibals < 0 \
                        or self.right_missionaries < 0 or self.right_cannibals < 0:
                    return False
                else:
                    return True

        def successors(curr_state):
            successor = []
            # Five possible moves: 2M, 2C, 1M+1C, 1M, or 1C
            possible_moves = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
            if curr_state.boat_position == "left":  # Move boat left -> right
                for move in possible_moves:
                    new_state = States(curr_state.left_missionaries - move[0],
                                       curr_state.left_cannibals - move[1],
                                       curr_state.right_missionaries + move[0],
                                       curr_state.right_cannibals + move[1],
                                       "right")
                    if new_state.valid_state():
                        successor.append(new_state)
                        new_state.parent = curr_state
            else:  # Move boat right -> left
                for move in possible_moves:
                    new_state = States(curr_state.left_missionaries + move[0],
                                       curr_state.left_cannibals + move[1],
                                       curr_state.right_missionaries - move[0],
                                       curr_state.right_cannibals - move[1],
                                       "left")
                    if new_state.valid_state():
                        successor.append(new_state)
                        new_state.parent = curr_state
            return successor

        # -------------------------------------------------------------------
        # CHANGE: Implemented Depth-First Search (DFS) instead of BFS
        # -------------------------------------------------------------------
        def dfs():
            initial_state = States(initial_missionaries, initial_cannibals, 0, 0, "left")
            if initial_state.goal_state():
                return initial_state
            stack = [initial_state]   # LIFO stack for DFS
            explored = []
            while stack:
                node = stack.pop()    # Pop from stack (deepest node first)
                if node.goal_state():
                    return node
                explored.append(node)
                node_children = successors(node)
                for child in node_children:
                    if (child not in explored) and (child not in stack):
                        stack.append(child)
            return None
        # -------------------------------------------------------------------

        def find_moves(result):
            path = []
            final_path = []
            result_parent = result.parent
            while result_parent:
                move = (abs(result.left_missionaries - result_parent.left_missionaries),
                        abs(result.left_cannibals - result_parent.left_cannibals))
                path.append(move)
                result = result_parent
                result_parent = result.parent
            for i in range(len(path)):
                final_result = path[len(path) - 1 - i]
                final_path.append(final_result)
            return final_path

        # -------------------------------------------------------------------
        # CHANGE: use dfs() instead of bfs()
        # -------------------------------------------------------------------
        solution = dfs()
        if solution:
            return find_moves(solution)
        else:
            return []
