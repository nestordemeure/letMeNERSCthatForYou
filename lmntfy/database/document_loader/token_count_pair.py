class TokenCountPair:
    """
    Represent a pair of numbers that can be added as well as compared.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return TokenCountPair(self.x + other.x, self.y + other.y)

    def __lt__(self, other):
        return (self.x < other.x) and (self.y < other.y)
    
    def build_pair_counter(token_counter1, token_counter2):
        """takes two function, that each turn a string into a number of tokens, and return a function that turns a string into a TokenCountPair"""
        return lambda input: TokenCountPair(token_counter1(input), token_counter2(input))

    def __repr__(self):
        return f"TokenCountPair({self.x}, {self.y})"