from enum import Enum, auto
from typing import Optional, Any
from abc import ABC, abstractmethod

import time
import sys


class CommandListener(ABC):
    @abstractmethod
    def tr(self, n):
        pass

    @abstractmethod
    def tl(self, n):
        pass

    @abstractmethod
    def inc(self, n):
        pass

    @abstractmethod
    def dec(self, n):
        pass

    @abstractmethod
    def ob(self, to):
        pass

    @abstractmethod
    def cb(self, to):
        pass

    @abstractmethod
    def gc(self, n):
        pass

    @abstractmethod
    def pc(self, n):
        pass


class Interpreter(CommandListener):
    def __init__(self, mem_size):
        super().__init__()

        self.mem = bytearray(mem_size)
        self.memmax = 0
        self.memptr = 0

    def tr(self, n):
        for _ in range(n):
            self.memptr += 1
            if self.memptr > self.memmax:
                self.memmax = self.memptr

    def tl(self, n):
        for _ in range(n):
            self.memptr -= 1

    def inc(self, n):
        for _ in range(n):
            if self.memptr >= 0 and self.memptr < self.mem_size:
                # memory over/underflows without warning
                self.mem[self.memptr] = (self.mem[self.memptr] + 1) % 256

            else:
                print(f"ERROR: memory index out of range at {self.memptr}")
                exit(1)

    def dec(self, n):
        for _ in range(n):
            if self.memptr >= 0 and self.memptr < self.mem_size:
                # memory over/underflows without warning
                self.mem[self.memptr] = (self.mem[self.memptr] - 1) % 256

            else:
                print(f"ERROR: memory index out of range at {self.memptr}")
                exit(1)

    def ob(self, to):
        if self.memptr >= 0 and self.memptr < self.mem_size:
            # jump back to beginning of loop if mem is not 0
            if self.mem[self.memptr] == 0:
                return to

        else:
            print(f"ERROR: memory index out of range at {self.memptr}")
            exit(1)

    def cb(self, to):
        if self.memptr >= 0 and self.memptr < self.mem_size:
            # jump back to beginning of loop if mem is not 0
            if self.mem[self.memptr] != 0:
                return to

        else:
            print(f"ERROR: memory index out of range at {self.memptr}")
            exit(1)

    def gc(self, n):
        for _ in range(n):
            char = sys.stdin.read(1)
            # char = input()[0]
            if self.memptr >= 0 and self.memptr < self.mem_size:
                self.mem[self.memptr] = ord(char)
            else:
                print(f"ERROR: memory index out of range at {self.memptr}")
                exit(1)

    def pc(self, n):
        for _ in range(n):
            if self.memptr >= 0 and self.memptr < self.mem_size:
                print(chr(self.mem[self.memptr]), end="")
            else:
                print(f"ERROR: memory index out of range at {self.memptr}")
                exit(1)


class TokenType(Enum):
    TapeLeft = auto()
    TapeRight = auto()
    Inc = auto()
    Dec = auto()
    OpenBracket = auto()
    CloseBracket = auto()
    GetChar = auto()
    PutChar = auto()
    Unknown = auto()


class Token:
    def __init__(self, ttype: TokenType, value: Optional[Any] = None):
        self.ttype = ttype
        self.value = value

    def __repr__(self):
        return f"Token({str(self.ttype)}, {str(self.value)})"


class Tracker:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.ptr = 0

    def peek(self):
        if self.has_next():
            return self.tokens[self.ptr]

    def consume(self):

        r = self.tokens[self.ptr]
        self.ptr += 1
        return r

    def has_next(self):
        return self.ptr < len(self.tokens)


def time_function(f):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        f(*args, **kwargs)
        t2 = time.perf_counter()
        print(f"{f.__name__} took {t2-t1} seconds to execute")

    return wrapper


def tokenize(code: str) -> list[Token]:
    tokens: list[Token] = []
    ptr = 0
    while ptr < len(code):
        # TODO: hashmap?
        match code[ptr]:
            case ">":
                tokens.append(Token(TokenType.TapeRight))
            case "<":
                tokens.append(Token(TokenType.TapeLeft))
            case "+":
                tokens.append(Token(TokenType.Inc))
            case "-":
                tokens.append(Token(TokenType.Dec))
            case "[":
                tokens.append(Token(TokenType.OpenBracket))
            case "]":
                tokens.append(Token(TokenType.CloseBracket))
            case ",":
                tokens.append(Token(TokenType.GetChar))
            case ".":
                tokens.append(Token(TokenType.PutChar))
            case _:
                # way conditional loops are used to make comments in bf, checking for the validity of tokens
                # at parse time seems to be impossible

                # error is thrown at the first unknown token, so we can ignore the tail of unknown sequences
                if tokens[-1].ttype != TokenType.Unknown:
                    tokens.append(Token(TokenType.Unknown, code[ptr]))

        ptr += 1
    return tokens


def contract_expressions(tokens: list[Token]) -> list[Token]:
    new_tokens: list[Token] = []
    track = Tracker(tokens)
    while track.has_next():
        t = track.consume()
        if t.ttype != TokenType.OpenBracket and t.ttype != TokenType.CloseBracket:
            count = 1
            while track.has_next() and t.ttype == track.peek().ttype:
                count += 1
                track.consume()
            t.value = count
        new_tokens.append(t)
    return new_tokens


def match_brackets(tokens: list[Token]) -> list[Token]:
    ptr = 0
    stack = []

    while ptr < len(tokens):
        if tokens[ptr].ttype == TokenType.OpenBracket:
            stack.append(ptr)
        if tokens[ptr].ttype == TokenType.CloseBracket:
            if len(stack) > 0:
                open_b_address = stack.pop()
                tokens[ptr].value = open_b_address
                tokens[open_b_address].value = ptr
            else:
                print(f"ERROR: you cannot close more brackets that you open")
                exit(1)
        ptr += 1
    if len(stack) > 0:
        print(f"ERROR: brackets that have been opened need to be closed")
        exit(1)
    return tokens


@time_function
def interpret(tokens: list[Token], mem_size: int) -> bool:
    mem = bytearray(mem_size)
    memmax = 0
    iptr = 0
    memptr = 0
    while -1 < iptr < len(tokens):
        val = tokens[iptr].value if tokens[iptr].value else 1
        match tokens[iptr].ttype:
            case TokenType.TapeRight:
                for _ in range(val):
                    memptr += 1
                    if memptr > memmax:
                        memmax = memptr

            case TokenType.TapeLeft:
                for _ in range(val):
                    memptr -= 1
            case TokenType.Inc:
                for _ in range(val):
                    if memptr >= 0 and memptr < mem_size:
                        # memory over/underflows without warning
                        mem[memptr] = (mem[memptr] + 1) % 256

                    else:
                        print(f"ERROR: memory index out of range at {memptr}")
                        exit(1)
            case TokenType.Dec:
                for _ in range(val):
                    if memptr >= 0 and memptr < mem_size:
                        # memory over/underflows without warning
                        mem[memptr] = (mem[memptr] - 1) % 256

                    else:
                        print(f"ERROR: memory index out of range at {memptr}")
                        exit(1)

            case TokenType.OpenBracket:
                if memptr >= 0 and memptr < mem_size:
                    # jump over loop if current mem is 0
                    if mem[memptr] == 0:
                        iptr = tokens[iptr].value

                else:
                    print(f"ERROR: memory index out of range at {memptr}")
                    exit(1)

            case TokenType.CloseBracket:
                if memptr >= 0 and memptr < mem_size:
                    # jump back to beginning of loop if mem is not 0
                    if mem[memptr] != 0:
                        iptr = tokens[iptr].value

                else:
                    print(f"ERROR: memory index out of range at {memptr}")
                    exit(1)

            case TokenType.GetChar:
                for _ in range(val):
                    char = sys.stdin.read(1)
                    # char = input()[0]
                    if memptr >= 0 and memptr < mem_size:
                        mem[memptr] = ord(char)
                    else:
                        print(f"ERROR: memory index out of range at {memptr}")
                        exit(1)

            case TokenType.PutChar:
                for _ in range(val):
                    if memptr >= 0 and memptr < mem_size:
                        print(chr(mem[memptr]), end="")
                    else:
                        print(f"ERROR: memory index out of range at {memptr}")
                        exit(1)

            case TokenType.Unknown:
                print(f'ERROR: unknown command at {iptr} "{tokens[iptr].value}"')
                exit(1)

        iptr += 1

    print("")
    # print(f"mem = {mem}")
    print(f"max memory size used: {memmax} bytes")


def main() -> None:
    if len(sys.argv) != 2:
        print("ERROR: no file given")
        print(f"Correct usage: {sys.argv[0]} <filename.b>")
        exit(1)

    filename = sys.argv[1]
    with open(filename, "r") as f:
        text = "".join(f.read().split())

    # TODO: having all these different functions might not be the best approach
    # tokens = tokenize(text)
    # print(f"tokens before contraction: {len(tokens)}")
    # tokens = contract_expressions(tokens)
    # print(f"tokens after contraction: {len(tokens)}")
    # tokens = match_brackets(tokens)
    # interpret(tokens, 1000)

    test_contraction_time(text, 1000)


def test_contraction_time(text: str, mem_size: int):
    print(f"without contraction")
    tokens = match_brackets(tokenize(text))
    interpret(tokens, mem_size)
    print(f"with contraction")
    tokens = match_brackets(contract_expressions(tokenize(text)))
    interpret(tokens, mem_size)


if __name__ == "__main__":
    main()


# TODO: Compile?
# TODO: cleaner error handling
# TODO: Infinite Tape?

# TODO: BF To python "transpiler"
# TODO: make interpreter agnostic to what happens after the match statement
# -> make it call to functions of an interface that is implemented in both interpreter and transpiler
