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
    def ob(self, to) -> int:
        pass

    @abstractmethod
    def cb(self, to) -> int:
        pass

    @abstractmethod
    def gc(self, n):
        pass

    @abstractmethod
    def pc(self, n):
        pass


class Transpiler(CommandListener):
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.code = ""
        self.indent = 1

    def _indent(self, s: str) -> str:
        return "    " * self.indent + s

    def _code_append(self, *args: str):
        for a in args:
            self.code += self._indent(a) + "\n"

    def get_code(self) -> str:
        pre = f"def main():\n    mem = bytearray({self.mem_size})\n    mem_ptr = 0\n\n"
        post = "    print(" ")\n\nif __name__ == '__main__':\n    main()\n"
        return pre + self.code + post

    def tr(self, n):
        self._code_append(f"mem_ptr += {n}")

    def tl(self, n):
        self._code_append(f"mem_ptr -= {n}")

    def inc(self, n):
        self._code_append(f"mem[mem_ptr] += {n}")

    def dec(self, n):
        self._code_append(f"mem[mem_ptr] -= {n}")

    def ob(self, to) -> int:
        self._code_append("")
        self._code_append("while mem[mem_ptr] != 0:")
        self.indent += 1

    def cb(self, to) -> int:
        self.indent -= 1
        self._code_append("")

    def gc(self, n):
        self._code_append("mem[mem_ptr] = ord(input()[0])")

    def pc(self, n):
        self._code_append("print(chr(mem[mem_ptr]), end='')")


class Interpreter(CommandListener):
    def __init__(self, mem_size):

        self.mem = bytearray(mem_size)
        self.memmax = 0
        self.memptr = 0
        self.mem_size = mem_size

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

    def ob(self, to) -> int:
        if self.memptr >= 0 and self.memptr < self.mem_size:
            # jump back to beginning of loop if mem is not 0
            if self.mem[self.memptr] == 0:
                return to

        else:
            print(f"ERROR: memory index out of range at {self.memptr}")
            exit(1)

    def cb(self, to) -> int:
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
def transpile(tokens: list[Token], mem_size: int):
    listener = Transpiler(mem_size)
    dispatch_commands(tokens, listener)
    return listener.get_code()


@time_function
def interpret(tokens: list[Token], mem_size: int):
    listener = Interpreter(mem_size)
    dispatch_commands(tokens, listener)


def dispatch_commands(tokens: list[Token], listener: CommandListener) -> bool:
    iptr = 0
    while -1 < iptr < len(tokens):
        val = tokens[iptr].value if tokens[iptr].value else 1
        match tokens[iptr].ttype:
            case TokenType.TapeRight:
                listener.tr(val)

            case TokenType.TapeLeft:
                listener.tl(val)

            case TokenType.Inc:
                listener.inc(val)

            case TokenType.Dec:
                listener.dec(val)

            case TokenType.OpenBracket:
                ret = listener.ob(tokens[iptr].value)
                if ret:
                    iptr = ret

            case TokenType.CloseBracket:
                ret = listener.cb(tokens[iptr].value)
                if ret:
                    iptr = ret

            case TokenType.GetChar:
                listener.gc(val)

            case TokenType.PutChar:
                listener.pc(val)

            case TokenType.Unknown:
                print(f'ERROR: unknown command at {iptr} "{tokens[iptr].value}"')
                exit(1)

        iptr += 1

    print("")


def main() -> None:
    if len(sys.argv) < 2:
        print("ERROR: no file given")
        print(f"Correct usage: {sys.argv[0]} <filename.b>")
        exit(1)

    filename = sys.argv[1]
    with open(filename, "r") as f:
        text = "".join(f.read().split())

    tokens = match_brackets(contract_expressions(tokenize(text)))
    # interpret(tokens, 1000)
    transpile(tokens, 1000)


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
