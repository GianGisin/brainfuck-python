from enum import Enum, auto
from typing import Optional, Any

import sys


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


def interpret(tokens: list[Token], mem_size: int) -> bool:
    mem = bytearray(mem_size)
    memmax = 0
    iptr = 0
    memptr = 0
    while -1 < iptr < len(tokens):
        match tokens[iptr].ttype:
            case TokenType.TapeRight:
                memptr += 1
                if memptr > memmax:
                    memmax = memptr
            case TokenType.TapeLeft:
                memptr -= 1
            case TokenType.Inc:
                if memptr >= 0 and memptr < mem_size:
                    # memory over/underflows without warning
                    mem[memptr] = (mem[memptr] + 1) % 256

                else:
                    print(f"ERROR: memory index out of range at {memptr}")
                    exit(1)
            case TokenType.Dec:
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
                char = sys.stdin.read(1)
                # char = input()[0]
                if memptr >= 0 and memptr < mem_size:
                    mem[memptr] = ord(char)
                else:
                    print(f"ERROR: memory index out of range at {memptr}")
                    exit(1)

            case TokenType.PutChar:
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

    tokens = match_brackets(tokenize(text))
    interpret(tokens, 1000)


if __name__ == "__main__":
    main()


# TODO: contract multiple equal statements (that are not brackets)
# e.g.: ++++++++ => Token(TokenType.Inc, 8)
# this might require lookahead features
# TODO: Compile?
# TODO: cleaner error handling
