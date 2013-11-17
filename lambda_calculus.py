# Abstract syntax tree and pretty-printing code for the lambda
# calculus.

import abc
import re


class LambdaExpr(object):
    """Data structure representing the input language which we will be
    type checking.

    We implement an extension of the lambda calculus which includes
    the let syntax, as follows:

    Construct           Example
    variable            x
    application         (f x)
    lambda abstraction  (\\x . e)
    let expression      (let x = e1 in e2)

    (As in Haskell, we use "\\" to represent lambda.)

    Note that in the untyped lambda calculus, let expressions are
    unnecessary, since (let x = e1 in e2) may be regarded as syntactic
    sugar for ((\\x . e2) e1).  However, for type inference, we make
    the following distinction between these two syntaxes: (let x = e1
    in e2) defines x to be polymorphic, so for example this is
    allowed:

    (let id = (\\x . x) in (addFloat (intToFloat (id 1)) (id 1.0)))

    since "id" can take on the type (int -> int) at one call site, and
    (float -> float) at the other.  However, the corresponding lambda
    representation would be illegal:

    (\\id . (addFloat (intToFloat (id 1)) (id 1.0))) (\\x . x)

    Since this would require a single monomorphic type to be assigned
    to id.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _pretty(self, precedence):
        """Pretty print this lambda expression to a string.
        Precedence determines whether parentheses are inserted--it is
        interpreted as follows:

        0: inside lambda abstraction or right side of let expression.
           Nothing is parenthesized.

        1: left side of let expression.  Let expressions are
           parenthesized.

        2: left side of application.  Lambda abstractions and let
           expressions are parenthesized.

        3: right side of application.  Applications, lambda
           abstractions, and let expressions are parenthesized.
        """
        raise NotImplementedError()

    @staticmethod
    def _paren_if(parens_needed, s):
        if parens_needed:
            return '({0})'.format(s)
        else:
            return s

    def __str__(self):
        return self._pretty(0)


class Variable(LambdaExpr):
    """Lambda calculus representation of a free or bound variable."""
    def __init__(self, name):
        self.name = name

    def _pretty(self, precedence):
        return self.name


class Application(LambdaExpr):
    """Lambda calculus representation of a function application
    (a.k.a. "function call").
    """
    def __init__(self, f, x):
        self.f = f
        self.x = x

    def _pretty(self, precedence):
        return self._paren_if(
            precedence >= 3,
            '{0} {1}'.format(self.f._pretty(2), self.x._pretty(3)))


class LambdaAbstraction(LambdaExpr):
    """Lambda calculus representation of a lambda abstraction."""
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr

    def _pretty(self, precedence):
        return self._paren_if(
            precedence >= 2,
            '\\{0} . {1}'.format(self.var, self.expr._pretty(0)))


class LetExpression(LambdaExpr):
    """Lambda calculus representation of a let expression."""
    def __init__(self, var, e1, e2):
        self.var = var
        self.e1 = e1
        self.e2 = e2

    def _pretty(self, precedence):
        return self._paren_if(
            precedence >= 1,
            'let {0} = {1} in {2}'.format(self.var,
                                          self.e1._pretty(1),
                                          self.e2._pretty(0)))


class BoolLiteral(LambdaExpr):
    """Lambda calculus representation of a boolean literal."""
    def __init__(self, value):
        assert isinstance(value, bool)
        self.value = value

    def _pretty(self, precedence):
        return '{0}'.format(self.value)


class ParseError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


LEXING_REGEXP = re.compile(r'[A-Za-z_][A-Za-z_0-9]*|[()=\\.]')
NON_WHITESPACE_REGEXP = re.compile(r'\S')
RESERVED_WORDS = ('let', 'in', '(', ')', '=', '\\', '.')


def parse(s):
    """Quick and dirty recursive descent parser to transform a string
    into a lambda calculus AST.  The grammar is as follows:

    expression: '(' expression ')' |
                '\\' identifier '.' expression |
                'let' identifier '=' expression 'in' expression |
                expression expression |
                identifier

    Ambiguities are resolved as follows:

    - Application associates to the left.  So "a b c" is parsed as "(a
      b) c".

    - The expressions appearing at the end of a lambda abstraction or
      a "let" expression are considered to extend as far to the right
      as possible.  So "\\x . y z" is parsed as "\\x . (y z)", not
      "(\\x . y) z".  Similarly, "let x = y in z w" is parsed as "let
      x = y in (z w)".
    """
    # Lex the input string into tokens.
    tokens = []
    pos = 0
    while pos < len(s):
        match = NON_WHITESPACE_REGEXP.search(s, pos)
        if match is None:
            break
        pos = match.start()
        match = LEXING_REGEXP.match(s, pos)
        if match is None:
            raise ParseError('Unexpected input character {0!r}'.format(s[pos]))
        tokens.append(match.group())
        pos = match.end()

    def peek():
        if len(tokens) == 0:
            return None
        else:
            return tokens[0]

    def advance():
        return tokens.pop(0)

    def expect(s):
        if peek() != s:
            raise ParseError('Expected {0!r}, got {1!r}'.format(s, peek()))
        return advance()

    def expect_identifier():
        token = advance()
        if token in RESERVED_WORDS:
            raise ParseError('Expected identifier, got {0!r}'.format(token))
        return token

    def consume_expression(precedence):
        """Consume an expression in the lambda grammar.  Precedence is
        as follows:

        0: consume as many tokens as possible.
        1: do not consume the RHS of an application.
        """
        if peek() == '(':
            advance()
            e1 = consume_expression(0)
            expect(')')
        elif peek() == '\\':
            advance()
            var = expect_identifier()
            expect('.')
            subexpr = consume_expression(0)
            return LambdaAbstraction(var, subexpr)
        elif peek() == 'let':
            advance()
            var = expect_identifier()
            expect('=')
            e1 = consume_expression(0)
            expect('in')
            e2 = consume_expression(0)
            return LetExpression(var, e1, e2)
        else:
            var = expect_identifier()
            e1 = Variable(var)
        if precedence < 1:
            while peek() not in (')', 'in', None):
                e2 = consume_expression(1)
                e1 = Application(e1, e2)
        return e1

    result = consume_expression(0)
    if peek() is not None:
        raise ParseError('Expected END, got {0!r}'.format(peek()))
    return result
