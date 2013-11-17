# Abstract syntax tree and pretty-printing code for the lambda
# calculus.

import abc


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


