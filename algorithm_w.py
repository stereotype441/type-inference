# An implementation of Hindley-Milner type inference in Python, as an
# exercise to improve my understanding of type inference algorithms.
#
# Based on http://en.wikipedia.org/wiki/Hindley%E2%80%93Milner.
#
# Tested with Python 2.7.

import abc
import disjoint_set
import unittest


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
            assert False
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
        assert False
        self.f = f
        self.x = x

    def _pretty(self, precedence):
        assert False
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
        assert False
        self.var = var
        self.e1 = e1
        self.e2 = e2

    def _pretty(self, precedence):
        assert False
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


class Monotype(object):
    def __init__(self, application = None):
        self.application = application

    def __repr__(self):
        if self.application is None:
            return 'Monotype()'
        else:
            return 'Monotype({0!r})'.format(self.application)


class Polytype(object):
    def __init__(self, bound_vars, monotype_var):
        self.bound_vars = bound_vars
        self.monotype_var = monotype_var


# TODO: detect infinite types ("occurs check")
class TypeInferrer(object):
    def __init__(self):
        # Each element of __type_sets is a type variable introduced by
        # type inference.  Each set in __type_sets is a set of type
        # variables which type inference has determined must be
        # equivalent.
        self.__type_sets = disjoint_set.DisjointSet()

        # For each set s of equivalent type variables in
        # self.__type_sets, self.__inferred_types[s] is the type that
        # has been inferred for that set.  Always a monotype.
        self.__inferred_types = {}

        # For each name that is currently in scope during the visiting
        # process, self.__env[name] is the polytype assigned to that
        # name.
        self.__env = {}

        # Create built-in types
        self.__bool_ty = self.new_type_application('Bool')

    def new_type_var(self):
        """Produce a new type variable with no meaning assigned to it
        yet.
        """
        # TODO: refactor the call to Monotype() into this function.
        return self.__type_sets.add_elem()

    def new_type_application(self, type_constructor, *args):
        """Produce a new type variable representing an instantiation
        of the given type constructor, with the given arguments.

        The type constructor should be a string; the arguments should
        be type variables.
        """
        type_var = self.new_type_var()
        self.__inferred_types[type_var] = Monotype(
            (type_constructor,) + tuple(args))
        return type_var

    def new_fn_type(self, x, y):
        """Produce a new type variable representing a function which
        takes type variable x to type variable y.
        """
        return self.new_type_application('->', x, y)

    def specialize(self, polytype):
        """Specialize a polytype into a monotype by replacing any
        bound variables with brand new type variables.
        """
        if len(polytype.bound_vars) == 0:
            return polytype.monotype_var
        else:
            assert False
            raise NotImplementedError() # TODO

    def canonicalize(self, type_var, types_seen):
        """Convert a type variable into a string representing the type
        in canonical form (for unit testing).

        Type variables are converted into integers, where 0 is the
        first type variable seen, 1 is the next type variable seen,
        and so on.  (types_seen is used to keep track of which types
        have been seen, and their mapping to integers).

        Type applications are converted into tuples, where the first
        element of the tuple is the type constant, and the remaining
        tuple elements are the type parameters, which have in turn
        been canonicalized.

        For example, calling canonicalize() on the type "a -> b -> a"
        (and passing in {} for types_seen) produces the result ('->',
        0, ('->', 1, 0)).
        """
        #print type_var, self.__inferred_types
        type_var = self.__type_sets.find(type_var)
        monotype = self.__inferred_types[type_var]
        if monotype.application is not None:
            result = [monotype.application[0]]
            for i in xrange(1, len(monotype.application)):
                result.append(self.canonicalize(monotype.application[i],
                                                types_seen))
            return tuple(result)
        else:
            if type_var not in types_seen:
                types_seen[type_var] = len(types_seen)
            return types_seen[type_var]

    def visit(self, expr):
        if isinstance(expr, Variable):
            # Look up the type bound to the variable name; if it was
            # introduced by a let expression we need to specialize it.
            result = self.specialize(self.__env[expr.name])
        elif isinstance(expr, LambdaAbstraction):
            # If the new declaration shadows a previous declaration
            # with the same variable name, save the type of the old
            # variable.
            old_var_type = self.__env.get(expr.var)

            # Generate a new monotype to represent the bound variable.
            type_var = self.new_type_var()
            monotype = Monotype()
            self.__inferred_types[type_var] = monotype

            # Generate a new polytype to store in the environment.
            # TODO: should polytype.bound_vars be a list or a tuple?
            self.__env[expr.var] = Polytype([], type_var)

            # Visit the subexpression and allow it to refine the type
            # of the bound variable.
            subexpr_type = self.visit(expr.expr)

            # Restore the old meaning of the bound variable.
            if old_var_type is None:
                del self.__env[expr.var]
            else:
                assert False
                self.__env[expr.var] = old_var_type

            # The inferred type of the resulting abstraction is
            # (var_type -> subexpr_type).
            result = self.new_fn_type(type_var, subexpr_type)
        elif isinstance(expr, BoolLiteral):
            result = self.__bool_ty
        else:
            assert False
        print 'Assigned {0} a type of {1}'.format(
            expr, self.canonicalize(result, {}))
        return result


class TestTypeInference(unittest.TestCase):
    def check_single_expr(self, expr, expected_type):
        ti = TypeInferrer()
        ty = ti.visit(expr)
        canonical_ty = ti.canonicalize(ty, {})
        self.assertEqual(canonical_ty, expected_type)

    def test_identity(self):
        self.check_single_expr(
            LambdaAbstraction('x', Variable('x')),
            ('->', 0, 0))

    def test_const(self):
        self.check_single_expr(
            LambdaAbstraction('x', LambdaAbstraction('y', Variable('x'))),
            ('->', 0, ('->', 1, 0)))

    def test_bool(self):
        self.check_single_expr(
            BoolLiteral(True),
            ('Bool',))


if __name__ == '__main__':
    unittest.main()
