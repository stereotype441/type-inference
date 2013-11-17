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


class Monotype(object):
    def __init__(self, application = None):
        # TODO: "application" isn't a helpful name.
        # TODO: don't use a simple tuple for self.application.
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

    def __repr__(self):
        return 'Polytype({0!r}, {1!r})'.format(self.bound_vars,
                                               self.monotype_var)


class TypeInferenceError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


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
            assignments = {}
            for v in polytype.bound_vars:
                v_set = self.__type_sets.find(v)
                assignments[v_set] = self.new_type_var()
                self.__inferred_types[assignments[v_set]] = Monotype()
            def specialize_part(type_variable):
                type_set = self.__type_sets.find(type_variable)
                if type_set in assignments:
                    return assignments[type_set]
                else:
                    monotype = self.__inferred_types[type_set]
                    if monotype.application is None:
                        assert False
                        return type_variable
                    else:
                        new_application = [monotype.application[0]]
                        for i in xrange(1, len(monotype.application)):
                            new_application.append(
                                specialize_part(monotype.application[i]))
                        new_application = tuple(new_application)
                        if (new_application == monotype.application):
                            assert False
                            # No changes were made, so re-use the old type.
                            return type_variable
                        else:
                            # Create a new type variable to represent
                            # the substituted monotype.
                            new_type = self.new_type_var()
                            self.__inferred_types[new_type] = Monotype(
                                new_application)
                            return new_type
            return specialize_part(polytype.monotype_var)

    def find_free_vars_in_type(self, type_var):
        free_vars = set()
        def recurse(type_var):
            type_set = self.__type_sets.find(type_var)
            monotype = self.__inferred_types[type_set]
            if monotype.application is None:
                free_vars.add(self.__type_sets.representative(type_set))
            else:
                for i in xrange(1, len(monotype.application)):
                    recurse(monotype.application[i])
        recurse(type_var)
        return frozenset(free_vars)

    def find_free_vars_in_env(self):
        free_vars = set()
        for polytype in self.__env.itervalues():
            forall_vars = set()
            for v in polytype.bound_vars:
                assert False
                type_set = self.__type_sets.find(type_var)
                forall_vars.add(self.__type_sets.representative(type_set))
            free_vars.update(
                self.find_free_vars_in_type(polytype.monotype_var) -
                forall_vars)
        return frozenset(free_vars)

    def generalize(self, type_var):
        """Generalize a type variable into a polytype, by quantifying
        all variables that are free in type_var but not free in env.
        """
        free_vars = (self.find_free_vars_in_type(type_var) -
                     self.find_free_vars_in_env())
        return Polytype(free_vars, type_var)

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

    def visit_with_binding(self, var, binding, expr):
        # If the new declaration shadows a previous declaration with
        # the same variable name, save the type of the old variable.
        old_binding = self.__env.get(var)

        self.__env[var] = binding
        result = self.visit(expr)

        # Restore the old meaning of the bound variable.
        if old_binding is None:
            del self.__env[var]
        else:
            assert False
            self.__env[var] = old_binding

        return result

    def visit(self, expr):
        if isinstance(expr, Variable):
            # Look up the type bound to the variable name; if it was
            # introduced by a let expression we need to specialize it.
            result = self.specialize(self.__env[expr.name])
            assert isinstance(result, int)
        elif isinstance(expr, LambdaAbstraction):
            # Generate a new monotype to represent the bound variable.
            type_var = self.new_type_var()
            monotype = Monotype()
            self.__inferred_types[type_var] = monotype

            # Generate a new polytype to store in the environment.
            polytype = Polytype(frozenset(), type_var)

            # Visit the subexpression and allow it to refine the type
            # of the bound variable.
            subexpr_type = self.visit_with_binding(expr.var, polytype,
                                                   expr.expr)

            # The inferred type of the resulting abstraction is
            # (var_type -> subexpr_type).
            result = self.new_fn_type(type_var, subexpr_type)
            assert isinstance(result, int)
        elif isinstance(expr, BoolLiteral):
            result = self.__bool_ty
            assert isinstance(result, int)
        elif isinstance(expr, Application):
            # First infer the types of the LHS ("f") and RHS ("x") of
            # the application.
            f_type = self.visit(expr.f)
            x_type = self.visit(expr.x)

            # Create a new type variable to represent the result of
            # the function application.
            result = self.new_type_var()
            monotype = Monotype()
            self.__inferred_types[result] = monotype

            # Figure out what the type of "f" must be given the type
            # of "x" and the result type.
            f_type_to_unify = self.new_fn_type(x_type, result)

            # Unify that with the type that we've already inferred for
            # f.
            self.unify(f_type, f_type_to_unify)
            assert isinstance(result, int)
        elif isinstance(expr, LetExpression):
            # Infer the type of the expression to be bound.
            e1_type = self.visit(expr.e1)

            # Generate a new monotype to represent the bound variable.
            polytype = self.generalize(e1_type)
            print polytype
            print self.__inferred_types

            # Visit the second expression and allow it to use the
            # bound variable.
            result = self.visit_with_binding(expr.var, polytype, expr.e2)
            assert isinstance(result, int)
        else:
            assert False
        print 'Assigned {0} a type of {1}'.format(
            expr, self.canonicalize(result, {}))
        return result

    def unify(self, type_x, type_y):
        print 'unify({0}, {1})'.format(type_x, type_y)
        self.check_invariants()
        # TODO: detect infinite types ("occurs check")
        set_x = self.__type_sets.find(type_x)
        set_y = self.__type_sets.find(type_y)
        monotype_x = self.__inferred_types[set_x]
        monotype_y = self.__inferred_types[set_y]
        if monotype_x.application is None or monotype_y.application is None:
            del self.__inferred_types[set_x]
            del self.__inferred_types[set_y]
            new_set = self.__type_sets.union(set_x, set_y)
            if monotype_x.application is not None:
                self.__inferred_types[new_set] = monotype_x
            else:
                # Covers the case where both x and y are free variables
                self.__inferred_types[new_set] = monotype_y
        else:
            # Neither x nor y is a free variable.
            if monotype_x.application[0] != monotype_y.application[0]:
                raise TypeInferenceError("Can't unify {0!r} with {1!r}".format(
                        monotype_x.application[0], monotype_y.application[0]))
            assert len(monotype_x.application) == len(monotype_y.application)
            for i in xrange(1, len(monotype_x.application)):
                self.unify(monotype_x.application[i],
                           monotype_y.application[i])

    def check_invariants(self):
        all_sets = set(self.__type_sets.get_all_sets())
        inferred_types_keys = set(self.__inferred_types.keys())
        if all_sets != inferred_types_keys:
            raise Exception(
                'all sets are {0!r}, but inferred_types has keys {1!r}'.format(
                    all_sets, inferred_types_keys))


class TestTypeInference(unittest.TestCase):
    def check_single_expr(self, expr, expected_type):
        ti = TypeInferrer()
        ty = ti.visit(expr)
        canonical_ty = ti.canonicalize(ty, {})
        self.assertEqual(canonical_ty, expected_type)

    def check_type_error(self, expr):
        ti = TypeInferrer()
        # Note: self.assertRaises doesn't give a useful stack trace if
        # there's an unexpected exception, so handroll the logic we
        # want.
        try:
            ty = ti.visit(expr)
        except TypeInferenceError:
            return
        self.fail('Expected a type inference error, got a type of {0!r}'.format(
                ti.canonicalize(ty, {})))

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

    def test_const_application(self):
        self.check_single_expr(
            Application(
                LambdaAbstraction('x', LambdaAbstraction('y', Variable('x'))),
                BoolLiteral(True)),
            ('->', 0, ('Bool',)))

    def test_bad_application(self):
        self.check_type_error(Application(BoolLiteral(True), BoolLiteral(False)))

    def test_simple_let(self):
        self.check_single_expr(
            LetExpression(
                'const',
                LambdaAbstraction('x', LambdaAbstraction('y', Variable('x'))),
                Application(Variable('const'), BoolLiteral(True))),
            ('->', 0, ('Bool',)))

    def test_let_with_bool(self):
        self.check_single_expr(
            LetExpression('t', BoolLiteral(True), Variable('t')),
            ('Bool',))

    def test_let_non_generalization(self):
        # When a variable bound with "let" is generalized, types that
        # are constrained by declarations in enclosing scopes should
        # not be generalized.  For example, in:
        #
        # (\f . let x = f True in f x)
        #
        # "let x = f True" constrains f to have type "Bool -> a",
        # giving "x" a type of "a".  Then, "in f x" constrains "x" to
        # have type Bool.  Therefore, "f"'s type becomes "Bool ->
        # Bool", and the final expression should have type "((Bool ->
        # Bool) -> Bool)".
        #
        # If, at the time of the let-binding, we had over-generalized
        # the type of "x" to "forall a. a", then when "x" was used in
        # "in f x", it would have been specialized to a new type
        # variable "b", which would have been constrained to have type
        # "Bool", but "a" would have remained general.  Therefore,
        # "f"'s type would have been "Bool -> a", and the final
        # expressino would have had type "((Bool -> a) -> a)", which
        # is incorrect.
        self.check_single_expr(
            LambdaAbstraction(
                'f',
                LetExpression(
                    'x',
                    Application(Variable('f'), BoolLiteral(True)),
                    Application(Variable('f'), Variable('x')))),
            ('->', ('->', ('Bool',), ('Bool',)), ('Bool',)))

    def test_let_generalization(self):
        # Check that variables bound with "let" can be used in a
        # general fashion.  For example, in:
        #
        # let id = \x . x in (id (\x . x)) (id True)
        #
        # The first usage of "id" has type "(Bool -> Bool) -> (Bool ->
        # Bool)", and the second usage has type "Bool -> Bool", giving
        # the final expression type "Bool".
        self.check_single_expr(
            LetExpression(
                'id',
                LambdaAbstraction('x', Variable('x')),
                Application(
                    Application(
                        Variable('id'),
                        LambdaAbstraction('x', Variable('x'))),
                    Application(Variable('id'), BoolLiteral(True)))),
            ('Bool',))

    # TODO: test that variables bound with lambda can't be used in a
    # general fashion.


if __name__ == '__main__':
    unittest.main()
