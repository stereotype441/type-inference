import disjoint_set
from lambda_calculus import (
    Variable, LetExpression, LambdaAbstraction, Application, parse)
import unittest


class Monotype(object):
    """Base class representing a "monotype" (a type in which all free
    variables are subject to type inference, and may in the future be
    unified with other type variables).
    """
    pass


class MonotypeVar(Monotype):
    """Monotype representing a single type variable."""
    def __repr__(self):
        return 'MonotypeVar()'


class MonotypeApp(Monotype):
    """Monotype representing a type constructor applied to zero or
    more type arguments.

    The type arguments are represented by "elements" of a DisjointSet
    object.
    """
    def __init__(self, constructor, *args):
        for arg in args:
            assert isinstance(arg, disjoint_set.DisjointSet.element_type)
        self.constructor = constructor
        self.args = args

    def __repr__(self):
        return 'MonotypeApp({0})'.format(
            ', '.join(repr(x) for x in (self.constructor,) + self.args))


class Polytype(object):
    """Representation of a polymorphic type, which should be
    specialized each time it is used by replacing the "forall"
    variables with new type variables.
    """
    def __init__(self, forall_vars, monotype_var):
        self.forall_vars = forall_vars
        self.monotype_var = monotype_var

    def __repr__(self):
        return 'Polytype({0!r}, {1!r})'.format(self.forall_vars,
                                               self.monotype_var)


class TypeInferenceError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class TypeInferrer(object):
    """Data structures and methods necessary to do type inference.

    After constructing an object of this class, call visit() to
    perform type inference on a lambda calculus expression.  The
    inferred type is returned.

    The data structure of the inferred type (Monotype) refers to
    private data stored inside TypeInferrer.  To convert it to a
    neutral representation, pass it to canonicalize().
    """

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

        # Pre-load the environment with some built-in symbols
        a = self.new_type_var(MonotypeVar())
        b = self.new_type_var(MonotypeVar())
        self.__env['mk_pair'] = self.generalize(
            self.new_fn_type(
                a,
                self.new_fn_type(b, self.new_type_application('Pair', a, b))))
        self.__env['fst'] = self.generalize(
            self.new_fn_type(self.new_type_application('Pair', a, b), a))
        self.__env['snd'] = self.generalize(
            self.new_fn_type(self.new_type_application('Pair', a, b), b))
        self.__env['True'] = self.generalize(
            self.new_type_application('Bool'))
        self.__env['False'] = self.generalize(
            self.new_type_application('Bool'))
        self.__env['if'] = self.generalize(
            self.new_fn_type(
                self.new_type_application('Bool'),
                self.new_fn_type(a, self.new_fn_type(a, a))))
        self.__env['maybe'] = self.generalize(
            self.new_fn_type(
                b,
                self.new_fn_type(
                    self.new_fn_type(a, b),
                    self.new_fn_type(
                        self.new_type_application('Maybe', a),
                        b))))
        self.__env['Nothing'] = self.generalize(
            self.new_type_application('Maybe', a))
        self.__env['Just'] = self.generalize(
            self.new_fn_type(a, self.new_type_application('Maybe', a)))

    def new_type_var(self, monotype):
        """Produce a new type variable whose meaning is the given
        monotype.
        """
        type_var = self.__type_sets.add_elem()
        type_set = self.__type_sets.find(type_var)
        self.__inferred_types[type_set] = monotype
        return type_var

    def new_type_application(self, type_constructor, *args):
        """Produce a new type variable representing an instantiation
        of the given type constructor, with the given arguments.

        The type constructor should be a string; the arguments should
        be type variables.
        """
        return self.new_type_var(MonotypeApp(type_constructor,*args))

    def new_fn_type(self, x, y):
        """Produce a new type variable representing a function which
        takes type variable x to type variable y.
        """
        return self.new_type_application('->', x, y)

    def specialize(self, polytype):
        """Specialize a polytype into a monotype by replacing any
        bound variables with brand new type variables.
        """
        if len(polytype.forall_vars) == 0:
            # No specialization needed.
            return polytype.monotype_var
        else:
            # Introduce a brand new type variable for each variable in
            # the polytype that is qualified with "forall", and keep
            # track of the mapping from old variables to new
            # variables.
            #
            # Note: since the type in polytype.monotype_var arose out
            # of a unification process, it may contain distinct type
            # variables that have since been unified.  We need to
            # treat any such variables as equivalent; so we actually
            # keep track of the mapping from the old variables'
            # equivalence sets to the new variables.
            assignments = {}
            for v in polytype.forall_vars:
                v_set = self.__type_sets.find(v)
                assignments[v_set] = self.new_type_var(MonotypeVar())

            # Recursively substitute the new type variables for the
            # old ones.
            def specialize_part(type_variable):
                type_set = self.__type_sets.find(type_variable)
                if type_set in assignments:
                    return assignments[type_set]
                else:
                    monotype = self.__inferred_types[type_set]
                    if isinstance(monotype, MonotypeVar):
                        return type_variable
                    else:
                        new_args = tuple(specialize_part(arg)
                                         for arg in monotype.args)
                        if (new_args == monotype.args):
                            # No changes were made, so re-use the old type.
                            return type_variable
                        else:
                            # Create a new type variable to represent
                            # the substituted monotype.
                            return self.new_type_var(MonotypeApp(
                                    monotype.constructor, *new_args))
            return specialize_part(polytype.monotype_var)

    def find_free_vars_in_type(self, type_var):
        """Find the set of free variables contained within a type.

        Since the types participating in type inference are monotypes,
        they don't contain "forall" qualifications, so all variables
        appearing recursively within the type are free variables.

        To account for unifications that have already been done, this
        function actually returns a representative of each free type
        set (computed by DisjointSet.representative()).
        """
        free_vars = set()
        def recurse(type_var):
            type_set = self.__type_sets.find(type_var)
            monotype = self.__inferred_types[type_set]
            if isinstance(monotype, MonotypeVar):
                free_vars.add(self.__type_sets.representative(type_set))
            else:
                for arg in monotype.args:
                    recurse(arg)
        recurse(type_var)
        return frozenset(free_vars)

    def find_free_vars_in_env(self):
        """Find the set of free variables contained in the
        environment.

        Since the types in the environment are polytypes, we have to
        be careful not to count "forall" qualified type variables as
        free.

        To account for unifications that have already been done, this
        function actually returns a representative of each free type
        set (computed by DisjointSet.representative()).
        """
        free_vars = set()
        for polytype in self.__env.itervalues():
            forall_vars = set()
            for v in polytype.forall_vars:
                type_set = self.__type_sets.find(v)
                forall_vars.add(self.__type_sets.representative(type_set))
            free_vars.update(
                self.find_free_vars_in_type(polytype.monotype_var) -
                forall_vars)
        return frozenset(free_vars)

    def generalize(self, type_var):
        """Generalize a type variable into a polytype, by "forall"
        qualifying all variables that are free in type_var but not
        free in env.
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
        type_var = self.__type_sets.find(type_var)
        monotype = self.__inferred_types[type_var]
        if isinstance(monotype, MonotypeApp):
            result = [monotype.constructor]
            for arg in monotype.args:
                result.append(self.canonicalize(arg, types_seen))
            return tuple(result)
        else:
            if type_var not in types_seen:
                types_seen[type_var] = len(types_seen)
            return types_seen[type_var]

    def visit_with_binding(self, var, binding, expr):
        """Temporarily bind var to binding in the environment, and
        visit expr.
        """
        # If the new declaration shadows a previous declaration with
        # the same variable name, save the type of the old variable.
        old_binding = self.__env.get(var)

        self.__env[var] = binding
        result = self.visit(expr)

        # Restore the old meaning of the bound variable.
        if old_binding is None:
            del self.__env[var]
        else:
            self.__env[var] = old_binding

        return result

    def visit(self, expr):
        """Visit the given expression and infer a type for it."""
        if isinstance(expr, Variable):
            # Look up the type bound to the variable name; if it was
            # introduced by a let expression we need to specialize it.
            result = self.specialize(self.__env[expr.name])
            assert isinstance(result, disjoint_set.DisjointSet.element_type)
        elif isinstance(expr, LambdaAbstraction):
            # Generate a new monotype to represent the bound variable.
            type_var = self.new_type_var(MonotypeVar())

            # Generate a new polytype to store in the environment.
            polytype = Polytype(frozenset(), type_var)

            # Visit the subexpression and allow it to refine the type
            # of the bound variable.
            subexpr_type = self.visit_with_binding(expr.var, polytype,
                                                   expr.expr)

            # The inferred type of the resulting abstraction is
            # (var_type -> subexpr_type).
            result = self.new_fn_type(type_var, subexpr_type)
            assert isinstance(result, disjoint_set.DisjointSet.element_type)
        elif isinstance(expr, Application):
            # First infer the types of the LHS ("f") and RHS ("x") of
            # the application.
            f_type = self.visit(expr.f)
            x_type = self.visit(expr.x)

            # Create a new type variable to represent the result of
            # the function application.
            result = self.new_type_var(MonotypeVar())

            # Figure out what the type of "f" must be given the type
            # of "x" and the result type.
            f_type_to_unify = self.new_fn_type(x_type, result)

            # Unify that with the type that we've already inferred for
            # f.
            self.unify(f_type, f_type_to_unify)
            assert isinstance(result, disjoint_set.DisjointSet.element_type)
        elif isinstance(expr, LetExpression):
            # Infer the type of the expression to be bound.
            e1_type = self.visit(expr.e1)

            # Generate a new monotype to represent the bound variable.
            polytype = self.generalize(e1_type)

            # Visit the second expression and allow it to use the
            # bound variable.
            result = self.visit_with_binding(expr.var, polytype, expr.e2)
            assert isinstance(result, disjoint_set.DisjointSet.element_type)
        else:
            assert False # Unrecognized lambda expression.
        return result

    def occurs_in(self, type_set, monotype_app):
        """Determine whether a type variable in the set "type_set"
        appears anywhere in "monotype_app".  This is used to avoid
        creating infinite types.
        """
        for arg in monotype_app.args:
            arg_set = self.__type_sets.find(arg)
            if arg_set == type_set:
                return True
            arg_monotype = self.__inferred_types[arg_set]
            if isinstance(arg_monotype, MonotypeApp) and \
                    self.occurs_in(type_set, arg_monotype):
                return True
        return False

    def unify(self, type_x, type_y):
        """Unify the types represented by type_x and type_y."""
        set_x = self.__type_sets.find(type_x)
        set_y = self.__type_sets.find(type_y)
        if set_x == set_y:
            # Already unified
            return
        monotype_x = self.__inferred_types[set_x]
        monotype_y = self.__inferred_types[set_y]
        if isinstance(monotype_x, MonotypeVar) or \
                isinstance(monotype_y, MonotypeVar):
            # Make sure we don't create infinite types.
            if isinstance(monotype_x, MonotypeApp) and \
                    self.occurs_in(set_y, monotype_x):
                raise TypeInferenceError(
                    "Unifying {0!r} would create infinite type".format(
                        monotype_x))
            if isinstance(monotype_y, MonotypeApp) and \
                    self.occurs_in(set_x, monotype_y):
                raise TypeInferenceError(
                    "Unifying {0!r} would create infinite type".format(
                        monotype_y))
        else:
            # Both x and y represent MonotypeApps, so we need to match
            # the type constructors and then unify each of the type
            # arguments.
            if monotype_x.constructor != monotype_y.constructor:
                raise TypeInferenceError("Can't unify {0!r} with {1!r}".format(
                        monotype_x, monotype_y))
            assert len(monotype_x.args) == len(monotype_y.args)
            for i in xrange(0, len(monotype_x.args)):
                self.unify(monotype_x.args[i], monotype_y.args[i])
            # Now that we've unified the contents of the MonotypeApps,
            # continue on to union set_x and set_y.  There's no harm,
            # and it may help shortcut future unify() operations.

        # Unioning set_x and set_y invalidates them and produces a new
        # object to represent the set.  So remove the set_x and set_y
        # keys from __inferred_types, and add the new inferred type
        # after unioning.
        del self.__inferred_types[set_x]
        del self.__inferred_types[set_y]
        new_set = self.__type_sets.union(set_x, set_y)
        # If either x or y maps to a MonotypeApp, then that
        # MonotypeApp needs to be stored in self.__inferred_types.  If
        # both of them map to a MonotypeApp, that's fine; thanks to
        # the recursive calls to unify() above, they are the same, so
        # we can just pick one.  If they both map to a MonotypeVar, it
        # doesn't matter which one we pick because MonotypeVar doesn't
        # store any additional information.
        if isinstance(monotype_x, MonotypeApp):
            self.__inferred_types[new_set] = monotype_x
        else:
            self.__inferred_types[new_set] = monotype_y

    def check_invariants(self):
        """For debugging and unit testing: check data structure
        invariants.
        """
        # Check that each set in self.__type_sets corresponds to a key
        # in self.__inferred_types.
        all_sets = set(self.__type_sets.get_all_sets())
        inferred_types_keys = set(self.__inferred_types.keys())
        if all_sets != inferred_types_keys:
            raise Exception(
                'all sets are {0!r}, but inferred_types has keys {1!r}'.format(
                    all_sets, inferred_types_keys))

    def get_builtin_names(self):
        """Find out the names of the built-in symbols."""
        return self.__env.keys()


def pretty_print_canonical_type(ty, precedence = 0):
    """Pretty print a return value from TypeInferrer.canonicalize().

    Precedence is as follows:
    0: only parenthesize pairs.
    1: parenthesize functions and pairs.
    2: parenthesize functions, pairs, and type applications.
    """
    if isinstance(ty, int):
        parens_needed = False
        result = 't{0}'.format(ty)
    elif ty[0] == '->':
        parens_needed = precedence >= 1
        result = '{0} -> {1}'.format(
            pretty_print_canonical_type(ty[1], 1),
            pretty_print_canonical_type(ty[2], 0))
    elif ty[0] == 'Pair':
        parens_needed = True
        result = '{0}, {1}'.format(
            pretty_print_canonical_type(ty[1], 0),
            pretty_print_canonical_type(ty[2], 0))
    else:
        terms = [ty[0]]
        for i in xrange(1, len(ty)):
            terms.append(pretty_print_canonical_type(ty[i], 2))
        result = ' '.join(terms)
        parens_needed = len(terms) > 1 and precedence >= 2
    if parens_needed:
        return '({0})'.format(result)
    else:
        return result


class TestTypeInference(unittest.TestCase):
    def check_single_expr(self, expr, expected_type):
        ti = TypeInferrer()
        ty = ti.visit(expr)
        canonical_ty = ti.canonicalize(ty, {})
        ti.check_invariants()
        self.assertEqual(canonical_ty, expected_type)

    def check_type_error(self, expr):
        ti = TypeInferrer()
        # Note: self.assertRaises doesn't give a useful stack trace if
        # there's an unexpected exception, so handroll the logic we
        # want.
        try:
            ty = ti.visit(expr)
        except TypeInferenceError:
            ti.check_invariants()
            return
        self.fail('Expected a type inference error, got a type of {0!r}'.format(
                ti.canonicalize(ty, {})))

    def test_identity(self):
        self.check_single_expr(
            parse(r'\x . x'),
            ('->', 0, 0))

    def test_const(self):
        self.check_single_expr(
            parse(r'\x . \y . x'),
            ('->', 0, ('->', 1, 0)))

    def test_bool(self):
        self.check_single_expr(
            parse('True'),
            ('Bool',))

    def test_const_application(self):
        self.check_single_expr(
            parse(r'(\x . \y . x) True'),
            ('->', 0, ('Bool',)))

    def test_bad_application(self):
        self.check_type_error(
            parse('True False'))

    def test_simple_let(self):
        self.check_single_expr(
            parse(r'let const = \x . \y . x in const True'),
            ('->', 0, ('Bool',)))

    def test_let_with_bool(self):
        self.check_single_expr(
            parse('let t = True in t'),
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
            parse(r'\f . let x = f True in f x'),
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
            parse(r'let id = \x . x in (id (\x . x)) (id True)'),
            ('Bool',))

    def test_lambda_non_generalization(self):
        # In contrast to variables bound by "let" expressions,
        # variables bound by lambda abstractions are not general.  For
        # example, this should fail to type check:
        #
        # (\f . (f (\x . x)) (f True))
        #
        # Because the first usage of f must have type "(a -> a) -> b",
        # whereas the second usage must have type "Bool -> c", and
        # these can't be unified.
        self.check_type_error(
            parse(r'\f . (f (\x . x)) (f True)'))

    def test_let_partial_generalization_general_part(self):
        # In some situations, "let" may generalize some of its type
        # variables but not others.  For example, in:
        #
        # (\x . let const_x = (\y . x) in const_x (\z . const_x True))
        #
        # if the type of "x" is "a", then the type assigned to
        # "const_x" by the "let" expression is "forall b . b -> a".
        # Then, the first usage of "const_x" is specialized to type (c
        # -> a) -> a, whereas the second is specialized to "Bool ->
        # a", giving the final expression type "a -> a".  This would
        # not work if the type assigned to "const_x" did not include
        # "forall b".
        self.check_single_expr(
            parse(r'\x . let const_x = \y . x in const_x (\z . const_x True)'),
            ('->', 0, 0))

    def test_let_partial_generalization_non_general_part(self):
        # In some situations, "let" may generalize some of its type
        # variables but not others.  For example, in:
        #
        # (\x . let const_x = (\y . x) in const_x ((const_x (\z . z)) True))
        #
        # if the type of "x" is initially "a", then the type assigned
        # to "const_x" by the "let" expression will initially be
        # "forall b . b -> a".  Then, the first usage of "const_x"
        # will be specialized to "c -> a", and the second will be
        # specialized to "d -> (Bool -> e)".  Since there is no
        # "forall a" in the type assigned to "const_x", "a" will be
        # unified with "Bool -> e", so the resulting expression will
        # have type "(Bool -> e) -> (Bool -> e)".  If a "forall a" had
        # been present, then unification would not have occurred, and
        # the type would have been "a -> a".
        self.check_single_expr(
            parse(r'\x . let const_x = \y . x in const_x ((const_x (\z . z)) True)'),
            ('->', ('->', ('Bool',), 0), ('->', ('Bool',), 0)))

    def test_partial_specialization(self):
        # When a type specialization does not affect the entire type,
        # we try to short-cut it to avoid creating extraneous type
        # variables.  For example, in:
        #
        # (\f . let g = (\x . \y . f y) in g True)
        #
        # type-checking of (\x . \y . f y) causes f's type to be
        # refined to "a -> b", and "g" is assigned a type of "forall c
        # . c -> a -> b".  When "g" is applied to "True", the "a -> b"
        # portion of the type doesn't need to be specialized.  The
        # final type of the whole expression should be "(a -> b) -> (a
        # -> b)".
        self.check_single_expr(
            parse(r'\f . let g = \x . \y . f y in g True'),
            ('->', ('->', 0, 1), ('->', 0, 1)))

    def test_nested_lets(self):
        # When a let-expression appears inside a second
        # let-expression, type variables appearing in the outer "let"
        # are not re-generalized.  For example, in:
        #
        # (\x . let f = (\y . x) in let g = f True in g)
        #
        # if "x" has type "a", then "f" is assigned a type of "forall
        # b . b -> a", and "g" is assigned a type of "a", *not*
        # "forall a . a".  Therefore the whole expression has type (a
        # -> a).
        self.check_single_expr(
            parse(r'\x . let f = \y . x in let g = f True in g'),
            ('->', 0, 0))

    def test_lambda_shadowing(self):
        # When a lambda expression redefines a variable bound in an
        # outer expression, the outer definition needs to be restored
        # once the lambda expression is exited.  For example, in:
        #
        # (\x . (\x . x True) (\y . x))
        #
        # The "x" appearing inside "(\x . x True)" has type "Bool ->
        # a", whereas the "x" appearing inside "(\y . x)" has type
        # "a", giving the entire expression type "a -> a".
        self.check_single_expr(
            parse(r'\x . (\x . x True) (\y . x)'),
            ('->', 0, 0))

    def test_let_shadowing(self):
        # When a let expression redefines a variable bound in an outer
        # expression, the outer definition needs to be restored once
        # the let expression is exited.  For example, in:
        #
        # (\x . (let x = \y . y in x) x)
        #
        # The "x" appearing inside "let x = \y . y in x" has type
        # "forall b . b -> b", whereas the "x" appearing at the end
        # has type "a", giving the entire expression type "a -> a".
        self.check_single_expr(
            parse(r'\x . (let x = \y . y in x) x'),
            ('->', 0, 0))

    def test_infinite_type(self):
        # Applying a function to itself produces an infinite type.
        # For example, in the expression:
        #
        # (\f . f f)
        #
        # if "f" has type "a", then in order to apply "f" to itself,
        # "a" must be unified with "a -> b".  It's impossible to do
        # this without creating an infinitely recursive type, which we
        # don't permit.
        self.check_type_error(
            parse(r'\f . f f'))

    def def_utils(self, subexpr):
        # Some useful utility functions for testing complex
        # unifications are:
        #
        # let ignore = \x . True
        #
        # "ignore" has type "a -> Bool"; it ignores its argument and
        # returns a boolean.
        #
        # let ignore2 = \x . \y . True
        #
        # "ignore2" has type "a -> b -> Bool"; it ignores two
        # arguments and returns a boolean.
        #
        # let second = \x . \y . y
        #
        # "second" has type "a -> b -> b"; it ignores its first
        # argument and returns its second.
        #
        # let unify = \x . \y . second (\z . ignore2 (z x) (z y)) x
        #
        # "unify" has type "a -> a -> a"; it forces its two arguments
        # to have the same type, and returns the first argument.
        #
        # let mk_fn = \x . \y . \z . second (unify x z) y
        #
        # "mk_fn" has type "a -> b -> (a -> b)"; given arguments of
        # types "a" and "b", it returns a function of type "a -> b".
        #
        # This function wraps the given subexpressions in the
        # necessary "let" constructs so that it can refer to "ignore",
        # "ignore2", and "unify".
        fns = [
            ('ignore', parse(r'\x . True')),
            ('ignore2', parse(r'\x . \y . True')),
            ('second', parse(r'\x . \y . y')),
            ('unify', parse(r'\x . \y . second (\z . ignore2 (z x) (z y)) x')),
            ('mk_fn', parse(r'\x . \y . \z . second (unify x z) y'))
            ]
        for name, defn in reversed(fns):
            subexpr = LetExpression(name, defn, subexpr)
        return subexpr

    def test_ignore_func(self):
        # Check the type of the "ignore" function defined in
        # def_utils().
        self.check_single_expr(
            self.def_utils(parse('ignore')),
            ('->', 0, ('Bool',)))

    def test_ignore2_func(self):
        # Check the type of the "ignore2" function defined in
        # def_utils().
        self.check_single_expr(
            self.def_utils(parse('ignore2')),
            ('->', 0, ('->', 1, ('Bool',))))

    def test_second_func(self):
        # Check the type of the "second" function defined in
        # def_utils().
        self.check_single_expr(
            self.def_utils(parse('second')),
            ('->', 0, ('->', 1, 1)))

    def test_unify_func(self):
        # Check the type of the "unify" function defined in
        # def_utils().
        self.check_single_expr(
            self.def_utils(parse('unify')),
            ('->', 0, ('->', 0, 0)))

    def test_mk_fn_func(self):
        # Check the type of the "mk_fn" function defined in
        # def_utils().
        self.check_single_expr(
            self.def_utils(parse('mk_fn')),
            ('->', 0, ('->', 1, ('->', 0, 1))))

    def test_mutually_recursive_type(self):
        # A more complex example of an infinite type, involving the
        # mutual recursion of two types, is the expression:
        #
        # (\f . \g . ignore2 (unify f (\x . g)) (unify g (\x . f)))
        #
        # (where "unify" and "ignore2" are defined in def_utils()).
        #
        # If "f" has type "a" and "g" has type "b", this expression
        # forces "a" to be unified with "c -> b" and forces "b" to be
        # unified with "d -> a", resulting in a mutual recursion of
        # two types; which produces an infinite type.
        self.check_type_error(
            self.def_utils(
                parse(r'\f . \g . ignore2 (unify f (\x . g)) (unify g (\x . f))')))

    def test_infinite_type_left(self):
        # Check that producing an infinite type by unifying "a" with
        # "a -> b" produces an error.  We do so by attempting to
        # type-check the expression:
        #
        # (\f . unify f (\x . f))
        self.check_type_error(
            self.def_utils(
                parse(r'\f . unify f (\x . f)')))

    def test_infinite_type_right(self):
        # Check that producing an infinite type by unifying "a -> b"
        # with "a" produces an error.  We do so by attempting to
        # type-check the expression:
        #
        # (\f . unify (\x . f) f)
        self.check_type_error(
            self.def_utils(
                parse(r'\f . unify (\x . f) f')))

    def test_s_combinator(self):
        # Check the type of the "S" combinator:
        #
        # (\x . \y . \z . x z (y z))
        #
        # It sould have type:
        #
        # (a -> b -> c) -> (a -> b) -> a -> c
        self.check_single_expr(
            parse(r'\x . \y . \z . x z (y z)'),
            ('->',
             ('->', 0, ('->', 1, 2)),
             ('->', ('->', 0, 1), ('->', 0, 2))))

    def test_pairs(self):
        # Check the types of the functions involving pairs:
        #
        # mk_pair :: a -> b -> (a, b)
        # fst :: (a, b) -> a
        # snd :: (a, b) -> b
        self.check_single_expr(
            parse('mk_pair'),
            ('->', 0, ('->', 1, ('Pair', 0, 1))))
        self.check_single_expr(
            parse('fst'),
            ('->', ('Pair', 0, 1), 0))
        self.check_single_expr(
            parse('snd'),
            ('->', ('Pair', 0, 1), 1))

    def test_curry(self):
        # Check the type of the curry function
        self.check_single_expr(
            parse(r'\f . \x . \y . f (mk_pair x y)'),
            ('->', ('->', ('Pair', 0, 1), 2), ('->', 0, ('->', 1, 2))))

    def test_uncurry(self):
        # Check the type of the uncurry function
        self.check_single_expr(
            parse(r'\f . \p . f (fst p) (snd p)'),
            ('->', ('->', 0, ('->', 1, 2)), ('->', ('Pair', 0, 1), 2)))

    def test_let_partial_generalization_with_utils(self):
        # More thorough test of "let" partial specialization using
        # utility functions.
        expr = parse(r"""
            \x . \y . \z .
            let const_x = \w . x in
                mk_pair (mk_pair (unify const_x (mk_fn y True))
                                 (unify const_x
                                        (mk_fn (mk_pair True False) z)))
                        const_x
            """)
        # In the expression above, if "x" has type "a", "y" has type
        # "b", and "z" has type "c", then "const_x" is initially
        # assigned type "forall d . d -> a".  On its first use it is
        # unified with "b -> Bool", forcing "a" to be "Bool".  On its
        # second use, it is unified with "(Bool, Bool) -> c", forcing
        # "c" to be "Bool".  Note that since "d" is qualified with
        # "forall", "b" is not unified with "(Bool, Bool)".  Therefore
        # the type of the final expression should be:
        #
        # Bool -> b -> Bool -> ((b -> Bool, (Bool, Bool) -> Bool), d -> Bool)
        self.check_single_expr(
            self.def_utils(expr),
            ('->',
             ('Bool',),
             ('->',
              0,
              ('->',
               ('Bool',),
               ('Pair',
                ('Pair',
                 ('->', 0, ('Bool',)),
                 ('->', ('Pair', ('Bool',), ('Bool',)), ('Bool',))),
                ('->', 1, ('Bool',)))))))

    def test_already_unified(self):
        # Make sure that nothing goes wrong when unifying two types
        # that are already the same.
        self.check_single_expr(
            self.def_utils(parse(r"\x . unify x x")),
            ('->', 0, 0))

    def test_isJust(self):
        self.check_single_expr(
            parse(r'maybe False (\x . True)'),
            ('->', ('Maybe', 0), ('Bool',)))

    def test_boolean_example(self):
        self.check_single_expr(
            parse(r'\b . if b False True'),
            ('->', ('Bool',), ('Bool',)))

    def test_polymorphic_binding_example(self):
        self.check_single_expr(
            parse(r'let id = \x . x in mk_pair (id True) (id id)'),
            ('Pair', ('Bool',), ('->', 0, 0)))

    def test_monomorphic_binding_example(self):
        self.check_type_error(
            parse(r'(\id . mk_pair (id True) (id id)) (\x . x)'))


if __name__ == '__main__':
    unittest.main()
