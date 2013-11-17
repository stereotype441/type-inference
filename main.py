import algorithm_w
import lambda_calculus
import sys

type_inferrer = algorithm_w.TypeInferrer()

if len(sys.argv) != 2:
    print """Usage: python main.py <expression>

Expression grammar:
    expression: '(' expression ')' |
                '\\' identifier '.' expression |
                'let' identifier '=' expression 'in' expression |
                expression expression |
                identifier

Predefined literals:
    True :: Bool
    False :: Bool

Predefined symbols:"""
    for name in sorted(type_inferrer.get_builtin_names()):
        expr = lambda_calculus.Variable(name)
        ty = type_inferrer.visit(expr)
        canonical_ty = type_inferrer.canonicalize(ty, {})
        print '    {0} :: {1}'.format(
            name, algorithm_w.pretty_print_canonical_type(canonical_ty))
    exit(1)

expr = lambda_calculus.parse(sys.argv[1])
ty = type_inferrer.visit(expr)
canonical_ty = type_inferrer.canonicalize(ty, {})

print """Expression:

{0}

has type:

{1!r}
""".format(expr, algorithm_w.pretty_print_canonical_type(canonical_ty))
