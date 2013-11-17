import algorithm_w
import lambda_calculus
import sys

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

Predefined functions:
    mk_pair :: a -> b -> (a, b)
    fst :: (a, b) -> a
    snd :: (a, b) -> b
"""
    exit(1)

expr = lambda_calculus.parse(sys.argv[1])
type_inferrer = algorithm_w.TypeInferrer()
ty = type_inferrer.visit(expr)
canonical_ty = type_inferrer.canonicalize(ty, {})

print """Expression:

{0}

has type:

{1!r}
""".format(expr, canonical_ty)
