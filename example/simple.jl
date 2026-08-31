using DynamicPolynomials
using TSSOS

@polyvar x y

# objective + inequalities + equalities (equalities last; numeq=1)
pop = [
    x^2 + y^2,          # objective
    1 - x^2 - y^2,      # inequality >= 0
    x + y - 1           # equality == 0
]

opt, sol, data = cs_tssos(pop, [x, y], 1; numeq=1, TS="block", CS="MF", QUIET=false, solution=false)

println("opt = ", opt)
