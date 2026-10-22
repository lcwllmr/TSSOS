module TSSOS

using Base.Threads
using MosekTools
using JuMP
using Graphs
using DynamicPolynomials
using MultivariatePolynomials
using Ipopt
using LinearAlgebra
using MetaGraphs
using SemialgebraicSets
using Groebner
using Dualization
using Printf
using AbstractAlgebra
using Random
using SymbolicWedderburn
using AbstractPermutations
import Reexport
Reexport.@reexport using MultivariateBases
import DynamicPolynomials as DP
import MultivariatePolynomials as MP
import CliqueTrees

export tssos, cs_tssos, complex_tssos, complex_cs_tssos, LinearPMI, sparseobj
export arrange, bfind, mosek_para
export local_solution, refine_sol, extract_solutions, extract_solutions_robust, extract_solutions_pmo, extract_solutions_pmo_robust, extract_weight_matrix
export add_SOS!, add_SOSMatrix!, add_poly!, add_psatz!, add_complex_psatz!, add_psatz_cheby!, add_poly_cheby!
export OnMonomials, tssos_symmetry, complex_tssos_symmetry, get_signsymmetry, add_psatz_symmetry!
export homogenize, solve_hpop, SumOfRatios, SparseSumOfRatios, get_dynamic_sparsity
export show_blocks, complex_to_real, get_mmoment, get_basis, get_moment, get_moment_matrix, get_cmoment
export run_H1, run_H1CS, run_H2, run_H2CS, construct_CDK, construct_marginal_CDK, construct_CDK_cs, construct_marginal_CDK_cs

mutable struct mosek_para
    tol_pfeas::Float64
    tol_dfeas::Float64
    tol_relgap::Float64
    time_limit::Int64
    num_threads::Int64
end

mosek_para() = mosek_para(1e-8, 1e-8, 1e-8, -1, 0)

include("polynomial.jl")
include("utils.jl")
include("chordal_extension.jl")
include("clique_merge.jl")
include("term_sparsity.jl")
include("all_sparsity.jl")
include("local_solution.jl")
include("extract_solutions.jl")
include("add_psatz.jl")
include("homogenization.jl")
include("sum_of_ratios.jl")
include("matrixsos.jl")
include("dynamic_system.jl")
include("CDK.jl")
include("Chebyshev_basis.jl")
include("complex_pop.jl")
include("symmetry.jl")

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    @polyvar _x[1:2]
    _pop = [_x[1]^2 + _x[2]^2, 1.0 - _x[1]^2 - _x[2]^2, _x[1] - 0.5]
    redirect_stdout(devnull) do
        @compile_workload begin
            _npop = [poly(p, _x) for p in _pop]
            local_solution(_npop, 2; numeq=1, startpoint=[0.5, 0.0], QUIET=true)
            try
                tssos(_pop, _x, 1; numeq=1, TS="block", QUIET=true, solution=true)
                opt, _, data = cs_tssos(_pop, _x, 1; numeq=1, CS="MF", TS="block", QUIET=true, solution=true, solution_mode="moment")
                cs_tssos(_pop, _x, 1; numeq=1, cliques=[[1, 2]], TS="block", QUIET=true, solution=true, solution_mode="local")
                refine_sol(opt, [0.5, 0.0], data; QUIET=true)
            catch
            end
        end
    end
end

end
