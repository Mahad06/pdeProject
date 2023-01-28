#ifndef NAVIERSTOKES_HPP
#define NAVIERSTOKES_HPP


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class NavierStokes
{
   
public:

    // Physical dimension.
    static constexpr unsigned int dim = 2;

    // Forcing term.
    class ForcingTerm: public Function<dim>
    {
    public:

        virtual void
        vector_value(const Point<dim> & /*p*/,
                    Vector<double> &values) const override
        {
            for (unsigned int i = 0; i < dim - 1; ++i)
                values[i] = 0.0;

            values[dim - 1] = -g;
        }

        virtual double
        value(const Point<dim> & /*p*/,
            const unsigned int component = 0) const override
        {
            if (component == dim - 1)
                return -g;
            else
                return 0.0;
        }

    protected:
        const double g = 0.0;

    };

    // Inlet velocity
    class InletVelocity : public Function<dim>
    {
    public:

        // Constructor.
        InletVelocity(double Re_)
         : Function<dim> (dim + 1)
         , Re(Re_)
        {}

        virtual void
        vector_value(const Point<dim> & p, Vector<double> & values) const override
        {
            if (dim == 2)
                values[0] = 4*u_m*(p[1]*(H-p[1])/(H*H)+offset)*std::sin(M_PI*get_time());    // 2D
            else if (dim == 3)
                values[0] = 16 * u_m * p[1] * p[2] * (H - p[1]) * (H-p[2]) / (std::pow(H, 4))*std::sin(M_PI*get_time());; // 3D

            for(unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
            
        } 

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0){
                if (dim == 2)
                    return 4*u_m*(p[1]*(H-p[1])/(H*H)+offset)*std::sin(M_PI*get_time());    // 2D
                else if (dim == 3)
                    return 16 * u_m * p[1] * p[2] * (H - p[1]) * (H-p[2]) / (std::pow(H, 4))*std::sin(M_PI*get_time());; // 3D
            }
            else    
                return 0.0;
        }

        virtual double
        mean_value() 
        {
            if (dim == 2)
                return  4.0*u_m*(1.0/6.0 + offset)*std::sin(M_PI*get_time());                     //(2.0/3.0)*u_m*std::sin(M_PI*get_time());    // 2D
            else if (dim == 3)
                return (4.0/9.0)*u_m*std::sin(M_PI*get_time());  // 3D
            else
                return 0.0;
        }

    private:

        // height of box [m]
        const double H = 0.41;

        // Reynolds Number
        double Re;

        // Kinematic viscosity [m^2/s].
        const double nu = 1.0;

        // average velocity [m/s]
        double u_m = dim == 2 ? 3*Re*nu/(2*0.1) : 9*Re*nu/(4*0.1);

        const double offset = 0.05;
    };

     // wall boundary condition
    class FunctionG : public Function<dim>
    {
    public:

        // Constructor.
        FunctionG(double Re_)
         : Function<dim> (dim + 1)
         , Re(Re_)
        {}

        virtual void
        vector_value(const Point<dim> & /*p*/, Vector<double> & values) const override
        {
            values[0] = 4*u_m*std::sin(M_PI*get_time())*offset;    // 2D

            for(unsigned int i = 1; i < dim + 1; ++i)
                values[i] = 0.0;
            
        } 

        virtual double
        value(const Point<dim> &/*p*/, const unsigned int component = 0) const override
        {
            if (component == 0)
                return 4*u_m*std::sin(M_PI*get_time())*offset;    // 2D
            else    
                return 0.0;
        }

    private:

        // height of box [m]
        const double H = 0.41;

        // Reynolds Number
        double Re;

        // Kinematic viscosity [m^2/s].
        const double nu = 1.0;

        // average velocity [m/s]
        double u_m = dim == 2 ? 3*Re*nu/(2*0.1) : 9*Re*nu/(4*0.1);

        const double offset = 0.05;
    };

    // Initial velocity
    class FunctionU0 : public Function<dim>
    {
    public:

        // Constructor.
        FunctionU0()
         : Function<dim>(dim + 1)
        {}

        virtual void
        vector_value(const Point<dim> &/*p*/, Vector<double> & values) const override
        {
            for (unsigned int i = 0; i < dim+1; ++i)  // i = 1
                values[i] = 0.0;

        }

        virtual double
        value(const Point<dim> &/*p*/, const unsigned int /*component*/ = 0) const override
        {
                return 0.0;
        }

    };

      // Block-triangular preconditioner.
    class PreconditionBlockTriangular
    {
    public:
        // Initialize the preconditioner, given the velocity stiffness matrix, the
        // pressure mass matrix.
        void
        initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                const TrilinosWrappers::SparseMatrix &pressure_mass_,
                const TrilinosWrappers::SparseMatrix &B_)
        {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;
        B                  = &B_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
        }

        // Application of the preconditioner.
        void
        vmult(TrilinosWrappers::MPI::BlockVector &      dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
        SolverControl                           solver_control_velocity(1000,
                                                1e-3);
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_velocity(
            solver_control_velocity);
        solver_gmres_velocity.solve(*velocity_stiffness,
                                dst.block(0),
                                src.block(0),
                                preconditioner_velocity);

        tmp.reinit(src.block(1));
        B->vmult(tmp, dst.block(0));
        tmp.sadd(-1.0, src.block(1));

        SolverControl                           solver_control_pressure(1000,
                                                1e-3);
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_pressure(
            solver_control_pressure);
        solver_gmres_pressure.solve(*pressure_mass,
                                dst.block(1),
                                tmp,
                                preconditioner_pressure);
        }

    protected:
        // Velocity stiffness matrix.
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        // Preconditioner used for the velocity block.
        TrilinosWrappers::PreconditionILU preconditioner_velocity;

        // Pressure mass matrix.
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        // Preconditioner used for the pressure block.
        TrilinosWrappers::PreconditionILU preconditioner_pressure;

        // B matrix.
        const TrilinosWrappers::SparseMatrix *B;

        // Temporary vector.
        mutable TrilinosWrappers::MPI::Vector tmp;
    };

    // Block-diagonal preconditioner.
    class PreconditionBlockDiagonal
    {
    public:
        // Initialize the preconditioner, given the velocity stiffness matrix, the
        // pressure mass matrix.
        void
        initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
                const TrilinosWrappers::SparseMatrix &pressure_mass_)
        {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass      = &pressure_mass_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
        }

        // Application of the preconditioner.
        void
        vmult(TrilinosWrappers::MPI::BlockVector &      dst,
            const TrilinosWrappers::MPI::BlockVector &src) const
        {
        SolverControl                           solver_control_velocity(1000,
                                                1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
            solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                dst.block(0),
                                src.block(0),
                                preconditioner_velocity);
        
        SolverControl                           solver_control_pressure(1000,
                                                1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
            solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                dst.block(1),
                                src.block(1),
                                preconditioner_pressure);
        }

    protected:
        // Velocity stiffness matrix.
        const TrilinosWrappers::SparseMatrix *velocity_stiffness;

        // Preconditioner used for the velocity block.
        TrilinosWrappers::PreconditionILU preconditioner_velocity;

        // Pressure mass matrix.
        const TrilinosWrappers::SparseMatrix *pressure_mass;

        // Preconditioner used for the pressure block.
        TrilinosWrappers::PreconditionILU preconditioner_pressure;
    };

    // Constructor.
    NavierStokes(const unsigned int &N_,
            const double &Re_,
            const unsigned int &degree_velocity_,
            const unsigned int &degree_pressure_,
            const double &      T_,
            const double &      deltat_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , N(N_)
    , Re(Re_)
    , degree_velocity(degree_velocity_)
    , degree_pressure(degree_pressure_)
    , T(T_)
    , deltat(deltat_)
    , inlet_velocity(Re_)
    , function_g(Re_)
    , mesh(MPI_COMM_WORLD)
    {}

    // Setup system.
    void
    setup();

    // Solve system.
    void
    solve();

private:

    // Assemble system
    void
    assemble_matrices();

    // Assemble the time-varying matrices side of the problem.
    void
    assemble_time_matrices(const double &time);

    // Solve the problem for one time step.
    void
    solve_time_step();

    // Output.
    void
    output(const unsigned int &time_step, const double &time) const;

    // lift and drag force.
    std::vector<double>
    calculate_force();


    /////// MPI parallel ///////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    /////// Problem definition ///////
    // viscosity, outlet pressure, forcing term, inlet velocity, Reynolds number, lift and drag coefficient


    // Kinematic viscosity [m^2/s].
    const double nu = 1.0;

    // Outlet pressure [Pa]
    const double p_out = 10.0;

    // Forcing term.
    ForcingTerm forcing_term;


    /////// Discretization ///////
    
    // Mesh refinement.
    const unsigned int N;

    // Reynolds Number
    const double Re;

    // Polynomial degree used for velocity.
    const unsigned int degree_velocity;

    // Polynomial degree used for pressure.
    const unsigned int degree_pressure;

    // Final time.
    const double T;

    // Time step delta t
    const double deltat;

    // Inlet velocity.
    InletVelocity inlet_velocity;

    // Initial condition.
    FunctionU0 u_0;

    FunctionG function_g;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula for face integrals.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs owned by current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // DoFs relevant to current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_relevant_dofs;


    // left hand side matrix that is steady
    TrilinosWrappers::BlockSparseMatrix lhs_matrix;

    // Pressure matrix for preconditioner.
    TrilinosWrappers::BlockSparseMatrix pressure_mass;

    // right hand side matrix
    TrilinosWrappers::BlockSparseMatrix rhs_matrix;

    // Right-hand side vector with the neumann bcs in the linear system.
    TrilinosWrappers::MPI::BlockVector rhs_neumann;

    // System matrix.
    TrilinosWrappers::BlockSparseMatrix system_matrix;

    // Right-hand side vector of the linear system.
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;

    // System solutio at previous time step.     
    TrilinosWrappers::MPI::BlockVector solution_old;
};




#endif