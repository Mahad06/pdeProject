#include "NavierStokes.hpp"


void
NavierStokes::setup()
{
    // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    const std::string mesh_file_name =
      "../mesh/" + std::to_string(dim) + "Dmesh-step-" + std::to_string(N + 1) + ".msh";

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
    const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
    fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                         dim,
                                         fe_scalar_pressure,
                                         1);

    pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
          << std::endl;
    pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
          << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;

    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We want to reorder DoFs so that all velocity DoFs come first, and then
    // all pressure DoFs.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    // Besides the locally owned and locally relevant indices for the whole
    // system (velocity and pressure), we will also need those for the
    // individual velocity and pressure blocks.
    std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    const unsigned int n_u = dofs_per_block[0];
    const unsigned int n_p = dofs_per_block[1];

    block_owned_dofs.resize(2);
    block_relevant_dofs.resize(2);
    block_owned_dofs[0]    = locally_owned_dofs.get_view(0, n_u);
    block_owned_dofs[1]    = locally_owned_dofs.get_view(n_u, n_u + n_p);
    block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
    block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    pcout << "  Number of DoFs: " << std::endl;
    pcout << "    velocity = " << n_u << std::endl;
    pcout << "    pressure = " << n_p << std::endl;
    pcout << "    total    = " << n_u + n_p << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // Velocity DoFs interact with other velocity DoFs (the weak formulation has
    // terms involving u times v), and pressure DoFs interact with velocity DoFs
    // (there are terms involving p times v or u times q). However, pressure
    // DoFs do not interact with other pressure DoFs (there are no terms
    // involving p times q). We build a table to store this information, so that
    // the sparsity pattern can be built accordingly.
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::none;
            else // other combinations
              coupling[c][d] = DoFTools::always;
          }
      }

    TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                    MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
    sparsity.compress();

    // We also build a sparsity pattern for the pressure mass matrix.
    for (unsigned int c = 0; c < dim + 1; ++c)
      {
        for (unsigned int d = 0; d < dim + 1; ++d)
          {
            if (c == dim && d == dim) // pressure-pressure term
              coupling[c][d] = DoFTools::always;
            else // other combinations
              coupling[c][d] = DoFTools::none;
          }
      }
    TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
      block_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sparsity_pressure_mass);
    sparsity_pressure_mass.compress();

    pcout << "  Initializing the matrices" << std::endl;
    lhs_matrix.reinit(sparsity);
    rhs_matrix.reinit(sparsity);
    system_matrix.reinit(sparsity);
    pressure_mass.reinit(sparsity_pressure_mass);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
    rhs_neumann.reinit(block_owned_dofs, MPI_COMM_WORLD);

    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    solution_old = solution; 
  }
}

void
NavierStokes::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the time-independent matrices and vectors" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FullMatrix<double> cell_lhs_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_rhs_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs_neumann(dofs_per_cell);
  FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  lhs_matrix         = 0.0;
  rhs_matrix         = 0.0;
  rhs_neumann        = 0.0;
  pressure_mass      = 0.0;

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_lhs_matrix           = 0.0; 
      cell_rhs_matrix           = 0.0;
      cell_rhs_neumann          = 0.0;
      cell_pressure_mass_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass term of velocity.
                  cell_lhs_matrix(i, j) += 1/deltat*
                    scalar_product(fe_values[velocity].value(j, q),
                                   fe_values[velocity].value(i,q))*
                    fe_values.JxW(q);

                  // Viscosity term of velocity.
                  cell_lhs_matrix(i, j) += nu*
                    scalar_product(fe_values[velocity].gradient(j,q),
                                   fe_values[velocity].gradient(i,q))*
                    fe_values.JxW(q);

                  // Pressure term in momentum equation.
                  cell_lhs_matrix(i, j) -= fe_values[velocity].divergence(i,q)*
                                           fe_values[pressure].value(j,q)*
                                           fe_values.JxW(q);
                  
                  // Pressure term in continuity equation.
                  cell_lhs_matrix(i,j) -= fe_values[velocity].divergence(j,q)*
                                          fe_values[pressure].value(i,q)*
                                          fe_values.JxW(q);

                  // Mass term in rhs
                  cell_rhs_matrix(i,j) += 1/deltat*
                    scalar_product(fe_values[velocity].value(j, q),
                                   fe_values[velocity].value(i,q))*
                    fe_values.JxW(q);

                  // Pressure mass matrix.
                  cell_pressure_mass_matrix(i, j) +=
                    fe_values[pressure].value(i, q) *
                    fe_values[pressure].value(j, q) * fe_values.JxW(q);
                }
            }
        }

      // Boundary integral for Neumann BCs.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == 2)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          cell_rhs_neumann(i) +=-p_out *scalar_product(fe_face_values.normal_vector(q),
                                           fe_face_values[velocity].value(i, q)) *
                            fe_face_values.JxW(q);
                        }
                    }
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      lhs_matrix.add(dof_indices, cell_lhs_matrix);
      rhs_matrix.add(dof_indices, cell_rhs_matrix);
      rhs_neumann.add(dof_indices, cell_rhs_neumann);
      pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

  lhs_matrix.compress(VectorOperation::add);
  rhs_matrix.compress(VectorOperation::add);
  rhs_neumann.compress(VectorOperation::add);
  pressure_mass.compress(VectorOperation::add);
}

void
NavierStokes::assemble_time_matrices(const double &time)
{
  
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim>  fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs        = 0.0;
  system_matrix     = 0.0;
  
  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_n(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs    = 0.0;
      cell_matrix = 0.0;

      fe_values[velocity].get_function_values(solution_old, velocity_n);
      
      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Compute f(tn+1)
          forcing_term.set_time(time);
          Vector<double> f_new_loc(dim);
          forcing_term.vector_value(fe_values.quadrature_point(q), f_new_loc);
          Tensor<1, dim> f_new_loc_tensor;

          for (unsigned int k = 0; k < dim; ++k)
            f_new_loc_tensor[k] = f_new_loc[k];  
          
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                cell_matrix(i, j) += scalar_product(fe_values[velocity].gradient(j, q) * velocity_n[q],
                                      fe_values[velocity].value(i, q)) * fe_values.JxW(q);
              }
              
              cell_rhs(i) += scalar_product(f_new_loc_tensor,
                             fe_values[velocity].value(i, q)) *
                             fe_values.JxW(q);
            }
          
        }
      
      cell->get_dof_indices(dof_indices);

      system_rhs.add(dof_indices, cell_rhs);
      system_matrix.add(dof_indices, cell_matrix);
    }

  system_rhs.compress(VectorOperation::add);
  system_matrix.compress(VectorOperation::add);

  // Add terms to rhs.
  rhs_matrix.vmult_add(system_rhs, solution_old);
  system_rhs.add(1.0, rhs_neumann);

  // Add terms to lhs.
  system_matrix.add(1.0, lhs_matrix);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double>           boundary_values;
    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    inlet_velocity.set_time(time);
    function_g.set_time(time);
    
    std::vector<bool> mask(dim+1);
    for (unsigned int i = 0; i < dim; i++)
      mask[i] = true;

    mask[dim] = false;

    boundary_functions[0] = &inlet_velocity;
    VectorTools::interpolate_boundary_values(dof_handler,
                                            boundary_functions,
                                            boundary_values,
                                            ComponentMask(
                                              mask));   

    boundary_functions.clear(); 
    Functions::ZeroFunction<dim> zero_function(dim + 1);
    boundary_functions[1] = &function_g;
    boundary_functions[3] = &zero_function;
    VectorTools::interpolate_boundary_values(dof_handler,
                                            boundary_functions,
                                            boundary_values,
                                            ComponentMask(
                                              mask));
    
    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
NavierStokes::solve_time_step()
{
  SolverControl solver_control(50000, 1e-6 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

  //Preconditioner has a lower triangular shape: [F 0; B Mp]
  PreconditionBlockTriangular preconditioner;
  preconditioner.initialize(system_matrix.block(0, 0),
                            pressure_mass.block(1, 1),
                            system_matrix.block(1, 0));

  // Preconditioner has a diagonal shape: [F 0; 0 Mp]
  // PreconditionBlockDiagonal preconditioner;
  // preconditioner.initialize(system_matrix.block(0, 0),
  //                           pressure_mass.block(1, 1));
 
  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);          
  pcout << "  " << solver_control.last_step() << " GMRES iterations" << std::endl;

  solution = solution_owned;
}

void
NavierStokes::output(const unsigned int &time_step, const double &time) const
{
  DataOut<dim> data_out;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  
  std::vector<std::string> names(dim+1);
  for (unsigned int i = 0; i < dim; ++i)
    names[i] = "velocity";
  
  names[dim] = "pressure";

  data_out.add_data_vector(dof_handler, 
                           solution, 
                           names, 
                           data_component_interpretation);
  
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  std::string output_file_name = std::to_string(time_step);

  // Pad with zeros.
  // output_file_name = "output-" + std::string(4 - output_file_name.size(), '0') +
  //                    output_file_name;
  
  output_file_name = "output-" 
                      + std::string(4 - output_file_name.size(), '0') +  output_file_name;

  DataOutBase::DataOutFilter data_filter(
  DataOutBase::DataOutFilterFlags(/*filter_duplicate_vertices = */ false,
                                    /*xdmf_hdf5_output = */ true));
  data_out.write_filtered_data(data_filter); 
  data_out.write_hdf5_parallel(data_filter, "/scratch/hpc/par5/results/"+
                               output_file_name + ".h5",      
                               MPI_COMM_WORLD);
  
  std::vector<XDMFEntry> xdmf_entries({data_out.create_xdmf_entry(
    data_filter, output_file_name + ".h5", time, MPI_COMM_WORLD)});
  data_out.write_xdmf_file(xdmf_entries, "/scratch/hpc/par5/results/"+
                           output_file_name + ".xdmf",
                           MPI_COMM_WORLD);
}


void
NavierStokes::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution_owned);
    solution = solution_owned;

    // Output the initial solution.
    output(0, 0.0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;
  std::vector<double> lift_force;
  std::vector<double> drag_force;
  std::vector<double> lift_coeff;
  std::vector<double> drag_coeff;
  std::vector<double> forces;
  std::vector<double> ReynoldsNumber;
  std::vector<double> timeVector;
  // std::ofstream coeff("coeff_data.csv");
  std::ofstream coeff("/scratch/hpc/par5/results/coeff_data.csv");
  coeff << "time,Re,F_L,cL,F_D,cD" << std::endl;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      solution_old = solution;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_time_matrices(time);
      solve_time_step();
      output(time_step, time);

      forces = calculate_force();
      coeff << time << "," << inlet_velocity.mean_value()*0.1/nu  << "," << forces[0] << "," <<
         2.0*forces[0] / (inlet_velocity.mean_value()*inlet_velocity.mean_value()*0.1) << "," << forces[1] << "," 
         << 2.0*forces[1] / (inlet_velocity.mean_value()*inlet_velocity.mean_value()*0.1) << std::endl;
    }
 
}


std::vector<double>
NavierStokes::calculate_force()
{
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();
  double F_L = 0.0;               // lift force
  double F_D = 0.0;               // lift force
  std::vector<double> force;
  Tensor<1, dim> n;               // normal vector

  FEValues<dim>     fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,
                                   *quadrature_face,
                                   update_values | update_normal_vectors |
                                     update_JxW_values);

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> velocity_loc(n_q);
  std::vector<double> pressure_loc(n_q);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      fe_values[velocity].get_function_values(solution, velocity_loc);
      fe_values[pressure].get_function_values(solution, pressure_loc);

      // Boundary integral for lift and drag forces.
      if (cell->at_boundary())
        {
          for (unsigned int f = 0; f < cell->n_faces(); ++f)
            {
              if (cell->face(f)->at_boundary() &&
                  cell->face(f)->boundary_id() == 3)
                {
                  fe_face_values.reinit(cell, f);

                  for (unsigned int q = 0; q < n_q_face; ++q)
                    {
                      n = fe_face_values.normal_vector(q);                         // normal vector
                      F_L += pressure_loc[q] * n[1] * fe_face_values.JxW(q);    // lift force

                      F_D += pressure_loc[q] * n[0] * fe_face_values.JxW(q);      // drag force
                    }
                }
            }
        }

    }

  force.push_back(F_L);
  force.push_back(F_D);
  return force;
}