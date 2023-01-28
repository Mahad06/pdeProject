#include "NavierStokes.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int N               = 19;
  const unsigned int degree_velocity = 2;
  const unsigned int degree_pressure = 1;
  const double Re = 200.0;

  const double T      = 0.1;
  const double deltat = 0.0005;

  NavierStokes problem(N, Re, degree_velocity, degree_pressure, T, deltat);

  problem.setup();
  problem.solve();

  return 0;
}