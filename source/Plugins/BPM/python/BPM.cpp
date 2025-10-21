#include <BPM/BPM.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace USTC_CG;

NB_MODULE(BPM_py, m)
{
    m.doc() = "Beam Propagation Method solver";

    // Grid parameters
    nb::class_<GridParameters>(m, "GridParameters")
        .def(nb::init<>())
        .def_rw("Nx", &GridParameters::Nx)
        .def_rw("Ny", &GridParameters::Ny)
        .def_rw("Lx", &GridParameters::Lx)
        .def_rw("Ly", &GridParameters::Ly)
        .def_rw("Lz", &GridParameters::Lz)
        .def_rw("dx", &GridParameters::dx)
        .def_rw("dy", &GridParameters::dy)
        .def_rw("dz", &GridParameters::dz)
        .def_rw("updates", &GridParameters::updates)
        .def_rw("lambda_", &GridParameters::lambda)
        .def_rw("n_0", &GridParameters::n_0)
        .def_rw("n_background", &GridParameters::n_background)
        .def_rw("xSymmetry", &GridParameters::xSymmetry)
        .def_rw("ySymmetry", &GridParameters::ySymmetry)
        .def_rw("alpha", &GridParameters::alpha)
        .def_rw("bendingRoC", &GridParameters::bendingRoC)
        .def_rw("bendDirection", &GridParameters::bendDirection)
        .def_rw("taperScaling", &GridParameters::taperScaling)
        .def_rw("twistRate", &GridParameters::twistRate)
        .def_rw("rho_e", &GridParameters::rho_e)
        .def_rw("useGPU", &GridParameters::useGPU)
        .def_rw("useAllCPUs", &GridParameters::useAllCPUs);

    // Electric field
    nb::class_<ElectricField>(m, "ElectricField")
        .def(nb::init<>())
        .def_rw("field", &ElectricField::field)
        .def_rw("Lx", &ElectricField::Lx)
        .def_rw("Ly", &ElectricField::Ly)
        .def_rw("xSymmetry", &ElectricField::xSymmetry)
        .def_rw("ySymmetry", &ElectricField::ySymmetry);

    // Refractive index
    nb::class_<RefractiveIndex>(m, "RefractiveIndex")
        .def(nb::init<>())
        .def_rw("n", &RefractiveIndex::n)
        .def_rw("Nx", &RefractiveIndex::Nx)
        .def_rw("Ny", &RefractiveIndex::Ny)
        .def_rw("Nz", &RefractiveIndex::Nz)
        .def_rw("Lx", &RefractiveIndex::Lx)
        .def_rw("Ly", &RefractiveIndex::Ly)
        .def_rw("xSymmetry", &RefractiveIndex::xSymmetry)
        .def_rw("ySymmetry", &RefractiveIndex::ySymmetry);

    // Propagation result
    nb::class_<PropagationResult>(m, "PropagationResult")
        .def_rw("finalField", &PropagationResult::finalField)
        .def_rw("finalRI", &PropagationResult::finalRI)
        .def_rw("powers", &PropagationResult::powers)
        .def_rw("z_positions", &PropagationResult::z_positions);

    // Mode
    nb::class_<BPMSolver::Mode>(m, "Mode")
        .def_rw("field", &BPMSolver::Mode::field)
        .def_rw("neff", &BPMSolver::Mode::neff)
        .def_rw("label", &BPMSolver::Mode::label);

    // Main solver
    nb::class_<BPMSolver>(m, "BPMSolver")
        .def(nb::init<const GridParameters&>())
        .def(
            "initializeRI",
            &BPMSolver::initializeRI,
            "Initialize refractive index from function")
        .def(
            "initializeE",
            &BPMSolver::initializeE,
            "Initialize electric field from function")
        .def("setRI", &BPMSolver::setRI, "Set refractive index from data")
        .def("setE", &BPMSolver::setE, "Set electric field from data")
        .def(
            "propagateFDBPM",
            &BPMSolver::propagateFDBPM,
            "Propagate using Finite Difference BPM")
        .def(
            "propagateFFTBPM",
            &BPMSolver::propagateFFTBPM,
            "Propagate using FFT BPM")
        .def(
            "findModes",
            &BPMSolver::findModes,
            "Find propagation modes",
            nb::arg("nModes"),
            nb::arg("sortByLoss") = false);
}
