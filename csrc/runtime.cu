#include <memory>
#include <string>

#include "buffers/advanced.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, extension) {
    using pcie_ep::host_staged::AdvancedRuntime;
    using pcie_ep::host_staged::AdvancedLayout;

    extension.def("create_advanced_segments", &pcie_ep::host_staged::create_advanced_segments);
    extension.def("prepare_advanced_segment", &pcie_ep::host_staged::prepare_advanced_segment);
    extension.def("get_device_numa_node", &pcie_ep::host_staged::get_device_numa_node);
    py::class_<AdvancedLayout, std::shared_ptr<AdvancedLayout>>(extension, "_AdvancedLayout");
    py::class_<AdvancedRuntime, std::shared_ptr<AdvancedRuntime>>(extension, "AdvancedRuntime")
        .def(py::init<std::string, int, int, int, int, int, int, py::dict>())
        .def("dispatch", &AdvancedRuntime::dispatch,
             py::arg("x"), py::arg("idx"), py::arg("weights"), py::arg("stage"), py::arg("alignment"),
             py::arg("expand"), py::arg("zero_padding"), py::arg("cached_layout") = py::none(), py::arg("graph_pool") = false)
        .def("pool_debug", &AdvancedRuntime::pool_debug)
        .def("relay_diagnostics", &AdvancedRuntime::relay_diagnostics)
        .def("set_test_delays", &AdvancedRuntime::set_test_delays)
        .def("set_relay_delay_ns", &AdvancedRuntime::set_relay_delay_ns)
        .def("close", &AdvancedRuntime::close);
}
