#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "qwen3_asr.h"

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;
namespace asr = qwen3_asr;

// ---------------------------------------------------------------------------
// Helper: wrap load_audio_file (out-params -> tuple return)
// ---------------------------------------------------------------------------
static py::tuple py_load_audio_file(const std::string &path) {
    std::vector<float> samples;
    int sample_rate = 0;
    bool ok = asr::load_audio_file(path, samples, sample_rate);
    return py::make_tuple(ok, samples, sample_rate);
}

// ---------------------------------------------------------------------------
// Python module: qwen3_asr._core
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = "Qwen3 ASR — Python bindings";

    // ---- transcribe_params ------------------------------------------------
    py::class_<asr::transcribe_params>(m, "TranscribeParams",
        "Transcription parameters.")
        .def(py::init<>())
        .def_readwrite("max_tokens",    &asr::transcribe_params::max_tokens,
            "Maximum number of tokens to generate.")
        .def_readwrite("language",      &asr::transcribe_params::language,
            "Language code (optional).")
        .def_readwrite("n_threads",     &asr::transcribe_params::n_threads,
            "Number of threads for mel computation.")
        .def_readwrite("print_progress",&asr::transcribe_params::print_progress,
            "Print progress during transcription.")
        .def_readwrite("print_timing",  &asr::transcribe_params::print_timing,
            "Print timing information.")
        .def("__repr__", [](const asr::transcribe_params &p) {
            return "<TranscribeParams max_tokens=" + std::to_string(p.max_tokens)
                 + " n_threads=" + std::to_string(p.n_threads) + ">";
        });

    // ---- transcribe_result ------------------------------------------------
    py::class_<asr::transcribe_result>(m, "TranscribeResult",
        "Result of a transcription call.")
        .def_readonly("text",       &asr::transcribe_result::text)
        .def_readonly("tokens",     &asr::transcribe_result::tokens)
        .def_readonly("success",    &asr::transcribe_result::success)
        .def_readonly("error_msg",  &asr::transcribe_result::error_msg)
        .def_readonly("t_load_ms",  &asr::transcribe_result::t_load_ms)
        .def_readonly("t_mel_ms",   &asr::transcribe_result::t_mel_ms)
        .def_readonly("t_encode_ms",&asr::transcribe_result::t_encode_ms)
        .def_readonly("t_decode_ms",&asr::transcribe_result::t_decode_ms)
        .def_readonly("t_total_ms", &asr::transcribe_result::t_total_ms)
        .def("__repr__", [](const asr::transcribe_result &r) {
            if (r.success) {
                return "<TranscribeResult success text=\""
                    + r.text.substr(0, 80) + "\">";
            }
            return "<TranscribeResult success=false error=\""
                + r.error_msg + "\">";
        });

    // ---- Qwen3ASR (RAII — destructor frees ggml resources) ---------------
    py::class_<asr::Qwen3ASR>(m, "Qwen3ASR",
        "Qwen3 ASR model wrapper. Use as a context manager or call .close().\n\n"
        "Typical usage::\n\n"
        "    asr = Qwen3ASR()\n"
        "    asr.load_model(\"model.gguf\")\n"
        "    result = asr.transcribe(\"audio.wav\")\n"
        "    print(result.text)\n")
        // RAII: construct in Python, destructor auto-runs on GC
        .def(py::init<>())
        .def("load_model", &asr::Qwen3ASR::load_model,
            py::arg("model_path"),
            "Load ASR model from a GGUF file. Returns True on success.")
        .def("transcribe",
            py::overload_cast<const std::string &, const asr::transcribe_params &>(
                &asr::Qwen3ASR::transcribe),
            py::arg("audio_path"),
            py::arg("params") = asr::transcribe_params(),
            "Transcribe a WAV audio file.")
        .def("transcribe_raw",
            [](asr::Qwen3ASR &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> samples,
               const asr::transcribe_params &params) {
                py::buffer_info buf = samples.request();
                return self.transcribe(
                    static_cast<const float *>(buf.ptr),
                    static_cast<int>(buf.size),
                    params);
            },
            py::arg("samples"),
            py::arg("params") = asr::transcribe_params(),
            "Transcribe raw audio samples (float32 numpy array, [-1, 1]).")
        .def("is_loaded", &asr::Qwen3ASR::is_loaded,
            "Check whether a model is loaded.")
        .def("get_error", &asr::Qwen3ASR::get_error,
            "Get the last error message.")
        // Context-manager support — destructor already handles cleanup
        .def("__enter__",
            [](asr::Qwen3ASR &self) -> asr::Qwen3ASR & { return self; })
        .def("__exit__",
            [](asr::Qwen3ASR &self,
               py::object, py::object, py::object) { /* RAII */ });

    // ---- Free functions ---------------------------------------------------
    m.def("load_audio_file", &py_load_audio_file,
        py::arg("path"),
        "Load a WAV file and return (ok, samples, sample_rate).");
}

// vim: ts=4 sw=4 et
