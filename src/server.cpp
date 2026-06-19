#include "server.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

// ============================================================
// JSON builder helpers (lightweight, no nlohmann dependency)
// ============================================================

std::string asr_server::json_string(const std::string &s) {
  std::string escaped;
  escaped.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
    case '"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += c;
    }
  }
  return "\"" + escaped + "\"";
}

std::string asr_server::json_pair(const std::string &key,
                                  const std::string &val) {
  return json_string(key) + ": " + json_string(val);
}

std::string asr_server::json_pair(const std::string &key, long val) {
  return json_string(key) + ": " + std::to_string(val);
}

std::string asr_server::json_pair(const std::string &key, double val,
                                  int precision) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision) << val;
  return json_string(key) + ": " + oss.str();
}

std::string asr_server::json_object(const std::vector<std::string> &pairs) {
  std::string result = "{";
  for (size_t i = 0; i < pairs.size(); i++) {
    if (i > 0)
      result += ", ";
    result += pairs[i];
  }
  result += "}";
  return result;
}

std::string asr_server::json_array(const std::vector<std::string> &elements) {
  std::string result = "[";
  for (size_t i = 0; i < elements.size(); i++) {
    if (i > 0)
      result += ", ";
    result += elements[i];
  }
  result += "]";
  return result;
}

// ============================================================
// Slot Pool implementation
// ============================================================

asr_slot_pool::asr_slot_pool(int n_slots, qwen3_asr::Qwen3ASR &model)
    : n_slots_(n_slots), model_(model) {
  slots_.resize(n_slots);
  for (int i = 0; i < n_slots; i++) {
    slots_[i].id = i;
  }
}

asr_slot_pool::~asr_slot_pool() {
  // all slots released by now
}

int asr_slot_pool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (int i = 0; i < n_slots_; i++) {
    if (!slots_[i].busy.load()) {
      slots_[i].reset();
      slots_[i].busy.store(true);
      return i;
    }
  }
  return -1;
}

void asr_slot_pool::release(int slot_id) {
  if (slot_id < 0 || slot_id >= n_slots_)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  slots_[slot_id].reset();
}

// ============================================================
// Server implementation
// ============================================================

asr_server::asr_server(const asr_server_params &params) : params_(params) {}

asr_server::~asr_server() { stop(); }

bool asr_server::init() {
  ggml_time_init();

  // Initialize model
  model_ = std::make_unique<qwen3_asr::Qwen3ASR>();
  if (!model_->load_model(params_.model_path)) {
    fprintf(stderr, "Failed to load model from %s\n",
            params_.model_path.c_str());
    return false;
  }

  // Initialize slot pool
  slot_pool_ = std::make_unique<asr_slot_pool>(params_.n_parallel, *model_);

  // Create HTTP server
  svr_ = std::make_unique<httplib::Server>();

  // Timeouts & limits
  svr_->set_read_timeout(120, 0);
  svr_->set_write_timeout(120, 0);
  svr_->set_payload_max_length(50 * 1024 * 1024);

  // Error handler
  svr_->set_error_handler(
      [](const httplib::Request &req, httplib::Response &res) {
        if (res.status == 404) {
          res.set_content("{\"error\":\"Not Found\",\"message\":\"The "
                          "requested endpoint does not exist\"}",
                          "application/json");
        }
      });

  // Exception handler
  svr_->set_exception_handler([](const httplib::Request &req,
                                 httplib::Response &res,
                                 std::exception_ptr ep) {
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception &e) {
      std::string body = "{\"error\":\"Internal Server Error\",\"message\":" +
                         asr_server::json_string(e.what()) + "}";
      res.status = 500;
      res.set_content(body, "application/json");
    }
  });

  register_routes();
  return true;
}

void asr_server::register_routes() {
  // Health
  svr_->Get("/health",
            [this](const httplib::Request &req, httplib::Response &res) {
              handle_health(req, res);
            });
  svr_->Get("/v1/health",
            [this](const httplib::Request &req, httplib::Response &res) {
              handle_health(req, res);
            });

  // OpenAI-compatible transcription
  svr_->Post("/v1/audio/transcriptions",
             [this](const httplib::Request &req, httplib::Response &res) {
               handle_transcriptions(req, res);
             });

  // Models
  svr_->Get("/v1/models",
            [this](const httplib::Request &req, httplib::Response &res) {
              handle_models(req, res);
            });
}

void asr_server::handle_health(const httplib::Request & /*req*/,
                               httplib::Response &res) {
  std::string body = json_object(
      {json_pair("status", ready_.load() ? "ok" : "loading"),
       json_pair("model", params_.model_path),
       json_pair("n_parallel", static_cast<long>(params_.n_parallel))});
  res.status = ready_.load() ? 200 : 503;
  res.set_content(body, "application/json");
}

void asr_server::handle_models(const httplib::Request & /*req*/,
                               httplib::Response &res) {
  std::string model_entry =
      json_object({json_pair("id", "qwen3-asr"), json_pair("object", "model"),
                   json_pair("created", static_cast<long>(std::time(nullptr))),
                   json_pair("owned_by", "user")});

  std::string body = json_object(
      {json_pair("object", "list"), "\"data\": " + json_array({model_entry})});
  res.set_content(body, "application/json");
}

void asr_server::handle_transcriptions(const httplib::Request &req,
                                       httplib::Response &res) {
  if (!ready_.load()) {
    std::string body = json_object(
        {json_pair("error", "Service Unavailable"),
         json_pair("message", "Server is still loading or not ready")});
    res.status = 503;
    res.set_content(body, "application/json");
    return;
  }

  // --- Parse multipart form-data ---
  auto content_type = req.get_header_value("Content-Type");
  if (content_type.find("multipart/form-data") == std::string::npos) {
    std::string body = json_object(
        {json_pair("error", "Bad Request"),
         json_pair("message", "Content-Type must be multipart/form-data")});
    res.status = 400;
    res.set_content(body, "application/json");
    return;
  }

  // Get uploaded file
  auto files = req.form.get_files("file");
  if (files.empty()) {
    std::string body = json_object({json_pair("error", "Bad Request"),
                                    json_pair("message", "No file provided")});
    res.status = 400;
    res.set_content(body, "application/json");
    return;
  }
  const auto &file = files[0];

  // Optional response_format
  std::string response_format = "json";
  auto fmt = req.get_param_value("response_format");
  if (!fmt.empty()) {
    response_format = fmt;
  }

  // Parse WAV from memory
  std::vector<float> audio_data;
  int sample_rate = 16000;
  int channels = 1;

  const auto &raw = file.content;
  std::vector<uint8_t> raw_bytes(raw.begin(), raw.end());

  if (!parse_wav_from_memory(raw_bytes, sample_rate, channels, audio_data)) {
    std::string body = json_object(
        {json_pair("error", "Bad Request"),
         json_pair(
             "message",
             "Failed to parse WAV audio file. Only PCM WAV is supported.")});
    res.status = 400;
    res.set_content(body, "application/json");
    return;
  }

  // Convert to mono 16kHz if needed
  if (channels > 1 || sample_rate != 16000) {
    audio_data = convert_to_mono_16k(audio_data, sample_rate, channels);
    sample_rate = 16000;
  }

  // --- Acquire slot ---
  int slot_id = slot_pool_->acquire();
  if (slot_id < 0) {
    std::string body = json_object(
        {json_pair("error", "Too Many Requests"),
         json_pair("message",
                   "All inference slots are busy. Please try again later.")});
    res.status = 429;
    res.set_content(body, "application/json");
    return;
  }

  // --- Run inference in this slot ---
  auto &slot = slot_pool_->get(slot_id);
  bool success = false;

  try {
    qwen3_asr::transcribe_params tp;
    tp.n_threads = params_.n_threads;
    tp.print_progress = params_.debug_mode;
    tp.print_timing = params_.debug_mode;

    auto result =
        model_->transcribe(audio_data.data(), (int)audio_data.size(), tp);
    if (result.success) {
      slot.result_text = result.text;
      success = true;
    } else {
      slot.error_message = result.error_msg.empty()
                               ? "Inference returned no result"
                               : result.error_msg;
    }
  } catch (const std::exception &e) {
    slot.error_message = e.what();
  }

  std::string text = slot.result_text;
  std::string err = slot.error_message;
  slot_pool_->release(slot_id);

  if (!success) {
    std::string body = json_object(
        {json_pair("error", "Internal Server Error"),
         json_pair("message", err.empty() ? "Inference failed" : err)});
    res.status = 500;
    res.set_content(body, "application/json");
    return;
  }

  // --- Build response ---
  if (response_format == "text") {
    res.set_content(text, "text/plain; charset=utf-8");
  } else if (response_format == "srt") {
    double dur = (double)audio_data.size() / (double)sample_rate;
    int total_sec = (int)dur;
    int h = total_sec / 3600;
    int m = (total_sec % 3600) / 60;
    int s = total_sec % 60;
    std::ostringstream ss;
    ss << "1\n"
       << std::setw(2) << std::setfill('0') << h << ":" << std::setw(2)
       << std::setfill('0') << m << ":" << std::setw(2) << std::setfill('0')
       << s << ",000 --> " << std::setw(2) << std::setfill('0') << h << ":"
       << std::setw(2) << std::setfill('0') << m << ":" << std::setw(2)
       << std::setfill('0') << s << ",000\n"
       << text << "\n\n";
    res.set_content(ss.str(), "text/plain; charset=utf-8");
  } else if (response_format == "vtt") {
    std::ostringstream ss;
    ss << "WEBVTT\n\n00:00:00.000 --> 00:00:00.000\n" << text << "\n";
    res.set_content(ss.str(), "text/plain; charset=utf-8");
  } else if (response_format == "verbose_json") {
    double dur = (double)audio_data.size() / (double)sample_rate;
    std::string body = json_object(
        {json_pair("task", "transcribe"), json_pair("language", "chinese"),
         json_pair("duration", dur), json_pair("text", text),
         "\"segments\": []"});
    res.set_content(body, "application/json");
  } else {
    // default: json
    std::string body = json_object({json_pair("text", text)});
    res.set_content(body, "application/json");
  }
}

// ============================================================
// WAV parsing from memory buffer
// ============================================================

bool asr_server::parse_wav_from_memory(const std::vector<uint8_t> &data,
                                       int &sample_rate, int &channels,
                                       std::vector<float> &samples) {
  if (data.size() < 44)
    return false;

  // RIFF header
  if (data[0] != 'R' || data[1] != 'I' || data[2] != 'F' || data[3] != 'F')
    return false;
  if (data[8] != 'W' || data[9] != 'A' || data[10] != 'V' || data[11] != 'E')
    return false;

  // fmt chunk
  if (data[12] != 'f' || data[13] != 'm' || data[14] != 't' || data[15] != ' ')
    return false;

  uint16_t audio_format;
  memcpy(&audio_format, &data[20], 2);
  if (audio_format != 1 && audio_format != 3)
    return false; // PCM or IEEE float

  memcpy(&channels, &data[22], 2);
  memcpy(&sample_rate, &data[24], 4);
  uint16_t bits_per_sample;
  memcpy(&bits_per_sample, &data[34], 2);

  if (channels == 0 || sample_rate == 0 || bits_per_sample == 0)
    return false;

  // Find data chunk
  size_t data_start = 0;
  size_t data_size = 0;
  for (size_t i = 36; i < data.size() - 8;) {
    if (data[i] == 'd' && data[i + 1] == 'a' && data[i + 2] == 't' &&
        data[i + 3] == 'a') {
      memcpy(&data_size, &data[i + 4], 4);
      data_start = i + 8;
      break;
    }
    uint32_t chunk_size;
    memcpy(&chunk_size, &data[i + 4], 4);
    i += 8 + chunk_size;
    if (i >= data.size())
      break;
  }

  if (data_size == 0 || data_start + data_size > data.size())
    return false;

  size_t num_samples = data_size / (bits_per_sample / 8);
  samples.reserve(num_samples);

  if (bits_per_sample == 16 && audio_format == 1) {
    const int16_t *pcm = reinterpret_cast<const int16_t *>(&data[data_start]);
    for (size_t i = 0; i < num_samples; i++) {
      samples.push_back((float)pcm[i] / 32768.0f);
    }
  } else if (bits_per_sample == 32 && audio_format == 3) {
    const float *fdata = reinterpret_cast<const float *>(&data[data_start]);
    samples.assign(fdata, fdata + num_samples);
  } else if (bits_per_sample == 32 && audio_format == 1) {
    const int32_t *pcm = reinterpret_cast<const int32_t *>(&data[data_start]);
    for (size_t i = 0; i < num_samples; i++) {
      samples.push_back((float)pcm[i] / 2147483648.0f);
    }
  } else if (bits_per_sample == 8) {
    for (size_t i = 0; i < num_samples; i++) {
      samples.push_back(((float)data[data_start + i] - 128.0f) / 128.0f);
    }
  } else {
    return false;
  }

  return !samples.empty();
}

std::vector<float>
asr_server::convert_to_mono_16k(const std::vector<float> &samples,
                                int sample_rate, int channels) {
  // Mix to mono
  std::vector<float> mono;
  if (channels > 1) {
    mono.resize(samples.size() / channels);
    for (size_t i = 0; i < mono.size(); i++) {
      float sum = 0.0f;
      for (int ch = 0; ch < channels; ch++) {
        sum += samples[i * channels + ch];
      }
      mono[i] = sum / (float)channels;
    }
  } else {
    mono = samples;
  }

  if (sample_rate == 16000)
    return mono;

  // Linear interpolation resampling
  double ratio = 16000.0 / (double)sample_rate;
  size_t new_size = (size_t)((double)mono.size() * ratio);
  std::vector<float> out(new_size);

  for (size_t i = 0; i < new_size; i++) {
    double src_pos = (double)i / ratio;
    size_t idx = (size_t)src_pos;
    double frac = src_pos - (double)idx;
    if (idx + 1 < mono.size()) {
      out[i] = (float)(mono[idx] * (1.0 - frac) + mono[idx + 1] * frac);
    } else {
      out[i] = mono[idx];
    }
  }

  return out;
}

// ============================================================
// Start / Stop
// ============================================================

bool asr_server::start() {
  if (!svr_)
    return false;

  ready_.store(true);
  running_.store(true);

  printf("===============================================\n");
  printf("  Qwen3-ASR Server\n");
  printf("  Model: %s\n", params_.model_path.c_str());
  printf("  Listening: http://%s:%d\n", params_.host.c_str(), params_.port);
  printf("  Parallel slots: %d\n", params_.n_parallel);
  printf("  Threads: %d\n", params_.n_threads);
  printf("===============================================\n");
  printf("  Endpoints:\n");
  printf("  - GET  /v1/health\n");
  printf("  - GET  /v1/models\n");
  printf("  - POST /v1/audio/transcriptions\n");
  printf("===============================================\n");

  if (!svr_->listen(params_.host.c_str(), params_.port)) {
    fprintf(stderr, "Failed to start HTTP server on %s:%d\n",
            params_.host.c_str(), params_.port);
    ready_.store(false);
    running_.store(false);
    return false;
  }
  return true;
}

void asr_server::stop() {
  if (svr_ && running_.load()) {
    svr_->stop();
    running_.store(false);
    ready_.store(false);
  }
}

// ============================================================
// CLI entry point
// ============================================================

static void print_usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  printf("\nOptions:\n");
  printf("  -h, --help            Show this help\n");
  printf("  -m, --model <path>    Path to Qwen3 ASR GGUF model (required)\n");
  printf("  --host <host>         Host to bind (default: 127.0.0.1)\n");
  printf("  --port <port>         Port to listen on (default: 8080)\n");
  printf("  -np, --n-parallel <n> Parallel inference slots (default: 1)\n");
  printf("  -t, --threads <n>     CPU threads (default: 4)\n");
  printf("  --debug               Enable debug output\n");
}

static asr_server_params parse_args(int argc, char **argv) {
  asr_server_params params;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      exit(0);
    } else if ((a == "-m" || a == "--model") && i + 1 < argc) {
      params.model_path = argv[++i];
    } else if (a == "--host" && i + 1 < argc) {
      params.host = argv[++i];
    } else if (a == "--port" && i + 1 < argc) {
      params.port = std::stoi(argv[++i]);
    } else if ((a == "-np" || a == "--n-parallel") && i + 1 < argc) {
      params.n_parallel = std::stoi(argv[++i]);
    } else if ((a == "-t" || a == "--threads") && i + 1 < argc) {
      params.n_threads = std::stoi(argv[++i]);
    } else if (a == "--debug") {
      params.debug_mode = true;
    }
  }
  return params;
}

int main(int argc, char **argv) {
  asr_server_params params = parse_args(argc, argv);

  if (params.model_path.empty()) {
    fprintf(stderr, "Error: --model is required\n");
    print_usage(argv[0]);
    return 1;
  }

  if (params.n_parallel < 1)
    params.n_parallel = 1;
  if (params.n_parallel > 32)
    params.n_parallel = 32;

  asr_server server(params);
  if (!server.init()) {
    fprintf(stderr, "Server init failed\n");
    return 1;
  }

  if (!server.start()) {
    fprintf(stderr, "Server start failed\n");
    return 1;
  }

  return 0;
}
