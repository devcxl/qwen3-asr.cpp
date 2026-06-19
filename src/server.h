#ifndef QWEN3_ASR_SERVER_H
#define QWEN3_ASR_SERVER_H

#include <string>
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>

#include "httplib.h"
#include "qwen3_asr.h"

// Server configuration
struct asr_server_params {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_parallel = 1;
    int n_threads = 4;
    bool debug_mode = false;
};

// A single inference slot
struct asr_slot {
    int id = 0;
    std::atomic<bool> busy{false};
    std::atomic<bool> done{false};
    std::string result_text;
    std::string error_message;
    std::vector<float> audio_data;
    int sample_rate = 16000;

    void reset() {
        busy.store(false);
        done.store(false);
        result_text.clear();
        error_message.clear();
        audio_data.clear();
        sample_rate = 16000;
    }
};

// Thread-safe slot pool – slots are independent so parallel requests
// each get their own slot without blocking each other.
class asr_slot_pool {
public:
    asr_slot_pool(int n_slots, qwen3_asr::Qwen3ASR &model);
    ~asr_slot_pool();

    // Returns slot id, or -1 if all slots busy
    int acquire();
    void release(int slot_id);
    asr_slot &get(int slot_id) { return slots_[slot_id]; }
    int size() const { return n_slots_; }

private:
    int n_slots_;
    qwen3_asr::Qwen3ASR &model_;
    std::vector<asr_slot> slots_;
    std::mutex mutex_;
};

class asr_server {
public:
    explicit asr_server(const asr_server_params &params);
    ~asr_server();

    bool init();
    bool start();
    void stop();
    bool is_ready() const { return ready_.load(); }

private:
    void register_routes();

    void handle_health   (const httplib::Request &, httplib::Response &);
    void handle_transcriptions(const httplib::Request &, httplib::Response &);
    void handle_models   (const httplib::Request &, httplib::Response &);

    // Audio conversion helpers
    bool parse_wav_from_memory(const std::vector<uint8_t> &data,
                               int &sample_rate, int &channels,
                               std::vector<float> &samples);
    std::vector<float> convert_to_mono_16k(const std::vector<float> &samples,
                                            int sample_rate, int channels);

    // Build JSON string helpers (avoid nlohmann/json dependency)
    static std::string json_string(const std::string &s);
    static std::string json_pair(const std::string &key, const std::string &val);
    static std::string json_pair(const std::string &key, long val);
    static std::string json_pair(const std::string &key, double val, int precision = 4);
    static std::string json_object(const std::vector<std::string> &pairs);
    static std::string json_array(const std::vector<std::string> &elements);

    asr_server_params params_;
    std::unique_ptr<httplib::Server> svr_;
    std::unique_ptr<qwen3_asr::Qwen3ASR> model_;
    std::unique_ptr<asr_slot_pool> slot_pool_;
    std::atomic<bool> ready_{false};
    std::atomic<bool> running_{false};
};

#endif // QWEN3_ASR_SERVER_H
