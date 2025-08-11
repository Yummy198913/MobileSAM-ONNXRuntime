// sam_infer_fixed.cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <codecvt>
#include <locale>
#include<Windows.h>
#include <psapi.h> // GetProcessMemoryInfo
#include<chrono>

using namespace std;

struct Args {
    string backbone_path;
    string model_path;
    string img_path;
    string points;
    string labels;
    string device = "cpu";
};

static bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        string s(argv[i]);
        if (s == "--backbone" && i + 1 < argc) args.backbone_path = argv[++i];
        else if (s == "--model" && i + 1 < argc) args.model_path = argv[++i];
        else if (s == "--img" && i + 1 < argc) args.img_path = argv[++i];
        else if (s == "--points" && i + 1 < argc) args.points = argv[++i];
        else if (s == "--labels" && i + 1 < argc) args.labels = argv[++i];
        else if (s == "--device" && i + 1 < argc) args.device = argv[++i];
        else {
            perror("Unknown or incomplete arg !");
            return false;
        }
    }
    if (args.backbone_path.empty() || args.model_path.empty() || args.img_path.empty()
        || args.points.empty() || args.labels.empty() || args.device.empty()) {
        cerr << "Missing required args!" << endl;
        return false;
    }
    cout << "Backbone path " << args.backbone_path << endl;
    cout << "model path " << args.model_path << endl;
    cout << "img path " << args.img_path << endl;
    cout << "Point prompts " << args.points << endl;
    cout << "labels " << args.labels << endl;
    cout << "device " << args.device << endl;

    return true;
}

static vector<float> parse_floats(const string& s) {
    vector<float> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        if (item.size() == 0) continue;
        out.push_back(stof(item));
    }
    return out;
}

static vector<int> parse_ints(const string& s) {
    vector<int> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        if (item.size() == 0) continue;
        out.push_back(stoi(item));
    }
    return out;
}

/*
process img
*/
static vector<float> preprocess_img(const cv::Mat& bgr, int target_w, int target_h, int& out_h, int& out_w) {
    // means & stds - same as Python
    const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
    const float std_vals[3] = { 58.395f, 57.12f, 57.375f };

    // bgr -> rgb
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3);

    int h = rgb.rows, w = rgb.cols;
    int size = max(h, w); // for padding to square

    // padding to square (bottom/right)
    cv::Mat padded = cv::Mat::zeros(size, size, rgb.type());
    rgb.copyTo(padded(cv::Rect(0, 0, w, h)));

    // resize
    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(target_w, target_h));

    // normalized: FIXED -> (channels - mean) / std
    vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean_vals[c]) / std_vals[c];
    }
    cv::Mat normalized;
    cv::merge(channels, normalized);

    // convert to CHW float vector
    vector<float> out;
    out.resize(1 * 3 * target_h * target_w);
    int plane = target_h * target_w;
    for (int c = 0; c < 3; c++) {
        // ensure continuous memory and correct type
        cv::Mat ch_float;
        if (channels[c].isContinuous() && channels[c].depth() == CV_32F) {
            ch_float = channels[c];
        }
        else {
            channels[c].convertTo(ch_float, CV_32F);
        }
        const float* p = reinterpret_cast<const float*>(ch_float.data);
        for (int i = 0; i < plane; ++i) {
            out[c * plane + i] = p[i];
        }
    }

    out_h = h;
    out_w = w;
    return out;
}

// transform coords to resized img
static void apply_cpprds(vector<float>& coords, int orig_h, int orig_w) {
    if (coords.empty() || coords.size() % 2 != 0) return;
    float new_w = (orig_w > orig_h) ? 1024.0f : 768.0f;
    float new_h = (orig_w > orig_h) ? 768.0f : 1024.0f;

    float scale_x = new_w / float(orig_w);
    float scale_y = new_h / float(orig_h);
    for (size_t i = 0; i < coords.size(); i += 2) {
        coords[i + 0] *= scale_x;
        coords[i + 1] *= scale_y;
    }
}

static Ort::Value create_tensor_from_vector(Ort::MemoryInfo& mem_info,
    float* data_ptr, size_t elem_count,
    const vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(mem_info, data_ptr, elem_count, shape.data(), shape.size());
}

// 获取当前进程 CPU 使用率（只测一次，不是实时）
double getCPUUsage()
{
    static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors;
    static HANDLE self;
    if (lastCPU.QuadPart == 0)
    {
        SYSTEM_INFO sysInfo;
        FILETIME ftime, fsys, fuser;

        GetSystemInfo(&sysInfo);
        numProcessors = sysInfo.dwNumberOfProcessors;

        self = GetCurrentProcess();

        GetSystemTimeAsFileTime(&ftime);
        memcpy(&lastCPU, &ftime, sizeof(FILETIME));

        GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
        memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
        memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));
        return 0.0;
    }

    FILETIME ftime, fsys, fuser;
    ULARGE_INTEGER now, sys, user;
    double percent;

    GetSystemTimeAsFileTime(&ftime);
    memcpy(&now, &ftime, sizeof(FILETIME));

    GetProcessTimes(self, &ftime, &ftime, &fsys, &fuser);
    memcpy(&sys, &fsys, sizeof(FILETIME));
    memcpy(&user, &fuser, sizeof(FILETIME));

    percent = (double)((sys.QuadPart - lastSysCPU.QuadPart) +
        (user.QuadPart - lastUserCPU.QuadPart));
    percent /= (now.QuadPart - lastCPU.QuadPart);
    percent /= numProcessors;

    lastCPU = now;
    lastUserCPU = user;
    lastSysCPU = sys;

    return percent * 100;
}

// 获取当前进程内存使用（MB）
double getMemoryUsageMB()
{
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(),
        (PROCESS_MEMORY_COUNTERS*)&pmc,
        sizeof(pmc));
    SIZE_T physMemUsed = pmc.WorkingSetSize;
    return physMemUsed / (1024.0 * 1024.0);
}

int main(int argc, char** argv) {
    try {
        // 记录 CPU/内存 初始状态
        getCPUUsage(); // 初始化CPU监控

        auto start = std::chrono::high_resolution_clock::now();
        Args args;
        if (!parse_args(argc, argv, args)) return -1;

        // parse points & labels
        vector<float> points = parse_floats(args.points); // x1,y1,x2,y2...
        vector<int> labels = parse_ints(args.labels);
        if (points.size() / 2 != labels.size()) {
            cerr << "point coords count and labels count mismatch\n";
            return -1;
        }

        // load image
        cv::Mat img = cv::imread(args.img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            cerr << "Failed to load image: " << args.img_path << "\n";
            return -1;
        }

        // ORT env & session options
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mobile_sam");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // ---- backbone session ----
        wstring_convert<codecvt_utf8_utf16<wchar_t>> converter;
        wstring w_backbone_path = converter.from_bytes(args.backbone_path);
        Ort::Session backbone_sess(env, w_backbone_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // backbone input info
        size_t backbone_inputs = backbone_sess.GetInputCount();
        if (backbone_inputs < 1) {
            cerr << "backbone onnx has no inputs\n";
            return -1;
        }
        string backbone_inname = backbone_sess.GetInputNameAllocated(0, allocator).get();
        Ort::TypeInfo in_type_info = backbone_sess.GetInputTypeInfo(0);
        auto tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
        vector<int64_t> backbone_input_shape = tensor_info.GetShape(); // [B,C,H,W] or dynamic
        int target_h = 1024, target_w = 1024; // fallback
        if (backbone_input_shape.size() >= 4) {
            if (backbone_input_shape[2] > 0) target_h = (int)backbone_input_shape[2];
            if (backbone_input_shape[3] > 0) target_w = (int)backbone_input_shape[3];
        }
        cout << "Backbone target size: " << target_w << "x" << target_h << endl;

        // preprocess
        int orig_h = 0, orig_w = 0;
        vector<float> inp_tensor = preprocess_img(img, target_w, target_h, orig_h, orig_w);
        vector<int64_t> in_shape = { 1, 3, target_h, target_w };
        size_t in_elems = 1ull * 3 * target_h * target_w;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = create_tensor_from_vector(mem_info, inp_tensor.data(), in_elems, in_shape);

        // prepare backbone output names -> SAFELY copy into strings
        vector<string> backbone_output_names_str;
        size_t backbone_output_count = backbone_sess.GetOutputCount();
        for (size_t i = 0; i < backbone_output_count; ++i) {
            auto alloc_name = backbone_sess.GetOutputNameAllocated(i, allocator);
            backbone_output_names_str.push_back(string(alloc_name.get())); // copy into std::string
        }
        vector<const char*> backbone_output_names_c;
        for (auto& s : backbone_output_names_str) backbone_output_names_c.push_back(s.c_str());

        vector<const char*> backbone_input_names = { backbone_inname.c_str() };

        // run backbone
        auto backbone_outputs = backbone_sess.Run(Ort::RunOptions{ nullptr },
            backbone_input_names.data(), &input_tensor, 1,
            backbone_output_names_c.data(), backbone_output_names_c.size());

        if (backbone_outputs.empty()) {
            cerr << "backbone returned no outputs\n";
            return -1;
        }
        Ort::Value& image_embedding_val = backbone_outputs[0];

        // embedding info
        auto emb_info = image_embedding_val.GetTensorTypeAndShapeInfo();
        vector<int64_t> emb_shape = emb_info.GetShape();
        size_t emb_elems = emb_info.GetElementCount();
        float* emb_data_ptr = image_embedding_val.GetTensorMutableData<float>();

        // ---- mask session ----
        wstring w_model_path = converter.from_bytes(args.model_path);
        Ort::Session mask_sess(env, w_model_path.c_str(), session_options);

        size_t mask_input_count = mask_sess.GetInputCount();
        vector<string> mask_input_names_str;
        for (size_t i = 0; i < mask_input_count; ++i) {
            auto alloc_name = mask_sess.GetInputNameAllocated(i, allocator);
            mask_input_names_str.push_back(string(alloc_name.get())); // safe copy
        }
        cout << "Mask model inputs (" << mask_input_count << "):\n";
        for (auto& n : mask_input_names_str) cout << "  " << n << "\n";

        // expected names (strict)
        vector<string> expected_names = {
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size"
        };

        // build selected_input_names in the order of expected_names
        vector<string> selected_input_names; selected_input_names.reserve(expected_names.size());
        for (const auto& en : expected_names) {
            bool found = false;
            for (const auto& actual : mask_input_names_str) {
                if (actual == en) {
                    selected_input_names.push_back(actual);
                    found = true;
                    break;
                }
            }
            if (!found) {
                cerr << "Required input name '" << en << "' NOT found in mask model inputs.\n";
                cerr << "Model provides these input names:\n";
                for (const auto& n : mask_input_names_str) cerr << "  - " << n << "\n";
                cerr << "Please verify your ONNX model input names or update the expected names.\n";
                return -1;
            }
        }

        // build point coords + labels (append pad point & label -1)
        int K = (int)labels.size();
        vector<float> point_coords; point_coords.reserve((K + 1) * 2);
        for (int i = 0; i < K; ++i) { point_coords.push_back(points[i * 2 + 0]); point_coords.push_back(points[i * 2 + 1]); }
        point_coords.push_back(0.0f); point_coords.push_back(0.0f);
        apply_cpprds(point_coords, orig_h, orig_w);
        vector<int64_t> coords_shape = { 1, (int64_t)K + 1, 2 };

        vector<float> point_labels; point_labels.reserve(K + 1);
        for (int v : labels) point_labels.push_back((float)v);
        point_labels.push_back(-1.0f);
        vector<int64_t> labels_shape = { 1, (int64_t)K + 1 };

        vector<int64_t> mask_input_shape = { 1, 1, 256, 256 };
        size_t mask_input_elems = 1ull * 1 * 256 * 256;
        vector<float> mask_input_data(mask_input_elems, 0.0f);

        vector<int64_t> has_mask_shape = { 1 };
        vector<float> has_mask_data = { 1.0f };

        vector<int64_t> orig_im_shape = { 2 };
        vector<float> orig_im_data = { (float)orig_h, (float)orig_w };

        // copy emb to contiguous vector
        vector<int64_t> emb_shape_vec = emb_shape;
        size_t emb_count = emb_elems;
        vector<float> emb_buffer(emb_count);
        copy(emb_data_ptr, emb_data_ptr + emb_count, emb_buffer.data());

        // create Ort::Value inputs
        Ort::Value emb_tensor = create_tensor_from_vector(mem_info, emb_buffer.data(), emb_count, emb_shape_vec);
        Ort::Value coords_tensor = create_tensor_from_vector(mem_info, point_coords.data(), point_coords.size(), coords_shape);
        Ort::Value labels_tensor = create_tensor_from_vector(mem_info, point_labels.data(), point_labels.size(), labels_shape);
        Ort::Value mask_input_tensor = create_tensor_from_vector(mem_info, mask_input_data.data(), mask_input_elems, mask_input_shape);
        Ort::Value has_mask_tensor = create_tensor_from_vector(mem_info, has_mask_data.data(), has_mask_data.size(), has_mask_shape);
        Ort::Value orig_im_tensor = create_tensor_from_vector(mem_info, orig_im_data.data(), orig_im_data.size(), orig_im_shape);

        // order inputs to match expected_names
        vector<const char*> mask_input_names_c;
        for (const auto& s : selected_input_names) mask_input_names_c.push_back(s.c_str());

        vector<Ort::Value> mask_inputs;
        mask_inputs.reserve(6);
        mask_inputs.push_back(move(emb_tensor));
        mask_inputs.push_back(move(coords_tensor));
        mask_inputs.push_back(move(labels_tensor));
        mask_inputs.push_back(move(mask_input_tensor));
        mask_inputs.push_back(move(has_mask_tensor));
        mask_inputs.push_back(move(orig_im_tensor));

        if (mask_input_names_c.size() != mask_inputs.size()) {
            cerr << "Internal error: input name count != input tensor count\n";
            return -1;
        }

        // get mask model output names safely
        size_t mask_output_count = mask_sess.GetOutputCount();
        vector<string> mask_output_names_str;
        mask_output_names_str.reserve(mask_output_count);
        for (size_t i = 0; i < mask_output_count; ++i) {
            auto alloc_name = mask_sess.GetOutputNameAllocated(i, allocator);
            mask_output_names_str.push_back(string(alloc_name.get()));
        }
        vector<const char*> mask_output_names_c;
        for (auto& s : mask_output_names_str) {
            if (s.empty()) {
                cerr << "Mask model has an empty output name - aborting\n";
                return -1;
            }
            mask_output_names_c.push_back(s.c_str());
        }

        cout << "Mask model outputs:\n";
        for (auto& n : mask_output_names_str) cout << "  " << n << "\n";

        // run mask session
        auto output_tensors = mask_sess.Run(Ort::RunOptions{ nullptr },
            mask_input_names_c.data(),
            mask_inputs.data(),
            mask_inputs.size(),
            mask_output_names_c.data(),
            mask_output_names_c.size());

        if (output_tensors.empty()) {
            cerr << "mask model returned no outputs\n";
            return -1;
        }

        // assume first output is masks
        Ort::Value& m0 = output_tensors[0];
        auto m0_info = m0.GetTensorTypeAndShapeInfo();
        vector<int64_t> m0_shape = m0_info.GetShape();
        size_t m0_count = m0_info.GetElementCount();
        float* m0_data = m0.GetTensorMutableData<float>();

        vector<uint8_t> mask_binary(m0_count);
        for (size_t i = 0; i < m0_count; ++i) mask_binary[i] = (m0_data[i] > 0.0f) ? 255 : 0;

        int out_h = (int)m0_shape[m0_shape.size() - 2];
        int out_w = (int)m0_shape[m0_shape.size() - 1];

        if (orig_w == 0 || orig_h == 0) {
            cerr << "Warning: original image size is zero (orig_h/orig_w). Skipping visualization\n";
            return 0;
        }

        cv::Mat mask_mat(out_h, out_w, CV_8UC1, mask_binary.data());
        cv::Mat mask_resized;
        cv::resize(mask_mat, mask_resized, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_NEAREST);

        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
        cv::Mat overlay;
        img_rgb.copyTo(overlay);
        for (int y = 0; y < orig_h; ++y) {
            for (int x = 0; x < orig_w; ++x) {
                if (mask_resized.at<uint8_t>(y, x) > 0) {
                    overlay.at<cv::Vec3b>(y, x)[0] = (uint8_t)(0.6f * 255 + 0.4f * overlay.at<cv::Vec3b>(y, x)[0]);
                    overlay.at<cv::Vec3b>(y, x)[1] = (uint8_t)(0.4f * overlay.at<cv::Vec3b>(y, x)[1]);
                    overlay.at<cv::Vec3b>(y, x)[2] = (uint8_t)(0.4f * overlay.at<cv::Vec3b>(y, x)[2]);
                }
            }
        }

        cv::cvtColor(overlay, overlay, cv::COLOR_RGB2BGR);
        string out_path = "mobile_sam_cpp_out.png";
        cv::imwrite(out_path, overlay);
        cout << "Saved overlay to " << out_path << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        double cpu_percent = getCPUUsage();
        double mem_mb = getMemoryUsageMB();

        std::cout << "推理耗时: " << elapsed_ms << " ms\n";
        std::cout << "CPU 占用率: " << cpu_percent << " %\n";
        std::cout << "内存占用: " << mem_mb << " MB\n";
    }
    catch (const Ort::Exception& e) {
        cerr << "ONNX Runtime Error: " << e.what() << endl;
        return -1;
    }
    catch (const std::exception& e) {
        cerr << "std::exception: " << e.what() << endl;
        return -1;
    }

    return 0;
}
