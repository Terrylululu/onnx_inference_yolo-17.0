#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	int outsize;
	string model_path;
	string classesFile;
	bool IsUseCUDA;
};
class TimerC
{
public:
	TimerC() : beg_(std::chrono::system_clock::now()) {}
	void reset() { beg_ = std::chrono::system_clock::now(); }

	void out(std::string message = "") {
		auto end = std::chrono::system_clock::now();
		std::cout << message << std::chrono::duration_cast<std::chrono::milliseconds>(end - beg_).count() << "ms" << std::endl;
		reset();
	}
private:
	typedef std::chrono::high_resolution_clock clock_;
	typedef std::chrono::duration<double, std::ratio<1> > second_;
	chrono::time_point<std::chrono::system_clock> beg_;
};

class yolov5
{
public:
	yolov5(Net_config config);
	void detect(Mat& frame);
private:

	int inpWidth;
	int inpHeight;
	vector<string> class_names;
	int num_class;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	int Outsize;
	cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
	vector<float> input_image_;
	void normalize_(Mat img);

	const bool keep_ratio = true;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-lite");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<std::string> input_names;
	vector<std::string> output_names;

	std::array<const char*, 1> InputNames;
	std::array<const char*, 1> OutNames;
	
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

