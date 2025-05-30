#include "yolov5.h"
yolov5::yolov5(Net_config config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->Outsize = config.outsize;

	string model_path = config.model_path;
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());


	////获得支持的执行提供者列表
	std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
	auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
	//创建CUDA提供者选项
	OrtCUDAProviderOptions cudaOption{};
	//判断是否使用GPU，并检查是否支持CUDA
	if (config.IsUseCUDA && (cudaAvailable == availableProviders.end()))
	{
		std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
		std::cout << "Inference device: CPU" << std::endl;
	}
	else if (config.IsUseCUDA && (cudaAvailable != availableProviders.end()))
	{
		std::cout << "Inference device: GPU" << std::endl;
		//添加CUDA执行提供者
		sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
	}
	else
	{
		std::cout << "Inference device: CPU" << std::endl;
	}



	/***优化水平***/
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		auto inputname = ort_session->GetInputNameAllocated(i, allocator).get();
		input_names.push_back(inputname);
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		auto outputname = ort_session->GetOutputNameAllocated(i, allocator).get();
		output_names.push_back(outputname);
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}


	// 推理的run 函数只认这种数据
	 InputNames = { input_names[0].c_str() };
	OutNames = { output_names[0].c_str() };


	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];
	string classesFile = config.classesFile;
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->class_names.push_back(line);
	this->num_class = class_names.size();
}

void yolov5::normalize_(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	
	int i = 0;
	for (int row = 0; row < 640; ++row) {
		uchar* uc_pixel = img.data + row * img.step;
		for (int col = 0; col < 640; ++col) {
			//bgr格式数据 归一化
			this->input_image_[i] = (float)uc_pixel[2] / 255.0;
			this->input_image_[i + 640 * 640] = (float)uc_pixel[1] / 255.0;
			this->input_image_[i + 2 * 640 * 640] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
}

cv::Mat yolov5:: preprocess_img(cv::Mat& img, int input_w, int input_h) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	return out;
}
void scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
	float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
		(float)imageShape.width / (float)imageOriginalShape.width);

	int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
				  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

	coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
	coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

	coords.width = (int)std::round(((float)coords.width / gain));
	coords.height = (int)std::round(((float)coords.height / gain));

	// // clip coords, should be modified for width and height
	// coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
	// coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
	// coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
	// coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}
void yolov5::detect(Mat& frame)
{
	Mat dstimg = this->preprocess_img(frame, this->inpHeight, this->inpWidth);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	

	
	vector<Value> ort_outputs;
	try {
		 ort_outputs = ort_session->Run(RunOptions{ nullptr }, InputNames.data(), &input_tensor_, 1, OutNames.data(), output_names.size());// 开始推理
	}
	catch (const Ort::Exception& e) {
		std::cerr << "Error during inference: " << e.what() << std::endl;
	}


	
	float* preds = ort_outputs[0].GetTensorMutableData<float>();

	/////generate proposals
	const int nout = this->num_class + 5;
	
	std::vector<cv::Rect> boxes;
	std::vector<float> confs;
	std::vector<int> classIds;
	
	for (int i = 0; i < this->Outsize; i += nout) {

		float* pp = preds + i;
		float box_score = pp[4];
		if (box_score > this->objThreshold)
		{
			float class_score = 0;
			int class_ind = 0;
			for (int k = 0; k < this->num_class; k++)
			{
				if (pp[k + 5] > class_score)
				{
					class_score = pp[k + 5];
					class_ind = k;
				}
			}
			float confidence = class_score * box_score;
			float cx = pp[0];  ///cx
			float cy = pp[1];   ///cy
			float w = pp[2];   ///w
			float h = pp[3];  ///h

			float xmin = pp[0] - w / 2;
			float ymin = pp[1] - h / 2;

			boxes.emplace_back(cv::Rect(xmin, ymin, w, h));
			confs.emplace_back(confidence);
			classIds.emplace_back(class_ind);

		}
		
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->nmsThreshold, indices);

	for (int idx : indices)
	{
		cv::Rect box = cv::Rect(boxes[idx]);
		scaleCoords(cv::Size(640, 640), box, cv::Size(frame.cols, frame.rows));
		rectangle(frame, box, Scalar(0, 0, 255), 2);
		string label = format("%.2f", confs[idx]);
		label = this->class_names[classIds[idx]] + ":" + label;
		putText(frame, label, Point(box.x,  box.y- 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

	}

}
