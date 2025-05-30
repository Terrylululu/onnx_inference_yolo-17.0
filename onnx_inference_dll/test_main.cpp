#include<windows.h>
#include"yolov5.h"
#include"dirent.h"


int read_files_in_dir2(const char* p_dir_name, std::vector<std::string>& file_names) {
	DIR* p_dir = opendir(p_dir_name);
	if (p_dir == nullptr) {
		return -1;
	}

	struct dirent* p_file = nullptr;
	while ((p_file = readdir(p_dir)) != nullptr) {
		if (strcmp(p_file->d_name, ".") != 0 &&
			strcmp(p_file->d_name, "..") != 0) {

			// Ö»Ñ°ÕÒ jpg, bmp, png ¸ñÊ½µÄÎÄ¼þ
			std::string file_name(p_file->d_name);
			std::string extension = file_name.substr(file_name.find_last_of('.') + 1);
			if (extension == "jpg" || extension == "jpeg" || extension == "bmp" || extension == "png") {
				/*std::string cur_file_name(p_dir_name);
				cur_file_name += "/";
				cur_file_name += file_name;*/
				std::string cur_file_name(p_file->d_name);
				file_names.push_back(cur_file_name);
			}
		}
	}

	closedir(p_dir);
	return 0;
}
void gen_train_voc2007_img()
{
	int size = (pow((640 / 32), 2) + pow((640 / 16), 2) + pow((640 / 8), 2)) * 3;//640Í¼Ïñ³ß´ç
	int OUTPUT_SIZE = size * (5 + 20);
	Net_config yolo_nets = { 0.25, 0.45, 0.45 ,OUTPUT_SIZE,"VOC2007/weights/best.onnx", "VOC2007/images/voc.names",true};
	yolov5 yolo_model(yolo_nets);
	
	std::vector<string> files;
	string rootPath = "VOC2007/images/";
	string dstPath = "VOC2007/OutputImage/";
	read_files_in_dir2(rootPath.c_str(), files);

	int j = 0;
	for (string file : files)
	{
		std::cout << file << std::endl;
		Mat img = cv::imread(rootPath + file);
		Mat colorImg = img.clone();
		TimerC time;
		yolo_model.detect(img);
		time.out("yolo:");
		int k = 0;
		//if (size(boxs) != 0)
		{
			imwrite(dstPath + file, img);
		}

		j++;
	}
}
void gen_train_luowen_img()
{

	int size = (pow((640 / 32), 2) + pow((640 / 16), 2) + pow((640 / 8), 2)) * 3;//640Í¼Ïñ³ß´ç
	int OUTPUT_SIZE = size * (5 + 6);
	Net_config yolo_nets = { 0.25, 0.45, 0.45 ,OUTPUT_SIZE,"LuoWen/weights/best.onnx", "LuoWen/images/voc.names" };
	yolov5 yolo_model(yolo_nets);

	std::vector<string> files;
	string rootPath = "LuoWen/images/";
	string dstPath = "LuoWen/OutputImage/";
	read_files_in_dir2(rootPath.c_str(), files);

	int j = 0;
	for (string file : files)
	{
		std::cout << file << std::endl;
		Mat img = cv::imread(rootPath + file);
		Mat colorImg = img.clone();
		TimerC time;
		yolo_model.detect(img);
		time.out("yolo:");
		int k = 0;
		//if (size(boxs) != 0)
		{
			imwrite(dstPath + file, img);
		}

		j++;
	}
}
int main()
{
	//gen_train_luowen_img();
	 gen_train_voc2007_img();
	
}