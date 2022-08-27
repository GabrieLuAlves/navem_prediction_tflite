#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// #include "cv.h"
// #include "cxcore.h"
// #include "highgui.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/optional_debug_tools.h"


int main(int argc, char *argv[])
{
	int foo;

	cv::Mat normalizedImage;
	cv::Mat resizedImage;
	cv::Mat grayScaleImage;

	int width = 200;
	int height = 200;

	cv::Mat m = cv::imread(argv[1], 1);
	char model_x_path[] = "models/dronet_model_x.tflite";
	char model_y_path[] = "models/dronet_model_y.tflite";

	if (m.empty()) {
		std::cout << "Image not loaded successfully" << std::endl;
	}
	else {
		std::cout << "Image loaded successfully" << std::endl;
	}

	// Resize image 
	cv::resize(m, resizedImage, cv::Size(width, height), cv::INTER_LINEAR);

	// Convert image to gray scale
	cv::cvtColor(resizedImage, grayScaleImage, CV_RGB2GRAY);

	// Normalize image	
	grayScaleImage.convertTo(normalizedImage, CV_32F, 1.0 / 255, 0);

	if (normalizedImage.empty()) {
		std::cout << "Image convertion failed" << std::endl;
	} 
	else {
		std::cout << "Image converted successfully" << std::endl;
	}


	std::unique_ptr<tflite::FlatBufferModel> model_x =
	tflite::FlatBufferModel::BuildFromFile(model_x_path);

	std::unique_ptr<tflite::FlatBufferModel> model_y =
	tflite::FlatBufferModel::BuildFromFile(model_y_path);

	if (!model_x) {
		std::cerr << "Failed to load model X" << std::endl;
	}
	else {
		std::cout << "Model X loaded sucessfully" << std::endl;
	}

	if (!model_y) {
		std::cerr << "Failed to load model Y" << std::endl;
	}
	else {
		std::cout << "Model Y loaded sucessfully" << std::endl;
	}

	tflite::ops::builtin::BuiltinOpResolver resolver_x;
	tflite::ops::builtin::BuiltinOpResolver resolver_y;

	tflite::InterpreterBuilder interpreterBuilder_x(*model_x, resolver_x);
	tflite::InterpreterBuilder interpreterBuilder_y(*model_y, resolver_y);

	std::unique_ptr<tflite::Interpreter> interpreter;

	interpreterBuilder_y(&interpreter);

	int input = interpreter->inputs()[0];
	auto input_height = interpreter->tensor(input)->dims->data[1];
	auto input_width = interpreter->tensor(input)->dims->data[2];
	auto input_channels = interpreter->tensor(input)->dims->data[3];

	std::cout << std::endl;

	std::cout << std::left << std::setfill(' ') << std::setw(20) << "Input Height";
	std::cout << std::left << std::setfill(' ') << std::setw(20) << "Input width";
	std::cout << std::left << std::setfill(' ') << std::setw(20) << "Input channels";
	std::cout << std::endl;

	std::cout << std::left << std::setfill(' ') << std::setw(20) << input_height;
	std::cout << std::left << std::setfill(' ') << std::setw(20) << input_width;
	std::cout << std::left << std::setfill(' ') << std::setw(20) << input_channels;
	std::cout << std::endl << std::endl;

	interpreter->AllocateTensors();

	int k = 0;
	for (int i = 0 ; i < input_height ; i++) {
		for(int j = 0 ; j < input_width ; j++) {
			interpreter->typed_input_tensor<float>(0)[k++] = normalizedImage.at<float>(i, j);
		}
	}

	interpreter->Invoke();

	int output = interpreter->outputs()[0];
	TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
	auto output_size = output_dims->data[output_dims->size - 1];

	std::cout << std::endl;
	std::cout << "Output size: " << output_size << std::endl;

	// std::vector<std::pair<float, int>> top_results;

	switch(interpreter->tensor(output)->type) {
		case kTfLiteFloat32:
			std::cout << "Output type is kTfLiteFloat32" << std::endl;
			break;
		case kTfLiteUInt8:
			std::cout << "Output type is kTfLiteUint8" << std::endl;
			break;
		case kTfLiteInt32:
			std::cout << "Output type is kTfLiteInt32" << std::endl;
			break;
		default:
			std::cerr << "Output type is unknown" << std::endl;
	}
	std::cout << std::endl;

	auto tensor = interpreter->typed_output_tensor<float>(0);

	int index = 0;
	float max = 0.0;
	for(int i = 0 ; i < output_size ; i++) {
		if(tensor[i] > max) {
			max = tensor[i];
			index = i;
		}
		std::cout << "[" << i << "] :" << tensor[i] << std::endl;
	}

	std::cout << "Class Predict: " << index << std::endl;
	std::cout << "Value: " << max << std::endl;

	//std::cout << std::endl;
	//std::cout << "Found " << top_results.size() << " Results" << std::endl;
	//std::cout << std::left << std::setfill(' ') << std::setw(10) << "Label";
	//std::cout << std::left << std::setfill(' ') << std::setw(10) << "Likelihood";
	//std::cout << std::endl;

	//for(int i = 0 ; i < top_results.size() ; i++) {
	//	std::pair<float, int> pair = top_results.at(i);
	//	std::cout << std::left << std::setfill(' ') << std::setw(10) << std::get<1>(pair);
	//	std::cout << std::left << std::setfill(' ') << std::setw(10) << std::get<0>(pair);
	//	std::cout << std::endl;
	//}


	return 0;
}
