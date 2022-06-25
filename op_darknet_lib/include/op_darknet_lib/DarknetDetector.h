/*
 * DarknetDetector.h
 *
 *  Created on: Mar 17, 2021
 *      Author: Hatem Darweesh
 */

#ifndef DARKNETDETECTOR_H_
#define DARKNETDETECTOR_H_

#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "op_planner/MappingHelpers.h"

extern "C"{
#include "darknet/network.h"
#include "darknet/detection_layer.h"
#include "darknet/region_layer.h"
#include "darknet/cost_layer.h"
#include "darknet/utils.h"
#include "darknet/parser.h"
#include "darknet/box.h"
#include "darknet/image.h"
#include "darknet/image_opencv.h"
#include "darknet/demo.h"
#include "darknet/option_list.h"
}

namespace op_darknet_lib_ns
{

enum TL_CLASS_TYPE{
	LEFT_RED = 0,
	RED = 1,
	RIGHT_RED = 2,
	LEFT_GREEN = 3,
	GREEN = 4,
	RIGHT_GREEN = 5,
	YELLOW = 6,
	OFF = 7,
	TL_UNKNOWN = 8,
	};

static PlannerHNS::EnumString<TL_CLASS_TYPE> TL_CLASS_TYPE_STR(TL_UNKNOWN,
		{
				{LEFT_RED, "Left Red"},
				{RED, "Red"},
				{RIGHT_RED, "Right Red"},
				{LEFT_GREEN, "Left Green"},
				{GREEN, "Green"},
				{RIGHT_GREEN, "Right Green"},
				{YELLOW, "Yellow"},
				{OFF, "Off"},
				{TL_UNKNOWN, "Unknown"},
		});

enum MAP_CLASS_TYPE{
	PERSON_CLASS = 0,
	BICYCLE_CLASS = 1,
	CAR_CLASS = 2,
	MOTORBIKE_CLASS = 3,
	BUS_CLASS = 5,
	TRUCK_CLASS = 7,
	TRAFFIC_LIGHT = 9,
	STOP_SIGN = 11,
	STOP_LINE = 101,
	CROSSING = 102,
	FORWARD_ARROW = 103,
	RIGHT_ONLY_ARROW = 104,
	LEFT_ONLY_ARROW = 105,
	RIGHT_FORWARD_ARROW = 106,
	LEFT_FORWARD_ARROW = 107,
	LEFT_RIGHT_ARROW = 108,
	LEFT_RIGHT_FORWARD_ARROW = 109,
	NO_UTURN = 110,
	INTERSECTION_BOX = 111,
	MAP_UNKNOWN = 150,
	};

static PlannerHNS::EnumString<MAP_CLASS_TYPE> MAP_CLASS_TYPE_STR(MAP_UNKNOWN,
{
	{PERSON_CLASS, "person"},
	{BICYCLE_CLASS, "bicycle"},
	{CAR_CLASS, "car"},
	{MOTORBIKE_CLASS, "motorbike"},
	{BUS_CLASS, "bus"},
	{TRUCK_CLASS, "truck"},
	{TRAFFIC_LIGHT, "traffic light"},
	{STOP_SIGN, "stop sign"},
	{STOP_LINE, "stop line"},
	{CROSSING, "crossing"},
	{FORWARD_ARROW, "arrow forward"},
	{RIGHT_ONLY_ARROW, "arrow only right"},
	{LEFT_ONLY_ARROW, "arrow only left"},
	{RIGHT_FORWARD_ARROW, "arrow right"},
	{LEFT_FORWARD_ARROW, "arrow left"},
	{LEFT_RIGHT_ARROW, "arrow left right"},
	{LEFT_RIGHT_FORWARD_ARROW, "arrow all"},
	{NO_UTURN, "no u turn"},
	{INTERSECTION_BOX, "intersection"},
	{MAP_UNKNOWN, "unknown"},
});

template <class T>
class DetectedObjClass
{
public:
	double score;
	cv::Point top_right;
	cv::Point bottom_left;
	std::vector<PlannerHNS::WayPoint> poly_contour;
	double width;
	double height;
	double center_x;
	double center_y;
	unsigned int class_type_id;
	T type;
	std::string class_label;

	DetectedObjClass()
	{
		center_x = 0;
		center_y = 0;
		width = 0;
		height = 0;
		score = 0;
		class_type_id = 0;
	}
};

class DarknetParams
{
public:
	std::string config_file;
	std::string weights_file;
	std::string names_file;
	std::vector<std::string> classes_names;
	double detect_threshold = 0.001;
};

template <class T>
class DarknetDetector
{

public:
	DarknetParams m_params;
	network m_net;
	int m_gCounter = 0;
	timespec m_Timer;

	DarknetDetector()
	{
		UtilityHNS::UtilityH::GetTickCount(m_Timer);
	}
	virtual ~DarknetDetector()
	{
//		if(m_classes_names != nullptr)
//		{
//			free_ptrs((void**)m_classes_names, m_params.classes_names.size());
//		}
	}

	int Init( DarknetParams& _params)
	{
		m_params = _params;

		char* c_name =  new char[m_params.config_file.size()];
		strcpy(c_name, m_params.config_file.c_str());

		char* w_name =  new char[m_params.weights_file.size()];
		strcpy(w_name, m_params.weights_file.c_str());

//		m_classes_names = new char*[m_params.classes_names.size()];
//		for(unsigned int i=0; i < _params.classes_names.size(); i++)
//		{
//			std::string str_name = _params.classes_names.at(i);
//			m_classes_names[i] = new char[str_name.size()];
//			strcpy(m_classes_names[i], str_name.c_str());
//		}

		m_net = parse_network_cfg_custom(c_name, 1, 1);
		load_weights(&m_net, w_name);
		fuse_conv_batchnorm(m_net);
		calculate_binary_weights(m_net);

		delete [] c_name;
		delete [] w_name;

		return 0;
	}

	std::vector<DetectedObjClass<T> > DetectObjects(cv::Mat& src_img);


	image mat_to_image(cv::Mat src_mat)
	{
		cv::Mat dst;
		if (src_mat.channels() == 3)
		{
			cv::cvtColor(src_mat, dst, cv::COLOR_RGB2BGR);
		}
		else if (src_mat.channels() == 4)
		{
			cv::cvtColor(src_mat, dst, cv::COLOR_RGBA2BGRA);
		}
		else
		{
			dst = src_mat;
		}

	    int w = dst.cols;
	    int h = dst.rows;
	    int c = dst.channels();
	    image im = make_image(w, h, c);
	    unsigned char *data = (unsigned char *)dst.data;
	    int step = dst.step;
	    for (int y = 0; y < h; ++y) {
	        for (int k = 0; k < c; ++k) {
	            for (int x = 0; x < w; ++x) {
	                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
	                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
	                //im.data[k*w*h + y*w + x] = val / 255.0f;

	                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
	            }
	        }
	    }
	    return im;
	}
};

}

#endif /* DARKNETDETECTOR_H_ */
