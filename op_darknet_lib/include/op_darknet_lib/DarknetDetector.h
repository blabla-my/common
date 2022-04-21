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
	MAP_UNKNOWN = 0,
	PERSON_CLASS = 1,
	BICYCLE_CLASS = 2,
	CAR_CLASS = 3,
	MOTORBIKE_CLASS = 4,
	BUS_CLASS = 6,
	TRUCK_CLASS = 8,
	TRAFFIC_LIGHT = 10,
	STOP_SIGN = 12,
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
	};

static PlannerHNS::EnumString<MAP_CLASS_TYPE> MAP_CLASS_TYPE_STR(MAP_UNKNOWN,
		{
				{PERSON_CLASS, "Person"},
				{BICYCLE_CLASS, "Bicycle"},
				{CAR_CLASS, "Car"},
				{MOTORBIKE_CLASS, "Motorbike"},
				{BUS_CLASS, "Bus"},
				{TRUCK_CLASS, "Truck"},
				{TRAFFIC_LIGHT, "Traffic Light"},
				{STOP_SIGN, "Stop Sign"},
				{STOP_LINE, "Stop Line"},
				{CROSSING, "Crossing"},
				{FORWARD_ARROW, "Arrow Forward"},
				{RIGHT_ONLY_ARROW, "Arrow Only Right"},
				{LEFT_ONLY_ARROW, "Arrow Only Left"},
				{RIGHT_FORWARD_ARROW, "Arrow Right"},
				{LEFT_FORWARD_ARROW, "Arrow Left"},
				{LEFT_RIGHT_ARROW, "Arrow Left Right"},
				{LEFT_RIGHT_FORWARD_ARROW, "Arrow All"},
				{NO_UTURN, "No U Turn"},
				{INTERSECTION_BOX, "Intersection"},
				{MAP_UNKNOWN, "Unknown"},
		});

template <class T>
class DetectedObjClass
{
public:
	double score;
	cv::Point top_right;
	cv::Point bottom_left;
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

	double detect_threshold;

	DarknetParams()
	{
		detect_threshold = 0.5;
	}
};

template <class T>
class DarknetDetector
{

public:
	DarknetParams m_params;
	network m_net;

	DarknetDetector()
	{
		UtilityHNS::UtilityH::GetTickCount(m_Timer);
	}
	virtual ~DarknetDetector() {}

	int Init( DarknetParams& _params)
	{
		m_params = _params;

		char* c_name =  new char[m_params.config_file.size()];
		strcpy(c_name, m_params.config_file.c_str());

		char* w_name =  new char[m_params.weights_file.size()];
		strcpy(w_name, m_params.weights_file.c_str());

		m_net = parse_network_cfg_custom(c_name, 1, 1);
		load_weights(&m_net, w_name);
		fuse_conv_batchnorm(m_net);
		calculate_binary_weights(m_net);

		delete [] c_name;
		delete [] w_name;

		return 0;
	}

	std::vector<DetectedObjClass<T> > DetectObjects(IplImage& src_img);


	image ipl_to_image(IplImage* src)
	{
		 image out = make_image(src->width, src->height, m_net.c);

		unsigned char *data = (unsigned char *)src->imageData;
		int h = src->height;
		int w = src->width;
		int c = src->nChannels;
		int step = src->widthStep;
		int i, j, k;

		for(i = 0; i < h; ++i){
			for(k= 0; k < c; ++k){
				for(j = 0; j < w; ++j){
					out.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
				}
			}
		}

		return out;
	}
	int m_gCounter = 0;
	timespec m_Timer;
	const int m_nClasses = 8;
};

}

#endif /* DARKNETDETECTOR_H_ */
