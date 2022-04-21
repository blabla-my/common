/*
 * DarknetDetector.cpp
 *
 *  Created on: Mar 17, 2021
 *      Author: Hatem Darweesh
 */

#include "op_darknet_lib/DarknetDetector.h"
#include "op_utility/UtilityH.h"

namespace op_darknet_lib_ns
{

// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr) {
    const detection_with_class* a = (detection_with_class*)a_ptr;
    const detection_with_class* b = (detection_with_class*)b_ptr;
    const float delta = (a->det.bbox.x - a->det.bbox.w/2) - (b->det.bbox.x - b->det.bbox.w/2);
    return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}

template <class T>
std::vector<DetectedObjClass<T> > DarknetDetector<T>::DetectObjects(IplImage& src_img)
{
	image img = ipl_to_image(&src_img);
	exposure_image(img, 0.75);
	image img_resized = resize_image(img, m_net.w, m_net.h);

	layer l = m_net.layers[m_net.n-1];
	int k;
	for (k = 0; k < m_net.n; ++k)
	{
		layer lk = m_net.layers[k];
		if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION)
		{
			l = lk;
		}
	}

	float *X = img_resized.data;
	network_predict(m_net, X);
	int nboxes = 0;
	detection *dets = get_network_boxes(&m_net, img.w, img.h, m_params.detect_threshold, 0.5, 0, 1, &nboxes, 0);

	do_nms_sort(dets, nboxes, l.classes, 0.45);

	int selected_detections_num;
	detection_with_class* selected_detections = get_actual_detections(dets, nboxes, m_params.detect_threshold, &selected_detections_num);
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);

	std::vector<DetectedObjClass<T> > detections;
	for (unsigned i = 0; i < selected_detections_num; ++i)
	{
		const int best_class = selected_detections[i].best_class;
		DetectedObjClass<T> _c;
		_c.score = selected_detections[i].det.prob[best_class];
		_c.class_type_id = best_class;
//		_c.type = TL_CLASS_TYPE_STR.GetEnum(_c.class_type_id);
		_c.bottom_left.x = (selected_detections[i].det.bbox.x-(selected_detections[i].det.bbox.w/2.0))*(double)img.w;
		_c.bottom_left.y = (selected_detections[i].det.bbox.y+(selected_detections[i].det.bbox.h/2.0))*(double)img.h;
		_c.top_right.x = (selected_detections[i].det.bbox.x+(selected_detections[i].det.bbox.w/2.0))*(double)img.w;
		_c.top_right.y = (selected_detections[i].det.bbox.y-(selected_detections[i].det.bbox.h/2.0))*(double)img.h;
		_c.width = selected_detections[i].det.bbox.w * (double)img.w;
		_c.height = selected_detections[i].det.bbox.h * (double)img.h;
		_c.center_x = _c.bottom_left.x;
		_c.center_y = _c.top_right.y;
		//std::cout << m_classes_names[best_class] << ": " << selected_detections[i].det.prob[best_class] * 100 << "% , (" << _c.width << ", " << _c.height << ")" <<  std::endl;
		if(_c.width > 5 && _c.height > 7 && _c.bottom_left.y < (img.h - 100))
		{
			detections.push_back(_c);
		}
	}

	free_detections(dets, nboxes);
	free_image(img);
	free_image(img_resized);
	return detections;
}

template std::vector<DetectedObjClass<TL_CLASS_TYPE> > DarknetDetector<TL_CLASS_TYPE>::DetectObjects(IplImage& src_img);
template std::vector<DetectedObjClass<MAP_CLASS_TYPE> > DarknetDetector<MAP_CLASS_TYPE>::DetectObjects(IplImage& src_img);

}

