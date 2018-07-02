#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include "Img2Landmark.h"
#include "matrix_basic.hpp"

using namespace std;
using namespace boost::program_options;
namespace ublas = boost::numeric::ublas;
using namespace cv;

Mat crop(Mat img,Rect bounding,Size size);
vector<Point2f> project(vector<Point> landmarks,Size sz);
vector<Point2f> reproject(vector<Point2f> landmarks,Rect area);

int main(int argc,char ** argv)
{
	options_description desc;
	string input;
	desc.add_options()
		("help","print this message")
		("input,i",value<string>(&input),"input image");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(1 == vm.count("input")) {
		Mat img = imread(input);
		if(img.empty()) {
			cout<<"invalid image"<<endl;
			return EXIT_FAILURE;
		}
		
		dlib::frontal_face_detector frontaldetector = dlib::get_frontal_face_detector();
		dlib::cv_image<dlib::bgr_pixel> cimg(img);
		vector<dlib::rectangle> faces = frontaldetector(cimg);
		Mat show = img.clone();
		Img2Landmark landmarker;
		namedWindow("debug",false);
		for(auto & face : faces) {
			Rect faceRect(Point(face.left(),face.top()),Point(face.right() + 1, face.bottom() + 1));
			Point center(faceRect.x + faceRect.width / 2,faceRect.y + faceRect.height / 2);
	//		int length = 1 * fmax(faceRect.width,faceRect.height);
			int length = 1.2 * fmax(faceRect.width,faceRect.height);
	//		int length = 1.3 * fmax(faceRect.width,faceRect.height);
	//		int length = 1.4 * fmax(faceRect.width,faceRect.height);
	//		int length = 1.7 * fmax(faceRect.width,faceRect.height);
			faceRect.x = center.x - length / 2;
			faceRect.y = center.y - length / 2;
			faceRect.width = length;
			faceRect.height = length;
			Mat faceImg = crop(img,faceRect,Size(length,length));
			Mat faceImg_rz;
			resize(faceImg,faceImg_rz,Size(256,256));
			vector<Point> landmarks = landmarker.predict(faceImg_rz);
			vector<Point2f> landmarks_proj = project(landmarks,Size(256,256));
			vector<Point2f> landmarks_reproj = reproject(landmarks_proj,faceRect);
#ifndef NDEBUG
			assert(landmarks.size() == 68);
#endif
			for(auto & pts : landmarks_reproj)
				circle(show,pts,img.cols / 200,Scalar(0,255,0),-1);
		}
		imshow("debug",show);
		waitKey();
	} else {
		VideoCapture vc(CV_CAP_ANY);
		if(false == vc.isOpened()) {
			cout<<"can't open any webcam"<<endl;
			return EXIT_FAILURE;
		}
		namedWindow("debug",false);
		Mat img;
		dlib::frontal_face_detector frontaldetector = dlib::get_frontal_face_detector();
		Img2Landmark landmarker;
		while(vc.read(img)) {
			dlib::cv_image<dlib::bgr_pixel> cimg(img);
			vector<dlib::rectangle> faces = frontaldetector(cimg);
			Mat show = img.clone();
			for(auto & face : faces) {
				Rect faceRect(Point(face.left(),face.top()),Point(face.right() + 1, face.bottom() + 1));
				Point center(faceRect.x + faceRect.width / 2,faceRect.y + faceRect.height / 2);
				int length = 1.4 * fmax(faceRect.width,faceRect.height);
				faceRect.x = center.x - length / 2;
				faceRect.y = center.y - length / 2;
				faceRect.width = length;
				faceRect.height = length;
				//faceRect is the detected facial area
				Mat faceImg = crop(img,faceRect,Size(length,length));
				Mat faceImg_rz;
				resize(faceImg,faceImg_rz,Size(256,256));
				vector<Point> landmarks = landmarker.predict(faceImg_rz);
				vector<Point2f> landmarks_proj = project(landmarks,Size(256,256));
				vector<Point2f> landmarks_reproj = reproject(landmarks_proj,faceRect);
				for(auto & pts : landmarks_reproj)
					circle(show,pts,2,Scalar(0,255,0),-1);
			}
			imshow("debug",show);
			char k = waitKey(10);
			if(k == 'q') break;
		}
	}
	
	return EXIT_SUCCESS;
}

vector<Point2f> project(vector<Point> landmarks,Size sz)
{
	Point2f center(sz.width / 2,sz.height / 2);
	vector<Point2f> retVal;
	for(auto & landmark : landmarks) {
		float x = static_cast<float>(landmark.x - center.x) / sz.width;
		float y = static_cast<float>(landmark.y - center.y) / sz.height;
		retVal.push_back(Point2f(x,y));
	}
	return retVal;
}

vector<Point2f> reproject(vector<Point2f> landmarks,Rect area)
{
	Point2f center(area.x + area.width / 2, area.y + area.height / 2);
	vector<Point2f> retVal;
	for(auto & landmark : landmarks) {
		float x = landmark.x * area.width + center.x;
		float y = landmark.y * area.height + center.y;
		retVal.push_back(Point2f(x,y));
	}
	return retVal;
}

Mat crop(Mat img,Rect bounding,Size size)
{
	ublas::matrix<float> A(3,4),B(3,4);
	A(0,0) = bounding.x;	A(0,1) = bounding.x;										A(0,2) = bounding.x + bounding.width;	A(0,3) = bounding.x + bounding.width;
	A(1,0) = bounding.y;	A(1,1) = bounding.y + bounding.height;	A(1,2) = bounding.y + bounding.height;	A(1,3) = bounding.y;
	A(2,0) = 1;					A(2,1) = 1;														A(2,2) = 1;														A(2,3) = 1;
	B(0,0) = 0;	B(0,1) = 0;					B(0,2) = size.width;	B(0,3) = size.width;
	B(1,0) = 0;	B(1,1) = size.height;	B(1,2) = size.height;	B(1,3) = 0;
	B(2,0) = 1;	B(2,1) = 1;					B(2,2) = 1;					B(2,2) = 1;
	ublas::matrix<float> AAt = prod(A,trans(A));
	ublas::matrix<float> ABt = prod(A,trans(B));
	ublas::matrix<float> AAt_inv;
	svd_inv(AAt,AAt_inv);
	ublas::matrix<float> tmp = prod(AAt_inv,ABt);
	tmp = trans(tmp);
	Mat affine(Size(3,2),CV_32FC1,tmp.data().begin());
	Mat patch;
	warpAffine(img, patch, affine, size);
	return patch;
}
