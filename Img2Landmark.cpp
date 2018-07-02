#include <algorithm>
#include "Img2Landmark.h"

using namespace std;

Img2Landmark::Img2Landmark()
:DeployModel<vector<Point> >("deploy/2DFAN4_deploy_init.pbtxt","deploy/2DFAN4_deploy_predict.pbtxt")
{
}

Img2Landmark::~Img2Landmark()
{
}

vector<Point> Img2Landmark::predict(Mat img)
{
	return DeployModel<vector<Point> >::predict(img,"data","l4");
}

TensorCPU Img2Landmark::preprocess(Mat img)
{
	assert(img.channels() == 3);
	assert(img.rows == 256);
	assert(img.cols == 256);
	vector<TIndex> dims({1,img.channels(),img.rows,img.cols});
	vector<float> data(1 * img.channels() * img.rows * img.cols);
	
	for(int c = 0 ; c < img.channels() ; c++)
		for(int h = 0 ; h < img.rows ; h++)
			for(int w = 0 ; w < img.cols ; w++)
				data[c * img.rows * img.cols + h * img.cols + w] = img.ptr<unsigned char>(h)[w * img.channels() + c];

	return TensorCPU(dims,data,NULL);
}

vector<Point> Img2Landmark::postprocess(TensorCPU output)
{
	const float * data = output.data<float>();
	vector<TIndex> dims = output.dims();
	assert(4 == dims.size());
	assert(1 == dims[0]);
	assert(68 == dims[1]);
	assert(64 == dims[2]);
	assert(64 == dims[3]);
	vector<Point> landmarks;
	for(int c = 0 ; c < 68 ; c++) {
		Mat heatmap(Size(64,64),CV_32F);
		for(int h = 0 ; h < 64 ; h++)
			for(int w = 0 ; w < 64 ; w++)
				heatmap.at<float>(h,w) = data[c * 64 * 64 + h * 64 + w];
		auto maxiter = max_element(heatmap.begin<float>(),heatmap.end<float>());
		int posindex = maxiter - heatmap.begin<float>();
		float h = posindex / 64;
		float w = posindex % 64;
		if(0 < h && h < 63 && 0 < w && w < 63) {
			h = (heatmap.at<float>(h + 1,w) - heatmap.at<float>(h - 1,w) > 0)? 
					h + 0.25 :
					(heatmap.at<float>(h + 1,w) - heatmap.at<float>(h - 1,w) < 0)?
						h - 0.25 : h;
			w = (heatmap.at<float>(h,w + 1) - heatmap.at<float>(h,w - 1) > 0)?
					w + 0.25 :
					(heatmap.at<float>(h,w + 1) - heatmap.at<float>(h,w - 1) < 0)?
						w - 0.25 : w;
		}
		landmarks.push_back(Point(w * 4, h * 4));
	}
	return landmarks;
}
