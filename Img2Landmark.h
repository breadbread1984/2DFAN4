#ifndef IMG2LANDMARK_H
#define IMG2LANDMARK_H

#include <vector>
#include "DeployModel.hpp"

using namespace std;

class Img2Landmark : public DeployModel<vector<Point> > {
public:
	Img2Landmark();
	virtual ~Img2Landmark();
	vector<Point> predict(Mat img);
protected:
	virtual TensorCPU preprocess(Mat img);
	virtual vector<Point> postprocess(TensorCPU output);
};

#endif
