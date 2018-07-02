#ifndef DEPLOYMODEL_H
#define DEPLOYMODEL_H

#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/core/module.h>

using namespace std;
using namespace std::chrono;
using namespace cv;
using namespace caffe2;

template<typename T>
class DeployModel {
public:
	DeployModel(string init,string predict);
	virtual ~DeployModel() = 0;
	T predict(Mat img,string inputName = "data",string outputName = "output");
protected:
	virtual TensorCPU preprocess(Mat img) = 0;
	virtual T postprocess(TensorCPU output) = 0;
	
	Workspace workspace;
	unique_ptr<NetBase> predict_net;
};

template<typename T>
DeployModel<T>::DeployModel(string initpb,string predictpb)
:workspace(nullptr)
{
	LoadModule("","libcaffe2_detectron_ops_gpu.so");
	NetDef init,predict;
	CAFFE_ENFORCE(ReadProtoFromFile(initpb,&init));
	CAFFE_ENFORCE(ReadProtoFromFile(predictpb,&predict));
	init.mutable_device_option()->set_device_type(CUDA);
	predict.mutable_device_option()->set_device_type(CUDA);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
}

template<typename T>
DeployModel<T>::~DeployModel()
{
}

template<typename T>
T DeployModel<T>::predict(Mat img,string inputName,string outputName)
{
	TensorCUDA input = TensorCUDA(preprocess(img));
	auto tensor = workspace.CreateBlob(inputName)->GetMutable<TensorCUDA>();
	tensor->ResizeLike(input);
	tensor->ShareData(input);
#ifndef NDEBUG
	time_point<system_clock> start_time = system_clock::now();
#endif
	predict_net->Run();
#ifndef NDEBUG
	nanoseconds elapse = system_clock::now() - start_time;
	cout<<static_cast<float>(elapse.count())/1e9<<"s"<<endl;
#endif
	TensorCPU output = TensorCPU(workspace.GetBlob(outputName)->Get<TensorCUDA>());
	return postprocess(output);
}

template<typename T>
TensorCPU DeployModel<T>::preprocess(Mat img)
{
}

template<typename T>
T DeployModel<T>::postprocess(TensorCPU output)
{
}

#endif
