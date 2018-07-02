#include <cstdlib>
#include <iostream>
#include <dlfcn.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <caffe2/core/init.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/core/module.h>
#include <caffe2/core/operator.h>
#include <caffe2/util/blob.h>
#include <caffe2/util/model.h>
#include <caffe2/util/net.h>
#include <cvplot/cvplot.h>
#include <opencv2/opencv.hpp>

#define NDEBUG
#define TRAINSIZE 612250
#define BATCHSIZE 10

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace boost::random;
using namespace boost::program_options;
using namespace boost::math::constants;
namespace ublas = boost::numeric::ublas;
using namespace cv;
using namespace caffe2;
using namespace cvplot;

void setupTrainNet(NetDef & init, NetDef & predict,int module_num = 4,bool test = false);
void setupSaveNet(NetDef & init, NetDef & save);
void setupLoadNet(NetDef & init, NetDef & load,string modelpath);
void makeList(path dir,vector<pair<string,ublas::matrix<float> > >& filelist);
pair<TensorCUDA,TensorCUDA> fillTrainBlob(vector<pair<string,ublas::matrix<float> > > filelist,string inputname,string outputname,int batchsize,int start_sample = -1);

unique_ptr<NetBase> predict_net;
unique_ptr<NetBase> save_net;
unique_ptr<NetBase> load_net;

void atexit_handler()
{
	cout<<"saving params"<<endl;
	remove_all("ResNetDepth_params");
	save_net->Run();
}

int main(int argc,char ** argv)
{
	options_description desc;
	string inputdir,modeldir;
	desc.add_options()
		("help,h","print this message")
		("input,i",value<string>(&inputdir),"directory containing training samples")
		("deploy,d","generate deploy model only without training")
		("load,l",value<string>(&modeldir),"load trained model and finetune on this one");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	if(0 != vm.count("deploy")) {
		cout<<"generating deploy model"<<endl;
		NetDef init,predict;
		setupTrainNet(init,predict,4,true);
		return EXIT_SUCCESS;
	}
	
	if(1 != vm.count("input")) {
		cout<<"can only process one input directory"<<endl;
		return EXIT_FAILURE;
	}
	
	if(false == exists(inputdir) || false == is_directory(inputdir)) {
		cout<<"invalid training sample directory"<<endl;
		return EXIT_FAILURE;
	}
	
	if(0 != vm.count("load") && (false == exists(modeldir) || false == is_directory(modeldir))) {
		cout<<"model loading directory is invalid"<<endl;
		return EXIT_FAILURE;
	}
	
	NetDef init,predict,save,load;
	setupTrainNet(init,predict,4,false);
	setupSaveNet(init,save);
	if(0 != vm.count("load")) setupLoadNet(init,load,modeldir);
	init.mutable_device_option()->set_device_type(CUDA);
	predict.mutable_device_option()->set_device_type(CUDA);
	save.mutable_device_option()->set_device_type(CUDA);
	Workspace workspace(nullptr);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
	save_net = CreateNet(save,&workspace);
	if(0 != vm.count("load")) load_net = CreateNet(load,&workspace);
	atexit(atexit_handler);
#ifndef NDEBUG
	//show loss degradation
	cvplot::window("loss revolution");
	cvplot::move("loss",300,300);
	cvplot::resize("loss",500,300);
	cvplot::figure("loss").series("train").color(cvplot::Purple);
#endif
	vector<pair<string,ublas::matrix<float> > > filelist;
	makeList(inputdir,filelist);
#ifndef NDEBUG
	cout<<filelist.size()<<" samples loaded"<<endl;
#endif
	int start_sample = -1;
	int start_iter = 0;
	if(0 != vm.count("load")) {
		load_net->Run();
		//move blob "iter" from cuda to cpu
		TensorCPU itertensor = TensorCPU(workspace.GetBlob("iter")->Get<TensorCUDA>());
		workspace.RemoveBlob("iter");
		auto tensor = workspace.CreateBlob("iter")->GetMutable<TensorCPU>();
		tensor->ResizeLike(itertensor);
		tensor->ShareData(itertensor);
		//read iter number
		const long * iter = itertensor.data<long>();
		cout<<"load model from "<<modeldir<<endl
			<<"and resume to "<<iter[0]<<" iteration"<<endl;
		start_sample = iter[0] % filelist.size();
		start_iter = iter[0];
	}
	for(int i = start_iter ; ; i++) {
		//fill a batch of training samples
		pair<TensorCUDA,TensorCUDA> tensors = fillTrainBlob(filelist,"data","depth",BATCHSIZE,start_sample);
		auto inputtensor = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
		auto outputtensor = workspace.CreateBlob("depth")->GetMutable<TensorCUDA>();
		inputtensor->ResizeLike(tensors.first);
		inputtensor->ShareData(tensors.first);
		outputtensor->ResizeLike(tensors.second);
		outputtensor->ShareData(tensors.second);
		//train on the batch of training samples
		predict_net->Run();
		cout<<"iter:"<<i<<endl;
		if(i % 1000 == 0) {
			cout<<"saving params"<<endl;
			remove_all("ResNetDepth_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupBottleneck(string input_blob,string output_blob, int inplanes,int planes, int stride, ModelUtil & network,bool test)
{
	//channel number: inplanes->planes * 4
	//size: size -> size / stride
	
	network.AddConvOps(input_blob,input_blob + "_conv1",inplanes,planes,1,0,1,0,test);
	network.AddSpatialBNOps(input_blob + "_conv1",input_blob + "_bn1",planes,1e-5f,0.9,test);
	network.AddLeakyReluOp(input_blob + "_bn1",input_blob + "_bn1",0.2);
	
	network.AddConvOps(input_blob + "_bn1", input_blob + "_conv2",planes,planes,stride,1,3,0,test);
	network.AddSpatialBNOps(input_blob + "_conv2",input_blob + "_bn2",planes,1e-5f,0.9,test);
	network.AddLeakyReluOp(input_blob + "_bn2",input_blob + "_bn2",0.2);
	
	network.AddConvOps(input_blob + "_bn2", input_blob + "_conv3",planes,planes * 4,1,0,1,0,test);
	network.AddSpatialBNOps(input_blob + "_conv3", input_blob + "_bn3",planes * 4,1e-5f,0.9,test);
	
	string residual;
	//stride != 1 the size of feature map shrinks
	//inplanes != planes * 4 the channel number of feature map shrinks
	if(stride != 1 || inplanes != planes * 4) {
		network.AddConvOps(input_blob,input_blob + "_res_conv",inplanes,planes * 4,stride,0,1,0,test);
		network.AddSpatialBNOps(input_blob + "_res_conv",input_blob + "_res_bn",planes * 4,1e-5f,0.9,test);
		residual = input_blob + "_res_bn";
	} else residual = input_blob;
	
	network.AddSumOp({input_blob + "_bn3",residual},output_blob);
	network.AddLeakyReluOp(output_blob,output_blob,0.2);
}

void setupConvBlock(string input_blob,string output_blob,int inplanes,int planes,int blocks, int stride,ModelUtil & network,bool test)
{
	//channel number: inplanes->planes * 4
	//size: size->size / stride
	
	//channel number: planes->planes * 4
	setupBottleneck(input_blob,input_blob + "_block1",inplanes,planes,stride,network,test);
	inplanes = planes * 4;
	for(int i = 1 ; i < blocks ; i++) {
		//channel number: planes * 4->planes * 4
		setupBottleneck(input_blob + "_block" + lexical_cast<string>(i),(i < blocks - 1)?(input_blob + "_block" + lexical_cast<string>(i+1)):output_blob,inplanes,planes,1,network,test);
	}
}

void setupTrainNet(NetDef & init, NetDef & predict,int module_num,bool test)
{
	//load module
	LoadModule("","libcaffe2_detectron_ops_gpu.so");
	
	//create network
	ModelUtil network(init,predict);
	network.init.AddConstantFloatFillOp({BATCHSIZE,3+68,256,256},0,"data");
	network.init.AddConstantFloatFillOp({BATCHSIZE,68},0,"depth");
	network.predict.AddInput("data");
	if(false == test)
		network.predict.AddInput("depth");
	//data 1x(3+68)x64x64
	//channel: 3+68->64
	//size: 64->32
	network.AddConvOps("data","conv1",3 + 68,64,2,3,7,0,test);
	network.AddSpatialBNOps("conv1","bn1",64,1e-5f,0.9,test);
	network.AddLeakyReluOp("bn1","bn1",0.2);
	//channel: 64->64
	//size: 32->16
	network.AddMaxPoolOp("bn1","pool1",2,1,3);
	//channel: 64->256
	//size: 16->16
	setupConvBlock("pool1","block1",64,64,3,1,network,test);
	//channel: 256->512
	//size: 16->8
	setupConvBlock("block1","block2",256,128,8,2,network,test);
	//channel: 512->1024
	//size: 8->4
	setupConvBlock("block2","block3",512,256,36,2,network,test);
	//channel: 1024->2048
	//size: 4->2
	setupConvBlock("block3","block4",1024,512,3,2,network,test);
	
	network.AddAveragePoolOp("block4","avg",7,0,7);
	network.AddReshapeOp("avg","avg_reshaped",{BATCHSIZE,-1});
	network.AddFcOps("avg_reshaped","predict",512 * 4,68,1,test);
	if(test) {
		network.predict.WriteText("models/ResNetDepth_deploy_predict.pbtxt");
		return;
	}
	//NOTE:kernel size 64x64 should be changed according to size of output tensor
	network.AddSquaredL2DistanceOp({"predict","depth"},"loss");
	//loss
	network.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	network.predict.AddGradientOps();
	network.AddIterOps();
#ifndef NDEBUG
	network.predict.AddTimePlotOp("loss","iter","loss","train",10);
#endif
	network.AddLearningRateOp("iter", "lr", -0.01,0.9,20*round(static_cast<float>(TRAINSIZE)/BATCHSIZE));
	string optimizer = "adam";
	network.AddOptimizerOps(optimizer);
	//输出网络结构
	network.predict.WriteText("models/ResNetDepth_train_predict.pbtxt");
	network.init.WriteText("models/ResNetDepth_train_init.pbtxt");
}

void setupSaveNet(NetDef & init, NetDef & save)
{
	NetUtil InitNet(init);
	NetUtil SaveNet(save);
	vector<string> params;
	for(auto & op : InitNet.net.op()) {
		if(op.type() == "CreateDB") continue;
		for(auto & output : op.output())
			params.push_back(output);
	}
	SaveNet.AddSaveOp(params,"lmdb","ResNetDepth_params");
	//output network
	SaveNet.WriteText("models/ResNetDepth_train_save.pbtxt");
}

void setupLoadNet(NetDef & init, NetDef & load,string modelpath)
{
	NetUtil InitNet(init);
	NetUtil LoadNet(load);
	vector<string> params;
	for(auto & op : InitNet.net.op()) {
		if(op.type() == "CreateDB") continue;
		for(auto & output : op.output())
			params.push_back(output);
	}
	LoadNet.AddLoadOp(params,"lmdb",modelpath);
	//output network
	LoadNet.WriteText("models/ResNetDepth_train_load.pbtxt");
}

void makeList(path dir,vector<pair<string,ublas::matrix<float> > >& filelist)
{
	for(directory_iterator it(dir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) makeList(it->path(),filelist);
		else if(it->path().extension() == ".csv") {
			string stem = it->path().filename().stem().string();
			path filename = it->path().parent_path() / (stem + ".jpg");
			if(false == exists(filename)) {
				cout<<it->path().string()<<" exists, but"<<endl;
				cout<<filename.string()<<" doesn't"<<endl;
				continue;
			}
			std::ifstream in(it->path().string());
			if(false == in.is_open()) {
				cout<<it->path().string()<<" can't be opened"<<endl;
				continue;
			}
			string line;
			ublas::matrix<float> kps(3,68);
			getline(in,line);
			tokenizer<escaped_list_separator<char> > tkx(line, escaped_list_separator<char>('\\',',','\"'));
			tokenizer<escaped_list_separator<char> >::iterator x(tkx.begin());
			for(int j = 0 ; j < 68 ; j++) kps(0,j) = lexical_cast<float>(*x++);
			getline(in,line);
			tokenizer<escaped_list_separator<char> > tky(line, escaped_list_separator<char>('\\',',','\"'));
			tokenizer<escaped_list_separator<char> >::iterator y(tky.begin());
			for(int j = 0 ; j < 68 ; j++) kps(1,j) = lexical_cast<float>(*y++);
			getline(in,line);
			tokenizer<escaped_list_separator<char> > tkz(line, escaped_list_separator<char>('\\',',','\"'));
			tokenizer<escaped_list_separator<char> >::iterator z(tkz.begin());
			for(int j = 0 ; j < 68 ; j++) kps(2,j) = lexical_cast<float>(*z++);
			filelist.push_back(make_pair(filename.string(),kps));
		}
	}
}

Mat drawGaussian(cv::Size sz, Point2f pt)
{
	if(false == (pt.x - 3 >= 0 && pt.y - 3 >= 0)) return Mat::zeros(sz,CV_32F);
	if(false == (pt.x + 3 < sz.width && pt.y + 3 < sz.height)) return Mat::zeros(sz,CV_32F);
	Mat heatmap = Mat::zeros(sz,CV_32FC1);
	//generate 7x7 gaussian kernel
	int ksize = 7;
	Mat kernel1D = getGaussianKernel(ksize,0.3*((ksize-1)*0.5 - 1) + 0.8,CV_32F);
	Mat kernel2D = 12.054 * kernel1D * kernel1D.t();
	//copy value to the subimg centered at pt
	kernel2D.copyTo(heatmap(cv::Rect(pt.x - 3,pt.y - 3,7,7)));
	return heatmap;
}

Point2f transform(Point2f pt, Point2f center, float s, float rot, int res)
{
	float rad = rot * pi<float>() / 180;
	Mat affine;
	//translate image to make center at the origin
	Mat translation1 = Mat::eye(cv::Size(3,3),CV_32FC1);
	translation1.at<float>(0,2) = -center.x;
	translation1.at<float>(1,2) = -center.y;
	//counter-clockwise rotate image for rot angle
	Mat rotate = Mat::eye(cv::Size(3,3),CV_32FC1);
	rotate.at<float>(0,0) = cos(rad);	rotate.at<float>(0,1) = -sin(rad);
	rotate.at<float>(1,0) = sin(rad);	rotate.at<float>(1,1) = cos(rad);
	//scale image
	affine = s * rotate * translation1;
	//translate image to let the new upper left at the origin
	Mat translation2 = Mat::eye(cv::Size(3,3),CV_32FC1);
	translation2.at<float>(0,2) = res / 2;
	translation2.at<float>(1,2) = res / 2;
	affine = translation2 * affine;
	//transform coordinate
	affine.at<float>(2,2) = 1;
	Mat pt_ = Mat::ones(cv::Size(1,3),CV_32FC1);
	pt_.at<float>(0,0) = pt.x;
	pt_.at<float>(1,0) = pt.y;
	Mat npt_ = affine * pt_;
	return Point2f(npt_.at<float>(0,0),npt_.at<float>(1,0));
}

Mat crop(Mat img, Point2f center, float s, float rot, int res)
{
	float rad = rot * pi<float>() / 180;
	Mat affine;
	//translate image to make center at the origin
	Mat translation1 = Mat::eye(cv::Size(3,3),CV_32FC1);
	translation1.at<float>(0,2) = -center.x;
	translation1.at<float>(1,2) = -center.y;
	//counter-clockwise rotate image for rot angle
	Mat rotate = Mat::eye(cv::Size(3,3),CV_32FC1);
	rotate.at<float>(0,0) = cos(rad);	rotate.at<float>(0,1) = -sin(rad);
	rotate.at<float>(1,0) = sin(rad);	rotate.at<float>(1,1) = cos(rad);
	//scale image
	affine = s * rotate * translation1;
	//translate image to let the new upper left at the origin
	Mat translation2 = Mat::eye(cv::Size(3,3),CV_32FC1);
	translation2.at<float>(0,2) = res / 2;
	translation2.at<float>(1,2) = res / 2;
	affine = translation2 * affine;
	Mat affine_submat = affine(cv::Rect(0,0,3,2));
	//crop
	Mat retVal;
	warpAffine(img, retVal, affine_submat, cv::Size(res,res));
	return retVal;
}

pair<Mat,ublas::matrix<float> > augment(Mat img,ublas::matrix<float> original)
{
#ifndef NDEBUG
	assert(68 == original.size2());
#endif
	
	static random::mt19937 engine;
	static random::variate_generator<random::mt19937&,random::uniform_real_distribution<float> > uniform(engine,random::uniform_real_distribution<float>(0,1));
	static random::variate_generator<random::mt19937&,random::uniform_real_distribution<float> > uniform_color(engine,random::uniform_real_distribution<float>(0.7,1.3));
	static random::variate_generator<random::mt19937&,random::normal_distribution<> > norm(engine,random::normal_distribution<>(0.0,1.0));
	//1) 100% scale
	float s = fmax(fmin(norm() * 0.2 + 0.8,0.6),1.0); 					//scale in [0.6,1.0]
	//2) 50% rotate
	float r = (uniform() >= 0.5)?fmax(fmin(norm() * 10,20),-20):0;		//rot in [-10,10]
	Point2f center(450 / 2,450 / 2 + 50);
	Mat subimg = crop(img,center,s,r,256);
	ublas::matrix<float> transformed(3,68);
	for(int i = 0 ; i < original.size2() ; i++) {
		Point2f p(original(0,i),original(1,i));
		Point2f tp = ::transform(p,center,s,r,256);
		transformed(0,i) = tp.x;
		transformed(1,i) = tp.y;
		transformed(2,i) = original(2,i);
	}
	//3) 20% emulate low resolution
	if(uniform() <= 0.2) {
		Mat rz;
		resize(subimg,rz,cv::Size(96,96));
		resize(rz,subimg,cv::Size(256,256));
	}
	//4) 50% flip
	if(uniform() <= 0.5) {
		flip(subimg,subimg,1);
		for(int i = 0 ; i < transformed.size2() ; i++) {
			transformed(0,i) = subimg.cols - transformed(0,i);
		}
	}
	//5) 100% color augmentation
	vector<Mat> channels;
	split(subimg,channels);
	for(int c = 0 ; c < 3 ; c++) channels[c] *= fmax(fmin(uniform_color(),1),0);
	merge(channels,subimg);
	return make_pair(subimg,transformed);
}

pair<TensorCUDA,TensorCUDA> fillTrainBlob(vector<pair<string,ublas::matrix<float> > > filelist,string inputname,string outputname,int batchsize,int start_sample)
{
	assert(0 == batchsize % 10);
	assert(start_sample < static_cast<long>(filelist.size()));
	//generate a batchsize of augmented samples of one person
	static int fileid = (-1 == start_sample)?0:start_sample;
	//loop among training samples
	if(fileid == filelist.size()) fileid = 0;
	vector<TIndex> inputdims({batchsize,3 + 68,256,256});
	vector<float> inputdata(batchsize * (3 + 68) * 256 * 256);
	vector<TIndex> outputdims({batchsize,68});
	vector<float> outputdata(batchsize * 68);
	
	int index = 0;
	for(int count = 0 ; count < batchsize / 10 ; count++,fileid++) {
		Mat img = imread(filelist[fileid].first);
		if(img.empty()) throw runtime_error(filelist[fileid].first + " can't be opened");
		Point2f center(450/2,450/2 + 50);
		float scale = 0.8;
		//1)one original sample
		Mat subimg = crop(img,center,scale,0,256);
		for(int c = 0 ; c < 3 + 68 ; c++) {
			if(c < 3) {
				//color map
				for(int h = 0 ; h < subimg.rows ; h++)
					for(int w = 0 ; w < subimg.cols ; w++)
						inputdata[index * (3 + 68) * 256 * 256 + c * 256 * 256 + h * 256 + w] = subimg.ptr<unsigned char>(h)[w * subimg.channels() + c];
			} else {
				//heat maps of keypoints
				Point2f keypoint(
					filelist[fileid].second(0,c - 3),
					filelist[fileid].second(1,c - 3)
				);
				Point2f transformed = ::transform(keypoint,center,scale,0,256);
				Mat heatmap = drawGaussian(cv::Size(256,256),transformed);
				for(int h = 0 ; h < heatmap.rows ; h++)
					for(int w = 0 ; w < heatmap.cols ; w++)
						inputdata[index * (3 + 68) * 256 * 256 + c * 256 * 256 + h * 256 + w] = heatmap.at<float>(h,w);
			}
		}
		for(int c = 0 ; c < 68 ; c++) {
			outputdata[index * 68 + c] = filelist[fileid].second(2,c);
		}
		index++;
		//2)nine augmented sample
		for(int i = 0 ; i < 9 ; i++,index++) {
			pair<Mat,ublas::matrix<float> > augmented = augment(img,filelist[fileid].second);
			for(int c = 0 ; c < 3 + 68 ; c++) {
				if(c < 3) {
					//color map 
					for(int h = 0 ; h < augmented.first.rows ; h++)
						for(int w = 0 ; w < augmented.first.cols ; w++)
							inputdata[index * (3 + 68) * 256 * 256 + c * 256 * 256 + h * 256 + w] = augmented.first.ptr<unsigned char>(h)[w * augmented.first.channels() + c];
				} else {
					//heat maps of keypoints
					Point2f keypoint(
						augmented.second(0,c - 3),
						augmented.second(1,c - 3)
					);
					Point2f transformed = ::transform(keypoint,center,scale,0,256);
					Mat heatmap = drawGaussian(cv::Size(256,256),transformed);
					for(int h = 0 ; h < heatmap.rows ; h++)
						for(int w = 0 ; w < heatmap.cols ; w++)
							inputdata[index * (3 + 68) * 256 * 256 + c * 256 * 256 + h * 256 + w] = heatmap.at<float>(h,w);
				}
			}
			for(int c = 0 ; c < 68 ; c++) {
				outputdata[index * 68 + c] = augmented.second(2,c);
			}
		}
	}
	TensorCUDA input = TensorCUDA(TensorCPU(inputdims,inputdata,NULL));
	TensorCUDA output = TensorCUDA(TensorCPU(outputdims,outputdata,NULL));
	return make_pair(input,output);
}
