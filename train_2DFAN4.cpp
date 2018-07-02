#include <cstdlib>
#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/constants/constants.hpp>
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
using namespace boost::math::constants;
using namespace boost::program_options;
using namespace cv;
using namespace caffe2;
using namespace cvplot;

void setupTrainNet(NetDef & init, NetDef & predict,int module_num = 4,bool test = false);
void setupSaveNet(NetDef & init, NetDef & save);
void setupLoadNet(NetDef & init, NetDef & load,string modeldir);
void makeList(path dir,vector<pair<string,vector<Point2f> > >& filelist);
pair<TensorCUDA,TensorCUDA> fillTrainBlob(vector<pair<string,vector<Point2f> > > filelist,string inputname,string outputname,int batchsize,int start_sample = -1);

unique_ptr<NetBase> predict_net;
unique_ptr<NetBase> save_net;
unique_ptr<NetBase> load_net;

void atexit_handler()
{
	cout<<"saving params"<<endl;
	remove_all("2DFAN4_params");
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
	load.mutable_device_option()->set_device_type(CUDA);
	Workspace workspace(nullptr);
	workspace.RunNetOnce(init);
	predict_net = CreateNet(predict,&workspace);
	save_net = CreateNet(save,&workspace);
	if(0 != vm.count("load")) load_net = CreateNet(load,&workspace);
	atexit(atexit_handler);
#ifndef NDEBUG
	//show loss degradation
	cvplot::Window::current("loss revolution");
	cvplot::moveWindow("loss",300,300);
	cvplot::resizeWindow("loss",500,300);
	cvplot::figure("loss").series("train").color(cvplot::Purple);
#endif
	vector<pair<string,vector<Point2f> > > filelist;
	makeList(inputdir,filelist);
#ifndef NDEBUG
	std::ofstream debug("debug.txt");
	debug<<filelist.size()<<" samples loaded"<<endl;
	debug.close();
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
		pair<TensorCUDA,TensorCUDA> tensors = fillTrainBlob(filelist,"data","heatmaps",BATCHSIZE,start_sample);
		auto inputtensor = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
		auto outputtensor = workspace.CreateBlob("heatmaps")->GetMutable<TensorCUDA>();
		inputtensor->ResizeLike(tensors.first);
		inputtensor->ShareData(tensors.first);
		outputtensor->ResizeLike(tensors.second);
		outputtensor->ShareData(tensors.second);
		//train on the batch of training samples
		predict_net->Run();
		cout<<"iter:"<<i<<endl;
		if(i % 1000 == 0) {
			cout<<"saving params"<<endl;
			remove_all("2DFAN4_params");
			save_net->Run();
		}
	}
	return EXIT_SUCCESS;
}

void setupConvBlock(string input_blob,string output_blob,int in_size,int out_size, ModelUtil & network,bool test)
{
	//NOTE: this is a size preserving operator set
#ifndef NDEBUG
	assert(out_size % 4 == 0);
#endif
	//densenet like block
	network.AddSpatialBNOps(input_blob,input_blob + "_bn1",in_size,1e-5f,0.9,test);
	network.AddLeakyReluOp(input_blob + "_bn1",input_blob + "_bn1",0.2);
	network.AddConvOps(input_blob + "_bn1",input_blob + "_conv1",in_size,out_size / 2,1,1,3,0,test);
	
	network.AddSpatialBNOps(input_blob + "_conv1", input_blob + "_bn2",out_size / 2,1e-5f,0.9,test);
	network.AddLeakyReluOp(input_blob + "_bn2",input_blob + "_bn2",0.2);
	network.AddConvOps(input_blob + "_bn2",input_blob + "_conv2",out_size / 2,out_size / 4,1,1,3,0,test);
	
	network.AddSpatialBNOps(input_blob + "_conv2", input_blob + "_bn3",out_size / 4,1e-5f,0.9,test);
	network.AddLeakyReluOp(input_blob + "_bn3",input_blob + "_bn3",0.2);
	network.AddConvOps(input_blob + "_bn3",input_blob + "_conv3",out_size / 4,out_size / 4,1,1,3,0,test);
	
	network.AddConcatOp({input_blob + "_conv1",input_blob + "_conv2",input_blob + "_conv3"},output_blob);
	
	//resnet like shortcut
	if(in_size != out_size) {
		network.AddSpatialBNOps(input_blob, input_blob + "_res_bn",in_size,1e-5f,0.9,test);
		network.AddLeakyReluOp(input_blob + "_res_bn", input_blob + "_res_bn",0.2);
		network.AddConvOps(input_blob + "_res_bn",input_blob + "_res_conv",in_size,out_size,1,0,1,0,test);
		network.AddSumOp({output_blob,input_blob + "_res_conv"},output_blob);
	}
}

void setupHourGlass(string input_blob,string output_blob,int depth, ModelUtil & network,bool test)
{
	//NOTE: this is a size preserving operator set
	//branch1: resNext like pass
	setupConvBlock(
		input_blob, 
		input_blob + "_l" + lexical_cast<string>(depth) + "_b1",
		256,256,network,test
	);
	//branch2: hour glass structure
	network.AddAveragePoolOp(
		input_blob, 
		input_blob + "_l" + lexical_cast<string>(depth) + "_b2_downsample",
		2,0,2
	);
	setupConvBlock(
		input_blob + "_l" + lexical_cast<string>(depth) + "_b2_downsample",
		input_blob + "_l" + lexical_cast<string>(depth - 1) + "_input",
		256,256,network,test
	);
	if(depth > 1) 
		setupHourGlass(
			input_blob + "_l" + lexical_cast<string>(depth - 1) + "_input",
			input_blob + "_l" + lexical_cast<string>(depth - 1) + "_output",
			depth - 1, network,test
		);
	else 
		setupConvBlock(
			input_blob + "_l" + lexical_cast<string>(depth - 1) + "_input",
			input_blob + "_l" + lexical_cast<string>(depth - 1) + "_output",
			256, 256, network,test
		);
	setupConvBlock(
		input_blob + "_l" + lexical_cast<string>(depth - 1) + "_output",
		input_blob + "_l" + lexical_cast<string>(depth) + "_b2",
		256, 256, network,test
	);
	network.AddUpsampleNearestOp(
		input_blob + "_l" + lexical_cast<string>(depth) + "_b2",
		input_blob + "_l" + lexical_cast<string>(depth) + "_b2_upsample",2
	);
	//branch1 + branch2
	network.AddSumOp(
		{
			input_blob + "_l" + lexical_cast<string>(depth) + "_b1",
			input_blob + "_l" + lexical_cast<string>(depth) + "_b2_upsample"
		},
		output_blob
	);
}

void setupTrainNet(NetDef & init, NetDef & predict,int module_num,bool test)
{
	//load module
	LoadModule("","libcaffe2_detectron_ops_gpu.so");
	
	//create network
	ModelUtil network(init,predict);
	network.init.AddConstantFloatFillOp({BATCHSIZE,3,256,256},0,"data");
	network.init.AddConstantFloatFillOp({BATCHSIZE,68,64,64},0,"heatmaps");
	network.predict.AddInput("data");
	if(false == test)
		network.predict.AddInput("heatmaps");
	//data 1x3x256x256
	//network structure
	network.AddConvOps("data","data_conv1",3,64,2,3,7,0,test);
	network.AddSpatialBNOps("data_conv1","data_bn1",64,1e-5f,0.9,test);
	network.AddLeakyReluOp("data_bn1","data_bn1",0.2);
	setupConvBlock("data_bn1","convblock1",64,128,network,test);
	network.AddAveragePoolOp("convblock1", "convblock1_pool",2,0,2);
	//convblock1_pool 1x128x128x128
	setupConvBlock("convblock1_pool","convblock2",128,128,network,test);
	setupConvBlock("convblock2","convblock3",128,256,network,test);
	string layer = "convblock3";
	string output;
	for(int i = 0 ; i < module_num ; i++) {
		setupHourGlass(
			layer,
			"m" + lexical_cast<string>(i + 1),
			4,network,test
		);
		setupConvBlock(
			"m" + lexical_cast<string>(i + 1),
			"top_m" + lexical_cast<string>(i + 1),
			256,256,network,test
		);
		network.AddConvOps(
			"top_m" + lexical_cast<string>(i + 1),
			"conv_last" + lexical_cast<string>(i + 1),
			256,256,1,0,1,0,test
		);
		network.AddSpatialBNOps(
			"conv_last" + lexical_cast<string>(i + 1),
			"bn_end" + lexical_cast<string>(i + 1),
			256,1e-5f,0.9,test
		);
		network.AddLeakyReluOp(
			"bn_end" + lexical_cast<string>(i + 1),
			"bn_end" + lexical_cast<string>(i + 1),0.2
		);
		//predict heatmaps
		network.AddConvOps(
			"bn_end" + lexical_cast<string>(i + 1),
			"l" + lexical_cast<string>(i + 1),
			256,68,1,0,1,0,test
		);
		if(i < module_num - 1) {
			network.AddConvOps(
				"bn_end" + lexical_cast<string>(i + 1),
				"bl" + lexical_cast<string>(i + 1),
				256,256,1,0,1,0,test
			);
			network.AddConvOps(
				"l" + lexical_cast<string>(i + 1),
				"al" + lexical_cast<string>(i + 1),
				68,256,1,0,1,0,test
			);
			network.AddSumOp(
				{
					layer,	//resnet like pass
					"bl" + lexical_cast<string>(i + 1),	//convolution of featuremap
					"al" + lexical_cast<string>(i + 1)	//convolution of heatmap
				},
				"prev" + lexical_cast<string>(i + 1)
			);
			layer = "prev" + lexical_cast<string>(i + 1);
		} else output = "l" + lexical_cast<string>(i + 1);
	}
	if(test) {
		network.predict.WriteText("models/2DFAN4_deploy_predict.pbtxt");
		return;
	}
	//NOTE:kernel size 64x64 should be changed according to size of output tensor
	network.AddSquaredL2DistanceOp({output,"heatmaps"},"loss");
	network.AddPrintOp("loss",false);
	//loss
	network.AddConstantFillWithOp(1.0, "loss", "loss_grad");
	network.predict.AddGradientOps();
	network.AddIterOps();
#ifndef NDEBUG
	network.predict.AddTimePlotOp("loss","iter","loss","train",10);
#endif
	network.AddLearningRateOp("iter", "lr", -0.0001,0.9,1*round(static_cast<float>(TRAINSIZE)/BATCHSIZE));
	string optimizer = "adam";
	network.AddOptimizerOps(optimizer);
	//输出网络结构
	network.predict.WriteText("models/2DFAN4_train_predict.pbtxt");
	network.init.WriteText("models/2DFAN4_train_init.pbtxt");
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
	SaveNet.AddSaveOp(params,"lmdb","2DFAN4_params");
	//output network
	SaveNet.WriteText("models/2DFAN4_train_save.pbtxt");
}

void setupLoadNet(NetDef & init, NetDef & load,string modeldir)
{
	NetUtil InitNet(init);
	NetUtil LoadNet(load);
	vector<string> params;
	for(auto & op : InitNet.net.op()) {
		if(op.type() == "CreateDB") continue;
		for(auto & output : op.output())
			params.push_back(output);
	}
	LoadNet.AddLoadOp(params,"lmdb",modeldir);
	//output network
	LoadNet.WriteText("models/2DFAN4_train_load.pbtxt");
}

void makeList(path dir,vector<pair<string,vector<Point2f> > >& filelist)
{
	for(directory_iterator it(dir) ; it != directory_iterator() ; it++) {
		if(is_directory(it->path())) makeList(it->path(),filelist);
		else if(it->path().extension() == ".csv") {
			string stem = it->path().filename().stem().string();
			path filename = it->path().parent_path() / (stem.substr(0,stem.size() - 4) + ".jpg");
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
			vector<Point2f> kps;
			while(false == in.eof()) {
				getline(in,line);
				trim(line);
				if("" == line) break;
				tokenizer<escaped_list_separator<char> > tk(line, escaped_list_separator<char>('\\',',','\"'));
				tokenizer<escaped_list_separator<char> >::iterator i(tk.begin());
				float x = lexical_cast<float>(*i++);
				float y = lexical_cast<float>(*i++);
				kps.push_back(Point2f(x,y));
			}
			if(68 != kps.size()) {
				cout<<it->path().string()<<" doesn't has 68 coordinates"<<endl;
				continue;
			}
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

pair<Mat,vector<Point2f> > augment(Mat img,vector<Point2f> & kps)
{
	static random::mt19937 engine;
	static random::variate_generator<random::mt19937&,random::uniform_real_distribution<float> > uniform(engine,random::uniform_real_distribution<float>(0,1));
	static random::variate_generator<random::mt19937&,random::uniform_real_distribution<float> > uniform_color(engine,random::uniform_real_distribution<float>(0.7,1.3));
	static random::variate_generator<random::mt19937&,random::normal_distribution<> > norm(engine,random::normal_distribution<>(0.0,1.0));
	//1) 0% scale
	float s = fmin(fmax(norm() * 0.2 + 0.8,0.6),1.0); 					//scale in [0.6,1.0]
	//2) 100% translate
	float dx = fmin(fmax(-50,norm() * 25),50);
	float dy = fmin(fmax(-50,norm() * 25),50);
	Point2f center(450 / 2 + dx,450 / 2 + 50 + dy);	
	//2) 50% rotate
	float r = (uniform() >= 0.5)?fmax(fmin(norm() * 10,20),-20):0;		//rot in [-10,10]
	Mat subimg = crop(img,center,s,r,256);
	vector<Point2f> transformed_kps;
	for(auto & p : kps) transformed_kps.push_back(::transform(p,center,s,r,256));
	//3) 20% emulate low resolution
	if(uniform() <= 0.2) {
		Mat rz;
		resize(subimg,rz,cv::Size(96,96));
		resize(rz,subimg,cv::Size(256,256));
	}
	//4) 50% flip
	if(uniform() <= 0.5) {
		flip(subimg,subimg,1);
		vector<Point2f> flpkps;
		for(auto & p : transformed_kps) flpkps.push_back(Point2f(subimg.cols - p.x,p.y));
		transformed_kps = flpkps;
	}
	//5) 100% color augmentation
	vector<Mat> channels;
	split(subimg,channels);
	for(int c = 0 ; c < 3 ; c++) channels[c] *= fmax(fmin(uniform_color(),1),0);
	merge(channels,subimg);
	return make_pair(subimg,transformed_kps);
}

pair<TensorCUDA,TensorCUDA> fillTrainBlob(vector<pair<string,vector<Point2f> > > filelist,string inputname,string outputname,int batchsize,int start_sample)
{
	assert(0 == batchsize % 10);
#ifndef NDEBUG
	std::ofstream debug("debug.txt");
	debug<<start_sample<<endl<<filelist.size()<<endl;
	debug.close();
#endif
	assert(start_sample < static_cast<long>(filelist.size()));
	//generate a batchsize of augmented samples of one person
	static int fileid = (-1 == start_sample)?0:start_sample;
	//loop among training samples
	if(fileid == filelist.size()) fileid = 0;
	vector<TIndex> inputdims({batchsize,3,256,256});
	vector<float> inputdata(batchsize * 3 * 256 * 256);
	vector<TIndex> outputdims({batchsize,68,64,64});
	vector<float> outputdata(batchsize * 68 * 64 * 64);
	
	int index = 0;
	for(int count = 0 ; count < batchsize / 10 ; count++,fileid++) {
		Mat img = imread(filelist[fileid].first);
		if(img.empty()) throw runtime_error(filelist[fileid].first + " can't be opened");
		Point2f center(450/2,450/2+50);
		float scale = 0.8;
		//1)one original sample
		Mat subimg = crop(img,center,scale,0,256);
		for(int c = 0 ; c < 3 ; c++)
			for(int h = 0 ; h < subimg.rows ; h++)
				for(int w = 0 ; w < subimg.cols ; w++)
					inputdata[index * 3 * 256 * 256 + c * 256 * 256 + h * 256 + w] = subimg.ptr<unsigned char>(h)[w * subimg.channels() + c];
		for(int c = 0 ; c < 68 ; c++) {
			Point2f transformed = ::transform(filelist[fileid].second[c],center,scale,0,256);
			transformed.x /= (256/64);
			transformed.y /= (256/64);
			Mat heatmap = drawGaussian(cv::Size(64,64),transformed);
			for(int h = 0 ; h < heatmap.rows ; h++)
				for(int w = 0 ; w < heatmap.cols ; w++)
					outputdata[index * 68 * 64 * 64 + c * 64 * 64 + h * 64 + w] = heatmap.at<float>(h,w);
		}
#if 0
		cv::namedWindow("debug");
		Mat debug = subimg.clone();
		for(int c = 0 ; c < 68 ; c++) {
			Point2f transformed = ::transform(filelist[fileid].second[c],center,scale,0,256);
			cout<<transformed<<endl;
			circle(debug,transformed,2,Scalar(0,0,255),-1);
		}
		imshow("debug",debug);
		cv::waitKey();
#endif
		index++;
		//2)nine augmented sample
		for(int i = 0 ; i < 9 ; i++,index++) {
			pair<Mat,vector<Point2f> > augmented = augment(img,filelist[fileid].second);
			for(int c = 0 ; c < 3 ; c++)
				for(int h = 0 ; h < augmented.first.rows ; h++)
					for(int w = 0 ; w < augmented.first.cols ; w++)
						inputdata[index * 3 * 256 * 256 + c * 256 * 256 + h * 256 + w] = augmented.first.ptr<unsigned char>(h)[w * augmented.first.channels() + c];
			for(int c = 0 ; c < 68 ; c++) {
				Point2f transformed = augmented.second[c];
				transformed.x /= (256/64);
				transformed.y /= (256/64);
				Mat heatmap = drawGaussian(cv::Size(64,64),transformed);
				for(int h = 0 ; h < heatmap.rows ; h++)
					for(int w = 0 ; w < heatmap.cols ; w++)
						outputdata[index * 68 * 64 * 64 + c * 64 * 64 + h * 64 + w] = heatmap.at<float>(h,w);
			}
#if 0
			Mat debug = augmented.first.clone();
			for(int c = 0 ; c < 68 ; c++) {
				circle(debug,augmented.second[c],2,Scalar(0,0,255),-1);
			}
			imshow("debug",debug);
			cv::waitKey();
#endif
		}
	}
	TensorCUDA input = TensorCUDA(TensorCPU(inputdims,inputdata,NULL));
	TensorCUDA output = TensorCUDA(TensorCPU(outputdims,outputdata,NULL));
	return make_pair(input,output);
}
