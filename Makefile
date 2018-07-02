CAFFE2_PREFIX=/home/sys411/opt/caffe2
CAFFE2_HELPER_PREFIX=/home/sys411/opt/caffe2_helper
CUDA_PREFIX=/usr/local/cuda
CXXFLAGS=`pkg-config --cflags opencv dlib-1 eigen3` -I. -I${CUDA_PREFIX}/include -I${CAFFE2_PREFIX}/include \
-I${CAFFE2_HELPER_PREFIX}/include -std=c++14 -O2 -msse3 -msse4
LIBS= -L${CUDA_PREFIX}/lib64 -L${CAFFE2_HELPER_PREFIX}/lib -lcaffe2_cpp -lcaffe2_cpp_gpu \
-L${CAFFE2_PREFIX}/lib -lcaffe2_gpu -lcaffe2 -lcaffe2_observers -lcaffe2_protos \
`pkg-config --libs opencv dlib-1 eigen3` \
-lglog -lprotobuf -lcudart -lcurand \
-lboost_filesystem -lboost_system -lboost_thread -lboost_regex -lboost_program_options -lpthread -ldl -llapack
OBJS=$(patsubst %.cpp,%.o,$(wildcard *.cpp))

all: train_2DFAN4 train_ResNetDepth 2DLandmarker

train_2DFAN4: train_2DFAN4.o
	$(CXX) $^ -o ${@} $(LIBS)

train_ResNetDepth: train_ResNetDepth.o
	$(CXX) $^ -o ${@} $(LIBS)

2DLandmarker: 2DLandmarker.o Img2Landmark.o
	$(CXX) $^ -o ${@} $(LIBS)

clean:
	$(RM) train_2DFAN4 train_ResNetDepth 2DLandmarker $(OBJS)
