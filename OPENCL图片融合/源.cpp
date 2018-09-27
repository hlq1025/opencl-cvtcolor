#include <CL/cl.h>
#include <opencv.hpp>
#include <iostream>
#include<fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#pragma warning( disable : 4996 )
using namespace std;
using namespace cv;
cl_context CreateContext()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	// 选择OpenCL平台
	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		std::cerr << "Failed to find any OpenCL platforms." << std::endl;
		return NULL;
	}

	// 在OpenCL平台上创建一个队列，先试GPU，再试CPU
	cl_context_properties contextProperties[] =
	{
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)firstPlatformId,
		0
	};
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
		NULL, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		std::cout << "Could not create GPU context, trying CPU..." << std::endl;
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
			NULL, NULL, &errNum);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
			return NULL;
		}
	}

	return context;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	// First get the size of the devices buffer  
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
		return NULL;
	}

	if (deviceBufferSize <= 0)
	{
		std::cerr << "No devices available.";
		return NULL;
	}

	// Allocate memory for the devices buffer  
	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Failed to get device IDs";
		return NULL;
	}

	// In this example, we just choose the first available device.  In a  
	// real program, you would likely use all available devices or choose  
	// the highest performance device based on OpenCL device queries  
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	if (commandQueue == NULL)
	{
		std::cerr << "Failed to create commandQueue for device 0";
		return NULL;
	}

	*device = devices[0];
	delete[] devices;
	return commandQueue;
}

//  Create an OpenCL program from the kernel source file  
//  
cl_program CreateProgram(cl_context context, cl_device_id device, std::string fileName)
{
	cl_int errNum;
	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);
	if (!kernelFile.is_open())
	{
		std::cerr << "Failed to open file for reading: " << fileName << std::endl;
		return NULL;
	}

	std::ostringstream oss;
	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();
	const char *srcStr = srcStdStr.c_str();
	program = clCreateProgramWithSource(context, 1,
		(const char**)&srcStr,
		NULL, NULL);
	if (program == NULL)
	{
		std::cerr << "Failed to create CL program from source." << std::endl;
		return NULL;
	}

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error  
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

cl_bool ImageSupport(cl_device_id device)
{
	cl_bool imageSupport = CL_FALSE;
	clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool),
		&imageSupport, NULL);
	return imageSupport;
}
cl_mem LoadImage(cl_context context, std::string fileName, int &width, int &height)
{

	cv::Mat image1 = cv::imread(fileName);
	width = image1.cols;
	height = image1.rows;
	char *buffer = new char[width * height * 4];

	//数据传入方式：一个像素一个像素，按照B G R顺序，中间空一格 就像： 
	// 12 237 34  221 88 99  22 33 99
	int w = 0;
	for (int v = height - 1; v >= 0; v--)
	{
		for (int u = 0; u < width; u++)
		{
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[0];
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[1];
			buffer[w++] = image1.at<cv::Vec3b>(v, u)[2];
			w++;
		}
	}

// Create OpenCL image  
cl_image_format clImageFormat;
clImageFormat.image_channel_order = CL_RGBA;
clImageFormat.image_channel_data_type = CL_UNORM_INT8;

cl_int errNum;
cl_mem clImage;
clImage = clCreateImage2D(context,
	CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
	&clImageFormat,
	width,
	height,
	0,
	buffer,
	&errNum);

if (errNum != CL_SUCCESS)
{
	std::cerr << "Error creating CL image object" << std::endl;
	return 0;
}

return clImage;
}
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel, cl_mem imageObjects[2], cl_sampler sampler)
{
	for (int i = 0; i < 2; i++)
	{
		if (imageObjects[i] != 0)
			clReleaseMemObject(imageObjects[i]);
	}
	if (commandQueue != 0)
		clReleaseCommandQueue(commandQueue);

	if (kernel != 0)
		clReleaseKernel(kernel);

	if (program != 0)
		clReleaseProgram(program);

	if (sampler != 0)
		clReleaseSampler(sampler);

	if (context != 0)
		clReleaseContext(context);

}

size_t RoundUp(int groupSize, int globalSize)
{
	int r = globalSize % groupSize;
	if (r == 0)
	{
		return globalSize;
	}
	else
	{
		return globalSize + groupSize - r;
	}
}

int main(int argc, char** argv)
{
	cl_context context = 0;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_mem imageObjects[2] = { 0, 0 }; //一个原图像 一个目标图像
	cl_sampler sampler = 0;
	cl_int errNum;
	string cl_kernel_file = "bgr2gray.cl";//OpenCL 文件路径
	// 获取设备 
	context = CreateContext();
	if (context == NULL)
	{
		cerr << "Failed to create OpenCL context." << endl;
		cin.get();
	}

	//创建队列
	commandQueue = CreateCommandQueue(context, &device);
	if (commandQueue == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 确保计算设备能够支持图片  
	if (ImageSupport(device) != CL_TRUE)
	{
		cerr << "OpenCL device does not support images." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 将图片载入OpenCL设备
	int width, height; //在LoadImage函数改变了其值
	string src0 = "1.jpg";
	imageObjects[0] = LoadImage(context, src0, width, height);
	if (imageObjects[0] == 0)
	{
		cerr << "Error loading: " << string(src0) << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 创建目标图像 （处理得到的）
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = CL_RGBA;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;
	imageObjects[1] = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &clImageFormat, width, height, 0, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL output image object." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	// Create sampler for sampling image object  
	sampler = clCreateSampler(context,
		CL_FALSE, // Non-normalized coordinates  
		CL_ADDRESS_CLAMP_TO_EDGE,
		CL_FILTER_NEAREST,
		&errNum);

	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL sampler object." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	// 创建函数项
	program = CreateProgram(context, device, cl_kernel_file);
	if (program == NULL)
	{
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	//创建一个OpenCL中的函数
	kernel = clCreateKernel(program, "bgr2gray", NULL);
	if (kernel == NULL)
	{
		cerr << "Failed to create kernel" << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// 传入参数
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageObjects[0]);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &imageObjects[1]);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error setting kernel arguments." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		return 1;
	}

	//貌似是空间大小？
	size_t localWorkSize[2] = { 16, 16 };
	size_t globalWorkSize[2] = { RoundUp(localWorkSize[0], width),
		RoundUp(localWorkSize[1], height) };
	//开始运算
	
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);


	if (errNum != CL_SUCCESS)
	{
		cerr << "Error queuing kernel for execution." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	// Read the output buffer back to the Host  
	char *buffer = new char[width * height * 4];
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	errNum = clEnqueueReadImage(commandQueue, imageObjects[1], CL_TRUE,
		origin, region, 0, 0, buffer,
		0, NULL, NULL);
	
	
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error reading result buffer." << endl;
		Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
		cin.get();
		return 1;
	}

	cv::Mat imageColor = cv::imread(src0);
	
	cv::cvtColor(imageColor, imageColor, CV_BGR2GRAY);
	cv::imshow("OpenCV-BGR2GRAY", imageColor);
	cv::Mat imageColor1 = cv::imread(src0);
	cv::Mat imageColor2;
	imageColor2.create(imageColor.rows, imageColor.cols, imageColor1.type());
	int w = 0;
	for (int v = imageColor2.rows - 1; v >= 0; v--)
	{
		for (int u = 0; u < imageColor2.cols; u++)
		{
			imageColor2.at<cv::Vec3b>(v, u)[0] = buffer[w++];
			imageColor2.at<cv::Vec3b>(v, u)[1] = buffer[w++];
			imageColor2.at<cv::Vec3b>(v, u)[2] = buffer[w++];
			w++;
		}
	}

cv::imshow("OpenCL-BGR2GRAY", imageColor2);
cv::waitKey(0);
delete[] buffer;
Cleanup(context, commandQueue, program, kernel, imageObjects, sampler);
return 0;
}
		