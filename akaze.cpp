//
// 2016/1/9
// for opencv3.1
// http://docs.opencv.org/3.1.0/db/d70/tutorial_akaze_matching.html#gsc.tab=0
//
#include <opencv2/opencv.hpp>

int main( int argc, char** argv )
{
	cv::Mat image, gimage;
	std::string fname = "lena.jpg";
	image = cv::imread(fname);
	cv::cvtColor(image, gimage, cv::COLOR_BGR2GRAY);

	//create (int descriptor_type=AKAZE::DESCRIPTOR_MLDB, int descriptor_size=0, int descriptor_channels=3,
		// float threshold=0.001f, int nOctaves=4, int nOctaveLayers=4, int diffusivity=KAZE::DIFF_PM_G2)

	cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
	std::vector<cv::KeyPoint> keypoints;
	akaze->detect(gimage, keypoints);

	//cv::Mat descriptors;
	//akaze->detectAndCompute(gimage, cv::noArray(), keypoints, descriptors);

	std::cout << keypoints.size() << std::endl;
	cv::drawKeypoints(image, keypoints, image);

	cv::imshow( "akaze image", image );

	cv::waitKey(0);

	return 0;
}
