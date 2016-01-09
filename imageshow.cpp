//
// 2016/1/9
//
#include <opencv2/opencv.hpp>

int main( int argc, char** argv )
{
  cv::Mat mat = cv::imread( "lena.jpg");
  cv::imshow( "Image", mat );

  cv::waitKey(0);

  return 0;
}
