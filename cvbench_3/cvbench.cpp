#include <opencv2/opencv.hpp>
#define LOOP 20

namespace cv
{
    class TickMeter
    {
    public:
        TickMeter();
        void start();
        void stop();

        int64 getTimeTicks() const;
        double getTimeMicro() const;
        double getTimeMilli() const;
        double getTimeSec()   const;
        int64 getCounter() const;

        void reset();
    private:
        int64 counter;
        int64 sumTime;
        int64 startTime;
    };

    cv::TickMeter::TickMeter() { reset(); }
    int64 cv::TickMeter::getTimeTicks() const { return sumTime; }
    double cv::TickMeter::getTimeSec()   const { return (double)getTimeTicks()/getTickFrequency(); }
    double cv::TickMeter::getTimeMilli() const { return getTimeSec()*1e3; }
    double cv::TickMeter::getTimeMicro() const { return getTimeMilli()*1e3; }
    int64 cv::TickMeter::getCounter() const { return counter; }
    void  cv::TickMeter::reset() {startTime = 0; sumTime = 0; counter = 0; }

    void cv::TickMeter::start(){ startTime = getTickCount(); }
    void cv::TickMeter::stop()
    {
        int64 time = getTickCount();
        if ( startTime == 0 )
            return;

        ++counter;

        sumTime += ( time - startTime );
        startTime = 0;
    }
}

cv::TickMeter	tmeter;

void message(std::string msg)
{
  tmeter.stop();
  double time = tmeter.getTimeMilli();
  std::cout << msg << cvRound(time / LOOP * 100.0)/ 100.0 << std::endl;
  tmeter.reset();
}

//
// 拡大縮小、白黒
//
cv::Mat bench1(cv::Mat mat1)
{
  cv::Mat tmp;

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    tmp = mat1.clone();
  }
  message("clone RGB, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::pyrDown(mat1, tmp);
  }
  message("pyrDown, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::pyrUp(mat1, tmp);
  }
  message("pyrUp, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::cvtColor(mat1, tmp, CV_BGR2GRAY);
  }
  message("cvtColor, ");

  return tmp;
}

//
//　ぼかし、エッジ、２値化
//
void bench2(cv::Mat gray)
{
  cv::Mat mat1;
  cv::Size ksize(5,5);

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::GaussianBlur(gray, mat1, ksize, 0, 0);
  }
  message("GaussianBlur ksize(5x5), ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::Sobel(gray, mat1, CV_32F, 1, 0);
  }
  message("Sobel CV_32F xorder, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::threshold(gray, mat1, 128, 255, cv::THRESH_BINARY);
  }
  message("threshold BINARY Th 128 Max 255, ");
}

//
//　特徴点、特徴量
//
void bench3(cv::Mat gray1, cv::Mat gray2)
{
  cv::Ptr<cv::ORB> detector = cv::ORB::create();

  std::vector<cv::KeyPoint> keypoint1;
  std::vector<cv::KeyPoint> keypoint2;
  cv::Mat descriptor1;
  cv::Mat descriptor2;

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    detector->detect(gray1, keypoint1);
  }
  message("Orb FeatureDetector, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    detector->compute(gray1, keypoint1, descriptor1);
  }
  message("Orb DescriptorExtractor, ");

  detector->detect(gray2, keypoint2);
  detector->compute(gray2, keypoint2, descriptor2);

  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector< cv::DMatch > matches;

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    matcher.match( descriptor1, descriptor2, matches );
  }
  message("BFMatcher NORM_L2, ");
}

//
// cascade detection
//
void bench4(cv::Mat mat2)
{
  std::vector<cv::Rect> objects;
  cv::CascadeClassifier cascade("haarcascade_frontalface_alt.xml");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cascade.detectMultiScale(mat2, objects);
  }
  message("CascadeClassifier, ");
}

//
//　StereoBM, オプティカルフロー
//
void bench5(cv::Mat mat2, cv::Mat mat3)
{
  cv::Mat tmp;
  cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create();
  //cv::StereoBM bm(cv::StereoBM::BASIC_PRESET);

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    bm->compute(mat2, mat3, tmp);
  }
  message("StereoBM BASIC_PRESET, ");

  tmeter.start();
  for (int i = 0 ; i < LOOP ; i++) {
    cv::calcOpticalFlowFarneback(mat2, mat3, tmp, 0.5, 3, 9, 3, 5, 1.1, 0);
  }
  message("calcOpticalFlowFarneback, ");
}

int main( int argc, char** argv )
{
  int loop = LOOP;
  if (argc == 2) {
    loop = std::atoi(argv[1]);
  } else {
    std::cout << cv::getBuildInformation() << std::endl;
  }

  std::cout << "OpenCV Benchmark Ver 0.2" << std::endl;
  std::cout << "Loop Count = " << loop << std::endl << std::endl;

  cv::Mat mat1 = cv::imread( "board.jpg");
  cv::Mat mat2 = cv::imread( "basketball1.png", 0);
  cv::Mat mat3 = cv::imread( "basketball2.png", 0);

  cv::Mat gray = bench1(mat1);
  bench2(gray.clone());
  bench3(mat2, mat3);
  bench5(mat2, mat3);
  bench4(mat2.clone());

  cv::imshow( "Push Any Key", gray );
  std::cout << "Push Any Key" << std::endl;
  cv::waitKey(0);

  return 0;
}
