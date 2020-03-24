#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

std::vector<std::string> getOutputsNames(const cv::dnn::Net &net);

int main(int argc, char **argv) {
    std::string modelConfiguration = "yolov3.cfg";
    std::string modelWeights = "yolov3.weights";

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture video("test.mp4");

    if (!video.isOpened()) {
        std::cout << "Error opening video file " << std::endl;
        return -1;
    }

    cv::Mat frame;
    int totalFrames = 0;
    int skipFrames = 30;

    cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();

    std::cout << "Start" << std::endl;

    std::vector<double> direction;
    int inCount = 0, outCount = 0;
    int differenceBetweenCounting = 0;

    while (video.isOpened()) {
        video >> frame;
        totalFrames++;

        if (frame.empty()) {
            break;
        }

        if (totalFrames % skipFrames == 0) {
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(640, 480), cv::Scalar(0, 0, 0), true, false);
            net.setInput(blob);

            std::vector<cv::Mat> outs;
            net.forward(outs, getOutputsNames(net));

            std::vector<cv::Rect> boxes;

            for (auto &out : outs) {
                auto *data = (float *) out.data;
                for (int j = 0; j < out.rows; ++j, data += out.cols) {
                    cv::Mat scores = out.row(j).colRange(5, out.cols);
                    cv::Point classIdPoint;
                    double confidence;
                    cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
                    if (confidence > 0.5) {
                        int centerX = (int) (data[0] * frame.cols);
                        int centerY = (int) (data[1] * frame.rows);
                        int width = (int) (data[2] * frame.cols);
                        int height = (int) (data[3] * frame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        boxes.emplace_back(left, top, width, height);
                    }
                }
            }

            std::cout << "Detect some objects" << std::endl;

            multiTracker = cv::MultiTracker::create();
            direction.clear();
            differenceBetweenCounting = 0;
            for (const auto &box : boxes) {
                multiTracker->add(cv::TrackerCSRT::create(), frame, cv::Rect(box));
                direction.emplace_back(box.x);
            }
        } else {
            multiTracker->update(frame);
            for (const auto &i : multiTracker->getObjects()) {
                rectangle(frame, i, cv::Scalar(0, 0, 0), 2, 1);
                for (auto dir : direction) {
                    if (differenceBetweenCounting == 3) {
                        dir - i.x < 0
                        ? inCount++
                        : outCount++;
                    }
                }
            }
            differenceBetweenCounting++;

            imshow("MultiTracker", frame);

            std::cout << "Show what tracking" << std::endl;
            std::cout << inCount << ' ' << outCount << std::endl;
        }
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    return 0;
}

std::vector<std::string> getOutputsNames(const cv::dnn::Net &net) {
    static std::vector<std::string> names;
    if (names.empty()) {
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::vector<std::string> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}