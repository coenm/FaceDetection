namespace FaceDetection.OpenCv
{
    using System.IO;
    using Core;
    using OpenCvSharp;
    using OpenCvSharp.Dnn;

    /// <summary>
    /// To run this example first download the face model available here:https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models
    /// Add the files to the bin folder
    /// </summary>
    public class OpenCvDnnTensorflow : IFaceDetection
    {
        const string TensorflowConfigFile = "model/opencv_face_detector.pbtxt";
        const string TensorflowWeightFile = "model/opencv_face_detector_uint8.pb";

        public string Name { get; } = "OpenCv-DNN-Tensorflow";

        public int Process(string inputFilename, string outputDirectory)
        {
            if (!File.Exists(inputFilename))
                throw new FileNotFoundException(nameof(inputFilename));

            // Read sample image
            using var frame = Cv2.ImRead(inputFilename);
            int frameHeight = frame.Rows;
            int frameWidth = frame.Cols;

            using var faceNet = CvDnn.ReadNetFromTensorflow(TensorflowWeightFile, TensorflowConfigFile);
            using var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(300, 300),
                new Scalar(104, 117, 123), false, false);
            faceNet.SetInput(blob, "data");

            using var detection = faceNet.Forward("detection_out");
            using var detectionMat = new Mat(detection.Size(2), detection.Size(3), MatType.CV_32F, detection.Ptr(0));

            var found = 0;
            for (int i = 0; i < detectionMat.Rows; i++)
            {
                float confidence = detectionMat.At<float>(i, 2);

                if (confidence > 0.15)
                {
                    found++;
                    int x1 = (int)(detectionMat.At<float>(i, 3) * frameWidth);
                    int y1 = (int)(detectionMat.At<float>(i, 4) * frameHeight);
                    int x2 = (int)(detectionMat.At<float>(i, 5) * frameWidth);
                    int y2 = (int)(detectionMat.At<float>(i, 6) * frameHeight);

                    Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 2, LineTypes.Link4);
                }
            }

            var origFilename = new FileInfo(inputFilename).Name;
            var outputFilename = Path.Combine(outputDirectory, $"{origFilename}_{Name}.jpg");

            Cv2.ImWrite(outputFilename, frame);

            return found;
        }
    }
}
