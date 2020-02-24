using System.Threading.Tasks;
using Core.Persistence;
using FaceDetection.OpenCv.OpenCvHelpers;

namespace FaceDetection.OpenCv
{
    using System.Collections.Generic;
    using System.Linq;
    using System.IO;
    using Core;
    using OpenCvSharp;
    using OpenCvSharp.Dnn;

    /// <summary>
    /// To run this example first download the face model available here:https://github.com/spmallick/learnopencv/tree/master/FaceDetectionComparison/models
    /// Add the files to the bin folder
    /// </summary>
    public class OpenCvDnnCaffe : IFaceDetection
    {
        const string ConfigFile = "model/deploy.prototxt";
        const string FaceModel = "model/res10_300x300_ssd_iter_140000_fp16.caffemodel";

        public string Name { get; } = "OpenCv-DNN-Caffe";

        public async Task<IEnumerable<Face>> ProcessAsync(string inputFilename)
        {
            if (!File.Exists(inputFilename))
                throw new FileNotFoundException(nameof(inputFilename));

            // Read sample image
            using var frame = Cv2.ImRead(inputFilename);
            var frameHeight = frame.Rows;
            var frameWidth = frame.Cols;

            using var faceNet = CvDnn.ReadNetFromCaffe(ConfigFile, FaceModel);
            using var blob = CvDnn.BlobFromImage(
                frame,
                1.0,
                new Size(300, 300),
                new Scalar(104, 117, 123),
                false,
                false);

            faceNet.SetInput(blob, "data");

            using var detection = faceNet.Forward("detection_out");
            using var detectionMat = new Mat(detection.Size(2), detection.Size(3), MatType.CV_32F, detection.Ptr(0));

            var list = new List<ConfidenceBox>(detectionMat.Rows);

            for (var i = 0; i < detectionMat.Rows; i++)
            {
                var confidence = detectionMat.At<float>(i, 2);
                var x1 = (int) (detectionMat.At<float>(i, 3) * frameWidth);
                var y1 = (int) (detectionMat.At<float>(i, 4) * frameHeight);
                var x2 = (int) (detectionMat.At<float>(i, 5) * frameWidth);
                var y2 = (int) (detectionMat.At<float>(i, 6) * frameHeight);

                list.Add(new ConfidenceBox(new Point(x1, y1), new Point(x2, y2), confidence));
            }

            // var orderedFaces = list.OrderByDescending(x => x.Confidence).Where(x => x.Confidence > 0.3).ToList();
            // var origFilename = new FileInfo(inputFilename).Name;

            // var faces = orderedFaces;
            // foreach (var face in faces)
            // {
            //     FaceBoxer.Draw(frame, face.P1, face.P2, face.Confidence);
            // }

            // var outputFilename = Path.Combine(outputDirectory, $"{origFilename}_{Name}.jpg");
            // Cv2.ImWrite(outputFilename, frame);

            // for (var i = 0; i < orderedList.Count; i++)
            // {
            //     var box = orderedList[i];
            //     FaceBoxer.Draw(frame, box.P1, box.P2, box.Confidence);
            //     var outputFilename = Path.Combine(outputDirectory, $"{origFilename}_{Name}_{i+1}.jpg");
            //     Cv2.ImWrite(outputFilename, frame);
            // }

            await Task.Yield();

            return list.Select(x => new Face
                {
                    Position = new RectangleDto
                    {
                        Top = x.P1.Y,
                        Right = x.P2.X,
                        Left = x.P1.X,
                        Bottom = x.P2.Y,
                    }.ToRectangle(),
                    Confidence = x.Confidence,
                })
                .ToList();
        }
    }
}
