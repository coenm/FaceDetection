using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Core;
using DlibDotNet;
using DlibDotNet.Dnn;
using FaceDetection.DLibDotNet.Helpers;

namespace FaceDetection.DLibDotNet
{
    public class DLibFaceIdentificationService
    {
        private readonly IImageRotationService imageRotationService;
        private readonly ShapePredictor predictor;
        private readonly LossMetric dnn;

        public DLibFaceIdentificationService(IImageRotationService imageRotationService)
        {
            this.imageRotationService = imageRotationService ?? throw new ArgumentNullException(nameof(imageRotationService));

            // set up a 5-point landmark detector
            predictor = ShapePredictor.Deserialize("model/shape_predictor_5_face_landmarks.dat");

            // set up a neural network for face recognition
            dnn = DlibDotNet.Dnn.LossMetric.Deserialize("model/dlib_face_recognition_resnet_model_v1.dat");
        }

        public List<List<int>> ClusterFaces(MatrixFloatDto[] faces)
        {
            var convertedFaces = faces
                .Select(d => new Matrix<float>(d.Data, d.Row, d.Columns))
                .ToArray();

            // compare each face with all other faces
            var edges = new List<SamplePair>();
            for (var i = 0; i < convertedFaces.Length; ++i)
            {
                for (var j = i; j < convertedFaces.Length; ++j)
                {
                    // record every pair of two similar faces
                    // faces are similar if they are less than 0.6 apart in the 128D embedding space
                    if (Dlib.Length(convertedFaces[i] - convertedFaces[j]) < 0.4)
                        edges.Add(new SamplePair((uint)i, (uint)j));
                }
            }

            // use the chinese whispers algorithm to find all face clusters
            Dlib.ChineseWhispers(edges, 100, out var foundClusterCount, out var outputLabels);

            var clusters = new List<List<int>>((int)foundClusterCount);
            for (var i = 0; i < foundClusterCount; i++)
                clusters.Add(new List<int>());

            for (var index = 0; index < faces.Length; index++)
            {
                var foundCluster = (int)outputLabels[index];
                clusters[foundCluster].Add(index);
            }

            return clusters;
        }

        public async Task<MatrixFloatDto[]> GetFaceDescriptors(string filename, System.Drawing.Rectangle[] faces)
        {
            var inputFilename = filename;
            var chips = new List<Matrix<RgbPixel>>();

            using var img = await DlibHelpers.LoadRotatedImage(imageRotationService, inputFilename);

            foreach (var face in faces.Select(x => new Rectangle(x.Left, x.Top, x.Right, x.Bottom)))
            {
                // detect landmarks
                var shape = predictor.Detect(img, face);

                // extract normalized and rotated 150x150 face chip
                var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                // convert the chip to a matrix and store
                var matrix = new Matrix<RgbPixel>(faceChip);
                chips.Add(matrix);
            }

            if (!chips.Any())
                return Array.Empty<MatrixFloatDto>();

            // put each fae in a 128D embedding space
            // similar faces will be placed close together
            var descriptors = dnn.Operator(chips);

            return descriptors
                .Select(x => new MatrixFloatDto
                {
                    Data = x.ToArray(),
                    Row = x.Rows,
                    Columns = x.Columns,
                })
                .ToArray();
        }
    }
}