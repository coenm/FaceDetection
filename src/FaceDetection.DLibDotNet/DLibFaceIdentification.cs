using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Core;
using Core.Persistence;
using DlibDotNet;
using DlibDotNet.Dnn;
using FaceDetection.DLibDotNet.Helpers;
using Newtonsoft.Json;

namespace FaceDetection.DLibDotNet
{
    public class FoundFacesData
    {
        public List<Rectangle> Faces { get; set; }
        public List<string> Filenames { get; set; }
        public List<Matrix<float>> Descriptors { get; set; }
    }

    public class FoundFacesDataDto
    {
        public List<RectangleDto> Faces { get; set; }

        public List<string> Filenames { get; set; }

        public List<MatrixFloatDto> Descriptors { get; set; }
    }

    public class DLibFaceIdentification : IDisposable
    {
        private readonly IImageRotationService imageRotationService;
        private FrontalFaceDetector detector;
        private ShapePredictor predictor;
        private LossMetric dnn;
        private RgbPixel[] palette;

        public DLibFaceIdentification(IImageRotationService imageRotationService)
        {
            this.imageRotationService = imageRotationService ?? throw new ArgumentNullException(nameof(imageRotationService));
            detector = Dlib.GetFrontalFaceDetector();

            // set up a 5-point landmark detector
            predictor = ShapePredictor.Deserialize("model/shape_predictor_5_face_landmarks.dat");

            // set up a neural network for face recognition
            dnn = DlibDotNet.Dnn.LossMetric.Deserialize("model/dlib_face_recognition_resnet_model_v1.dat");

            // create a color palette for plotting
            palette = new RgbPixel[]
            {
                new RgbPixel(0xe6, 0x19, 0x4b),
                new RgbPixel(0xf5, 0x82, 0x31),
                new RgbPixel(0xff, 0xe1, 0x19),
                new RgbPixel(0xbc, 0xf6, 0x0c),
                new RgbPixel(0x3c, 0xb4, 0x4b),
                new RgbPixel(0x46, 0xf0, 0xf0),
                new RgbPixel(0x43, 0x63, 0xd8),
                new RgbPixel(0x91, 0x1e, 0xb4),
                new RgbPixel(0xf0, 0x32, 0xe6),
                new RgbPixel(0x80, 0x80, 0x80)
            };
        }

        public async Task ProcessAsync(string[] inputFilenames)
        {
            var chips = new List<Matrix<RgbPixel>>();
            var faces = new List<Rectangle>();
            var filename = new List<string>();
            var jsonFilename = inputFilenames.First() + ".json";

            foreach (var inputFilename in inputFilenames)
            {
                if (!File.Exists(inputFilename))
                    break;

                if (File.Exists(jsonFilename))
                    continue;

                // load the image
                using var img = await DlibHelpers.LoadRotatedImage(imageRotationService, inputFilename);
                // Dlib.SaveJpeg(img, inputFilename + "__1.jpg", 25);
                // Dlib.SaveJpeg(img, inputFilename + "__2.jpg", 25);

                // detect all faces
                foreach (var face in detector.Operator(img))
                {
                    // detect landmarks
                    var shape = predictor.Detect(img, face);

                    // extract normalized and rotated 150x150 face chip
                    var faceChipDetail = Dlib.GetFaceChipDetails(shape, 150, 0.25);
                    var faceChip = Dlib.ExtractImageChip<RgbPixel>(img, faceChipDetail);

                    // convert the chip to a matrix and store
                    var matrix = new Matrix<RgbPixel>(faceChip);
                    chips.Add(matrix);
                    faces.Add(face);
                    filename.Add(inputFilename);
                }
            }

            if (!File.Exists(jsonFilename))
            {
                var ffd = new FoundFacesData
                {
                    // Chips = chips,
                    Faces = faces,
                    Filenames = filename,
                };

                OutputLabels<Matrix<float>> descriptors = null;
                if (chips.Any())
                {
                    // put each fae in a 128D embedding space
                    // similar faces will be placed close together
                    // Console.WriteLine("Recognizing faces...");
                    descriptors = dnn.Operator(chips);
                    ffd.Descriptors = descriptors.ToList();
                }
                else
                {
                    ffd.Descriptors = new List<Matrix<float>>(0);
                }

                var dto = new FoundFacesDataDto
                {
                    Faces = ffd.Faces
                        .Select(f => new RectangleDto
                        {
                            Bottom = f.Bottom,
                            Left = f.Left,
                            Top = f.Top,
                            Right = f.Right,
                        })
                        .ToList(),

                    Filenames = ffd.Filenames,

                    Descriptors = ffd.Descriptors
                        .Select(x => new MatrixFloatDto
                        {
                            Data = x.ToArray(),
                            Row = x.Rows,
                            Columns = x.Columns,
                        })
                        .ToList()
                };

                var x = JsonConvert.SerializeObject(dto);
                File.WriteAllText(jsonFilename, JsonConvert.SerializeObject(dto));
            }

            FoundFacesData items;
            using (var r = new StreamReader(jsonFilename))
            {
                var json = r.ReadToEnd();
                var itemsdto = JsonConvert.DeserializeObject<FoundFacesDataDto>(json);
                items = new FoundFacesData
                {
                    Faces = itemsdto.Faces.Select(f => new Rectangle(f.Left, f.Top, f.Right, f.Bottom)).ToList(),
                    Filenames = itemsdto.Filenames.ToList(),
                    Descriptors = itemsdto.Descriptors.Select(d => new Matrix<float>(d.Data, d.Row, d.Columns)).ToList(),
                };
            }

            if (items.Faces.Count <= 0)
                return;

            // // compare each face with all other faces
            var edges = new List<SamplePair>();
            // for (uint i = 0; i < descriptors.Count; ++i)
            // for (var j = i; j < descriptors.Count; ++j)
            //
            //     // record every pair of two similar faces
            //     // faces are similar if they are less than 0.6 apart in the 128D embedding space
            //     if (Dlib.Length(descriptors[i] - descriptors[j]) < 0.5)
            //         edges.Add(new SamplePair(i, j));
            //
            // // use the chinese whispers algorithm to find all face clusters
            // Dlib.ChineseWhispers(edges, 100, out var clusters, out var labels);
            // // Console.WriteLine($"   Found {clusters} unique person(s) in the image");
            //
            // // draw rectangles on each face using the cluster color
            // for (var i = 0; i < faces.Count; i++)
            // {
            //     var color = new RgbPixel(255, 255, 255);
            //     if (labels[i] < palette.Length)
            //         color = palette[labels[i]];
            //
            //     using var img = Dlib.LoadImage<RgbPixel>(filename[i] + "__1.jpg");
            //     Dlib.DrawRectangle(img, faces[i], color: color, thickness: 4);
            //     Dlib.SaveJpeg(img, filename[i] + "__1.jpg", 25);
            // }
            //
            // Console.WriteLine("end 1");

            // compare each face with all other faces
            edges = new List<SamplePair>();
            for (var i = 0; i < items.Descriptors.Count; ++i)
            for (var j = i; j < items.Descriptors.Count; ++j)

                // record every pair of two similar faces
                // faces are similar if they are less than 0.6 apart in the 128D embedding space
                if (Dlib.Length(items.Descriptors[i] - items.Descriptors[j]) < 0.4)
                    edges.Add(new SamplePair((uint)i, (uint)j));

            // use the chinese whispers algorithm to find all face clusters
            Dlib.ChineseWhispers(edges, 100, out var clusters2, out var labels2);
            // Console.WriteLine($"   Found {clusters} unique person(s) in the image");

            // draw rectangles on each face using the cluster color
            for (var i = 0; i < items.Faces.Count; i++)
            {
                var color = palette[0];
                if (labels2[i] < palette.Length)
                    color = palette[labels2[i]];

                if (!File.Exists(items.Filenames[i] + $"_x{labels2[i]}.jpg"))
                {
                    using var img2 = await DlibHelpers.LoadRotatedImage(imageRotationService, items.Filenames[i]);
                    Dlib.SaveJpeg(img2, items.Filenames[i] + $"_x{labels2[i]}.jpg", 25);
                }

                using var img = Dlib.LoadImage<RgbPixel>(items.Filenames[i] + $"_x{labels2[i]}.jpg");
                Dlib.DrawRectangle(img, items.Faces[i], color: color, thickness: 4);
                Dlib.SaveJpeg(img, items.Filenames[i] + $"_x{labels2[i]}.jpg", 25);
            }

            // var origFilename = new FileInfo(inputFilename).Name;
            // var outputFilename = Path.Combine(outputDirectory, $"{origFilename}_Identification.jpg");

            // Dlib.SaveJpeg(img, inputFilename, 75);
        }

        public void Dispose()
        {
            dnn.Dispose();
            predictor.Dispose();
            detector.Dispose();
        }
    }

    public class MatrixFloatDto
    {
        public float[] Data { get; set; }
        public int Row { get; set; }
        public int Columns { get; set; }
    }
}