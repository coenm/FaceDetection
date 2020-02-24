using System.Drawing;
using Core.Persistence;
using FaceDetection.OpenCv.OpenCvHelpers;

namespace FaceDetection.OpenCv
{
    using System.Collections.Generic;
    using System.IO;
    using OpenCvSharp;

    public class OpenCvDrawFaces
    {
        public void Draw(string inputFilename, string outputFilename, Color color, IEnumerable<Face> faces)
        {
            if (!File.Exists(inputFilename))
                throw new FileNotFoundException(nameof(inputFilename));

            // Read sample image
            using var frame = Cv2.ImRead(inputFilename);

            foreach (var face in faces)
            {
                var p1 = new Point(face.Position.Left, face.Position.Top);
                var p2 = new Point(face.Position.Right, face.Position.Bottom);
                FaceBoxer.Draw(frame, p1, p2, face.Confidence ?? 1, color);
            }

            Cv2.ImWrite(outputFilename, frame);
        }
    }
}
