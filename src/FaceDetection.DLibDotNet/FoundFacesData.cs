using System.Collections.Generic;
using DlibDotNet;

namespace FaceDetection.DLibDotNet
{
    public class FoundFacesData
    {
        public List<Rectangle> Faces { get; set; }
        public List<string> Filenames { get; set; }
        public List<Matrix<float>> Descriptors { get; set; }
    }
}