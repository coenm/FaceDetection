using System.Collections.Generic;
using Core.Persistence;

namespace FaceDetection.DLibDotNet
{
    public class FoundFacesDataDto
    {
        public List<RectangleDto> Faces { get; set; }

        public List<string> Filenames { get; set; }

        public List<MatrixFloatDto> Descriptors { get; set; }
    }
}