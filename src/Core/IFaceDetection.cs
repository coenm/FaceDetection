using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Core.Persistence;

namespace Core
{
    public interface IFaceDetection
    {
        string Name { get; }

        Task<IEnumerable<Face>> ProcessAsync(string inputFilename);
    }
}
