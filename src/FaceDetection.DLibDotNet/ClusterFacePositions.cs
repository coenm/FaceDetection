using System;
using System.Collections.Generic;
using System.Linq;
using DlibDotNet;

namespace FaceDetection.DLibDotNet
{
    public static class ClusterFacePositions
    {
        public class Matches
        {
            public int? Set1Index { get; private set; }
            public int? Set2Index { get; private set; }
            public int? Set3Index { get; private set; }

            public void Set(int index, int value)
            {
                switch (index)
                {
                    case 1:
                        Set1Index = value;
                        break;
                    case 2:
                        Set2Index = value;
                        break;
                    case 3:
                        Set3Index = value;
                        break;

                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
        }

        public static List<Matches> Abc(List<System.Drawing.Rectangle> set1, List<System.Drawing.Rectangle> set2, List<System.Drawing.Rectangle> set3)
        {
            var totalset = set1.Concat(set2).Concat(set3).ToList();

            var matches = new List<SamplePair>();

            for (var i = 0; i < totalset.Count; i++)
                matches.Add(new SamplePair((uint)i, (uint)i, 0d));

            for (var i = 0; i < set1.Count; i++)
            {
                for (var j = set1.Count; j < totalset.Count; j++)
                {
                    var intersection = System.Drawing.Rectangle.Intersect(totalset[i], totalset[j]);
                    if (intersection.IsEmpty)
                        continue;

                    var intersectionSize = intersection.Width * intersection.Height;
                    var unionSize = totalset[i].Width * totalset[i].Height;
                    unionSize += (totalset[j].Width * totalset[j].Height);
                    unionSize -= intersectionSize;
                    var iou = (double)intersectionSize / (double)unionSize;

                    if (iou > 0.7)
                        matches.Add(new SamplePair((uint)i, (uint)(j), 1 - iou));
                }
            }


            for (var i = set1.Count; i < set1.Count + set2.Count; i++)
            {
                for (var j = set1.Count + set2.Count; j < totalset.Count; j++)
                {
                    var intersection = System.Drawing.Rectangle.Intersect(totalset[i], totalset[j]);
                    if (intersection.IsEmpty)
                        continue;

                    var intersectionSize = intersection.Width * intersection.Height;
                    var unionSize = totalset[i].Width * totalset[i].Height;
                    unionSize += (totalset[j].Width * totalset[j].Height);
                    unionSize -= intersectionSize;
                    var iou = (double)intersectionSize / (double)unionSize;

                    if (iou > 0.7)
                        matches.Add(new SamplePair((uint)i, (uint)(j), 1 - iou));
                }
            }

            // use the chinese whispers algorithm to find all face clusters
            Dlib.ChineseWhispers(matches, 100, out var clusters2, out var labels2);

            var result = new List<Matches>((int)clusters2);
            for (int i = 0; i < clusters2; i++)
            {
                result.Add(new Matches());
            }

            for (int i = 0; i < set1.Count; i++)
            {
                var j = i;
                var label = (int) labels2[j];
                result[label].Set(1, i);
            }

            for (int i = 0; i < set2.Count; i++)
            {
                var j = i + set1.Count;
                var label = (int)labels2[j];
                result[label].Set(2, i);
            }

            for (int i = 0; i < set3.Count; i++)
            {
                var j = i + set1.Count + set2.Count;
                var label = (int)labels2[j];
                result[label].Set(3, i);
            }

            return result;
        }
    }
}