using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System.Reflection;

namespace Company.Function
{
    public static class HttpTrigger1
    {
        [FunctionName("HttpTrigger1")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Function, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");
            log.LogInformation("Current DIR: "+Environment.CurrentDirectory);
            var runDir = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
            log.LogInformation("Assembly DIR: "+runDir);


            const string configFile = "C:\\home\\site\\wwwroot\\deploy.prototxt";
            const string faceModel = "C:\\home\\site\\wwwroot\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
            using var faceNet = CvDnn.ReadNetFromCaffe(configFile, faceModel);
            using (Stream stream = req.Body)
            {
                using (MemoryStream mStream = new MemoryStream())
                {
                    stream.CopyTo(mStream);
                    var imgBytes = mStream.ToArray();
                    using (Mat image = Mat.FromImageData(imgBytes, ImreadModes.Unchanged))
                    {
                        var newSize = new Size(640, 480);
                        using var frame = new Mat();
                        Cv2.Resize(image, frame, newSize);

                        int frameHeight = frame.Rows;
                        int frameWidth = frame.Cols;

                        using var blob = CvDnn.BlobFromImage(frame, 1.0, new Size(300, 300),
                            new Scalar(104, 117, 123), false, false);
                        faceNet.SetInput(blob, "data");

                        using var detection = faceNet.Forward("detection_out");
                        using var detectionMat = new Mat(detection.Size(2), detection.Size(3), MatType.CV_32F,
                            detection.Ptr(0));
                        for (int i = 0; i < detectionMat.Rows; i++)
                        {
                            float confidence = detectionMat.At<float>(i, 2);

                            if (confidence > 0.7)
                            {
                                int x1 = (int)(detectionMat.At<float>(i, 3) * frameWidth);
                                int y1 = (int)(detectionMat.At<float>(i, 4) * frameHeight);
                                int x2 = (int)(detectionMat.At<float>(i, 5) * frameWidth);
                                int y2 = (int)(detectionMat.At<float>(i, 6) * frameHeight);
                                Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), Scalar.Green);

                                // create a new Mat with the detected face
                                var faceImg = new Mat(frame,
                                    new OpenCvSharp.Range(y1, y2),
                                    new OpenCvSharp.Range(x1, x2));

                                // blur the face area in the original frame
                                var faceBlur = new Mat();
                                Cv2.GaussianBlur(faceImg, faceBlur, new Size(23, 23), 30);
                                frame[new OpenCvSharp.Range(y1, y2), new OpenCvSharp.Range(x1, x2)] = faceBlur;
                            }
                        }

                        return new FileContentResult(frame.ToBytes(), "application/octet-stream")
                        {
                            FileDownloadName = "blurred.png"
                        };
                    }
                }
            }
            // return new OkObjectResult(responseMessage);
        }
    }
}
