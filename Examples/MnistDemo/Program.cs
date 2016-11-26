﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using ConvNetSharp;
using ConvNetSharp.Layers;
using ConvNetSharp.Training;

namespace MnistDemo
{
    internal class Program
    {
        private const int BatchSize = 3000;
        private readonly Random random = new Random();
        private readonly CircularBuffer<double> trainAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> valAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> wLossWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> xLossWindow = new CircularBuffer<double>(100);
        private Net net;
        private int stepCount;
        private List<MnistEntry> testing;
        private AdadeltaTrainer trainer;
        private List<MnistEntry> training;
        private int trainingCount = BatchSize;

        private const string urlMnist = @"http://yann.lecun.com/exdb/mnist/";
        private const string mnistFolder = @"..\Mnist\";
        private const string trainingLabelFile = "train-labels-idx1-ubyte.gz";
        private const string trainingImageFile = "train-images-idx3-ubyte.gz";
        private const string testingLabelFile = "t10k-labels-idx1-ubyte.gz";
        private const string testingImageFile = "t10k-images-idx3-ubyte.gz";

        int[] numbers = new int[] { 0, 1, 2, 3/*, 4, 5, 6, 7, 8, 9*/ };

        private void MnistDemo()
        {
            Directory.CreateDirectory(mnistFolder);

            string trainingLabelFilePath = Path.Combine(mnistFolder, trainingLabelFile);
            string trainingImageFilePath = Path.Combine(mnistFolder, trainingImageFile);
            string testingLabelFilePath = Path.Combine(mnistFolder, testingLabelFile);
            string testingImageFilePath = Path.Combine(mnistFolder, testingImageFile);

            // Download Mnist files if needed
            Console.WriteLine("Downloading Mnist training files...");
            DownloadFile(urlMnist + trainingLabelFile, trainingLabelFilePath);
            DownloadFile(urlMnist + trainingImageFile, trainingImageFilePath);

            Console.WriteLine("Downloading Mnist testing files...");
            DownloadFile(urlMnist + testingLabelFile, testingLabelFilePath);
            DownloadFile(urlMnist + testingImageFile, testingImageFilePath);

            // Load data
            Console.WriteLine("Loading the datasets...");
            
            this.training = MnistReader.Load(trainingLabelFilePath, trainingImageFilePath).Where(p => numbers.Contains(p.Label)).ToList();
            this.testing = MnistReader.Load(testingLabelFilePath, testingImageFilePath).Where(p => numbers.Contains(p.Label)).ToList();

            if (this.training.Count == 0 || this.testing.Count == 0)
            {
                Console.WriteLine("Missing Mnist training/testing files.");
                Console.ReadKey();
                return;
            }

            Console.WriteLine($"datasets training: {this.training.Count}, testing: {this.testing.Count}");

            //ExtractImages();

            var netFile = Path.Combine(mnistFolder, $"net{string.Join("", numbers)}.bin");
            if (File.Exists(netFile))
            {
                Console.WriteLine($"load {netFile}?");
                if (Console.ReadKey(true).Key == ConsoleKey.Enter)
                {
                    Console.WriteLine($"loading...");
                    this.net = Net.Load(netFile);
                }
            }

            if(this.net == null)
            {
                this.net = new Net();
                this.net.AddLayer(new InputLayer(24, 24, 1));
               
                this.net.AddLayer(new ConvLayer(3, 3, 11)
                {
                    Stride = 1,
                    Pad = 2,
                    Activation = Activation.Relu
                });
                this.net.AddLayer(new ConvLayer(3, 3, 11)
                {
                    Stride = 1,
                    Pad = 2,
                    Activation = Activation.Relu
                });
                
                this.net.AddLayer(new SoftmaxLayer(numbers.Length * numbers.Length));
            }

            this.trainer = new AdadeltaTrainer(this.net)
            {
                BatchSize = 20,
                L2Decay = 0.001,
            };

            Console.WriteLine("Training...[Press any key to stop]");

            bool ok = false;
            while (!ok)
            {
                do
                {
                    var sample = this.SampleTrainingInstance();

                    ok = this.Step(sample);
                }
                while (!ok && !Console.KeyAvailable);

                if(!ok)
                Console.ReadKey(true);

                Console.WriteLine($"stop? [ENTER: continue]");
                ok = !(Console.ReadKey(true).Key == ConsoleKey.Enter);
            }

            Console.WriteLine($"save {netFile}?");
            if (Console.ReadKey(true).Key == ConsoleKey.Enter)
            {
                var f = Path.Combine(mnistFolder, netFile);
                Console.WriteLine($"saving...");

                net.Save(f);
            }
            Console.WriteLine("done.");
            Console.ReadKey();
        }

        private void ExtractImages()
        {
            int x = 0;
            foreach (var t in training)
            {
                var dir = Path.Combine(mnistFolder, Path.GetFileNameWithoutExtension(trainingImageFile), $"{t.Label}");
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                var f = Path.Combine(dir, $"{x++}.raw");

                File.WriteAllBytes(f, t.Image);
            }

            x = 0;
            foreach (var t in testing)
            {
                var dir = Path.Combine(mnistFolder, Path.GetFileNameWithoutExtension(testingImageFile), $"{t.Label}");
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                var f = Path.Combine(dir, $"{x++}.raw");

                File.WriteAllBytes(f, t.Image);
            }
        }

        private void DownloadFile(string urlFile, string destFilepath)
        {
            if (!File.Exists(destFilepath))
            {
                try
                {
                    using (var client = new WebClient())
                    {
                        client.DownloadFile(urlFile, destFilepath);
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("Failed downloading " + urlFile);
                    Console.WriteLine(e.Message);
                }
            }
        }

        private bool Step(Item sample)
        {
            var x = sample.Volume;
            var y = sample.Label;

            if (sample.IsValidation)
            {
                // use x to build our estimate of validation error
                this.net.Forward(x);
                var yhat = this.net.GetPrediction();
                var valAcc = yhat == y ? 1.0 : 0.0;
                this.valAccWindow.Add(valAcc);
                return false;
            }

            // train on it with network
            this.trainer.Train(x, y);
            var lossx = this.trainer.CostLoss;
            var lossw = this.trainer.L2DecayLoss;

            // keep track of stats such as the average training error and loss
            var prediction = this.net.GetPrediction();
            var trainAcc = prediction == y ? 1.0 : 0.0;
            this.xLossWindow.Add(lossx);
            this.wLossWindow.Add(lossw);
            this.trainAccWindow.Add(trainAcc);

            if (this.stepCount % 50 == 0)
            {
                if (this.xLossWindow.Count == this.xLossWindow.Capacity)
                {
                    var xa = this.xLossWindow.Items.Average();
                    var xw = this.wLossWindow.Items.Average();
                    var loss = xa + xw;

                    var trainAccu = Math.Round(this.trainAccWindow.Items.Average() * 100.0, 2);
                    var testAccu = Math.Round(this.valAccWindow.Items.Average() * 100.0, 2);

                    Console.WriteLine($"Loss: {loss} Train accuracy: {trainAccu}% Test accuracy: {testAccu}%");

                    Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this.stepCount,
                        Math.Round(this.trainer.ForwardTime.TotalMilliseconds, 2),
                        Math.Round(this.trainer.BackwardTime.TotalMilliseconds, 2));

                    Console.WriteLine();

                    if (testAccu >= 99)
                        return true;
                }
            }

            if (this.stepCount % 1000 == 0)
            {
                //this.TestPredict();
            }

            this.stepCount++;

            return false;
        }

        private void TestPredict()
        {
            for (var i = 0; i < 50; i++)
            {
                List<Item> sample = this.SampleTestingInstance();
                var y = sample[0].Label; // ground truth label

                // forward prop it through the network
                var average = new Volume(1, 1, numbers.Length, 0.0);
                var n = sample.Count;
                for (var j = 0; j < n; j++)
                {
                    var a = this.net.Forward(sample[j].Volume);
                    average.AddFrom(a);
                }

                var predictions = average.Weights.Select((w, k) => new { Label = k, Weight = w }).OrderBy(o => -o.Weight);
            }
        }

        private Item SampleTrainingInstance()
        {
            var n = this.random.Next(this.trainingCount);
            var entry = this.training[n];

            // load more batches over time
            if (this.stepCount % 5000 == 0 && this.stepCount > 0)
            {
                this.trainingCount = Math.Min(this.trainingCount + BatchSize, this.training.Count);
            }

            // Create volume from image data
            var x = new Volume(28, 28, 1, 0.0);

            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    x.Weights[j + i * 28] = entry.Image[j + i * 28] / 255.0;
                }
            }

            x = x.Augment(24);

            return new Item { Volume = x, Label = entry.Label, IsValidation = n % 10 == 0 };
        }

        private List<Item> SampleTestingInstance()
        {
            var result = new List<Item>();
            var n = this.random.Next(this.testing.Count);
            var entry = this.testing[n];

            // Create volume from image data
            var x = new Volume(28, 28, 1, 0.0);

            for (var i = 0; i < 28; i++)
            {
                for (var j = 0; j < 28; j++)
                {
                    x.Weights[j + i * 28] = entry.Image[j + i * 28] / 255.0;
                }
            }

            for (var i = 0; i < 4; i++)
            {
                result.Add(new Item { Volume = x.Augment(24), Label = entry.Label });
            }

            return result;
        }

        private static void Main(string[] args)
        {
            var program = new Program();
            program.MnistDemo();
        }

        private class Item
        {
            public Volume Volume
            {
                get; set;
            }

            public int Label
            {
                get; set;
            }

            public bool IsValidation
            {
                get; set;
            }
        }
    }
}