﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Xml;
using ConvNetSharp.Layers;

namespace ConvNetSharp
{
    [DataContract]
    public class Net
    {
        [DataMember]
        public readonly List<LayerBase> Layers = new List<LayerBase>();

        public void AddLayer(LayerBase layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            if (this.Layers.Count > 0)
            {
                inputWidth = this.Layers[this.Layers.Count - 1].OutputWidth;
                inputHeight = this.Layers[this.Layers.Count - 1].OutputHeight;
                inputDepth = this.Layers[this.Layers.Count - 1].OutputDepth;
            }

            var classificationLayer = layer as IClassificationLayer;
            if (classificationLayer != null)
            {
                var fullyConnLayer = new FullyConnLayer(classificationLayer.ClassCount);
                fullyConnLayer.Init(inputWidth, inputHeight, inputDepth);
                inputWidth = fullyConnLayer.OutputWidth;
                inputHeight = fullyConnLayer.OutputHeight;
                inputDepth = fullyConnLayer.OutputDepth;

                this.Layers.Add(fullyConnLayer);
            }

            var regressionLayer = layer as RegressionLayer;
            if (regressionLayer != null)
            {
                var fullyConnLayer = new FullyConnLayer(regressionLayer.NeuronCount);
                fullyConnLayer.Init(inputWidth, inputHeight, inputDepth);
                inputWidth = fullyConnLayer.OutputWidth;
                inputHeight = fullyConnLayer.OutputHeight;
                inputDepth = fullyConnLayer.OutputDepth;

                this.Layers.Add(fullyConnLayer);
            }

            var dotProductLayer = layer as IDotProductLayer;
            if (dotProductLayer != null)
            {
                if (dotProductLayer.Activation == Activation.Relu)
                {
                    dotProductLayer.BiasPref = 0.1; // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.
                }
            }

            if (this.Layers.Count > 0)
            {
                layer.Init(inputWidth, inputHeight, inputDepth);
            }

            this.Layers.Add(layer);

            if (dotProductLayer != null)
            {
                switch (dotProductLayer.Activation)
                {
                    case Activation.Undefined:
                        break;
                    case Activation.Relu:
                        var reluLayer = new ReluLayer();
                        reluLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.Layers.Add(reluLayer);
                        break;
                    case Activation.Sigmoid:
                        var sigmoidLayer = new SigmoidLayer();
                        sigmoidLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.Layers.Add(sigmoidLayer);
                        break;
                    case Activation.Tanh:
                        var tanhLayer = new TanhLayer();
                        tanhLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.Layers.Add(tanhLayer);
                        break;
                    case Activation.Maxout:
                        var maxoutLayer = new MaxoutLayer { GroupSize = dotProductLayer.GroupSize };
                        maxoutLayer.Init(layer.OutputWidth, layer.OutputHeight, layer.OutputDepth);
                        this.Layers.Add(maxoutLayer);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }

            var lastLayer = this.Layers[this.Layers.Count - 1];

            if (!(layer is DropOutLayer) && layer.DropProb.HasValue)
            {
                var dropOutLayer = new DropOutLayer(layer.DropProb.Value);
                dropOutLayer.Init(lastLayer.OutputWidth, lastLayer.OutputHeight, lastLayer.OutputDepth);
                this.Layers.Add(dropOutLayer);
            }
        }

        public Volume Forward(Volume volume, bool isTraining = false)
        {
            var activation = this.Layers[0].Forward(volume, isTraining);

            for (var i = 1; i < this.Layers.Count; i++)
            {
                var layerBase = this.Layers[i];
                activation = layerBase.Forward(activation, isTraining);
            }

            return activation;
        }

        public double GetCostLoss(Volume volume, double y)
        {
            this.Forward(volume);

            var lastLayer = this.Layers[this.Layers.Count - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double GetCostLoss(Volume volume, double[] y)
        {
            this.Forward(volume);

            var lastLayer = this.Layers[this.Layers.Count - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y);
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double Backward(double y)
        {
            var n = this.Layers.Count;
            var lastLayer = this.Layers[n - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y); // last layer assumed to be loss layer
                for (var i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input
                    this.Layers[i].Backward();
                }
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public double Backward(double[] y)
        {
            var n = this.Layers.Count;
            var lastLayer = this.Layers[n - 1] as ILastLayer;
            if (lastLayer != null)
            {
                var loss = lastLayer.Backward(y); // last layer assumed to be loss layer
                for (var i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input
                    this.Layers[i].Backward();
                }
                return loss;
            }

            throw new Exception("Last layer doesnt implement ILastLayer interface");
        }

        public int GetPrediction()
        {
            // this is a convenience function for returning the argmax
            // prediction, assuming the last layer of the net is a softmax
            var softmaxLayer = this.Layers[this.Layers.Count - 1] as SoftmaxLayer;
            if (softmaxLayer == null)
            {
                throw new Exception("GetPrediction function assumes softmax as last layer of the net!");
            }

            double[] p = softmaxLayer.OutputActivation.Weights;
            var maxv = p[0];
            var maxi = 0;

            for (var i = 1; i < p.Length; i++)
            {
                if (p[i] > maxv)
                {
                    maxv = p[i];
                    maxi = i;
                }
            }

            return maxi; // return index of the class with highest class probability
        }

        public List<ParametersAndGradients> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients>();

            foreach (LayerBase t in Layers)
            {
                List<ParametersAndGradients> parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }

        public void Save(string fileName)
        {
            using (FileStream fs = new FileStream(fileName, FileMode.Create))
            {
                using (XmlDictionaryWriter bdw = XmlDictionaryWriter.CreateBinaryWriter(fs))
                {
                    var ser = new DataContractSerializer(typeof(Net));
                    ser.WriteObject(bdw, this);
                    bdw.Flush();
                }                               
            }
        }

        public static Net Load(string fileName)
        {
            Net ret;
            using (FileStream fs = new FileStream(fileName, FileMode.Open))
            {
                var q = new XmlDictionaryReaderQuotas()
                {
                    MaxArrayLength = 1024 * 1024 * 10, // 10MB
                    MaxBytesPerRead = 1024 * 1024 * 10 // 10MB
                };

                using (var reader = XmlDictionaryReader.CreateBinaryReader(fs, q))
                {
                    var ser = new DataContractSerializer(typeof(Net));
                    ret = (Net)ser.ReadObject(reader, true);
                }
            }
            return ret;
        }
    }
}