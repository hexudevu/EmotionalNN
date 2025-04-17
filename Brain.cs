using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.VisualScripting;
using UnityEngine;
using static Brain;
using System.Linq;
using static Brain.Memory;

[System.Serializable]
public class BrainSaveData
{
    public struct INeuronSaveData
    {
        public float[] weights;
        public float bias;
        public float[] lastInputs;
        public float lastSum;
        public float lastOutput;
    }
    public struct ENeuronSaveData
    {
        public float dopamine;
        public float serotonin;
        public float adrenaline;
    }
    public struct MemorySaveData
    {
        public Queue<ExperienceSaveData> _buffer;
        public int _capacity;
    }
    public struct ExperienceSaveData
    {
        public float[] State;
        public float[] Action;
        public float Reward;
    }

    public int IQ, EQ, memory_level;
    public float relateIQ, relateEQ;
    public float previousLoss;
    public int inputCount, outputCount;
    public int hiddenLayersCount;
    public int neuronsCountInHiddenLayer;
    public ENeuronSaveData[] emotionsPerLayer;
    public float[] inputs;
    public INeuronSaveData[][] hiddenLayers;
    public INeuronSaveData[] outputLayer;
    public MemorySaveData memory;
    public float gamma;
    public float explorationRate;
    public float explorationMin;
    public float explorationDecay;
}

public class Brain
{
    public int IQ, EQ;
    public int memory_level;
    public float relateIQ, relateEQ;
    public float previousLoss;
    public int inputCount, outputCount;
    public int hiddenLayersCount;
    public int neuronsCountInHiddenLayer;
    public List<ENeuron> emotionsPerLayer;
    public float[] inputs;
    public List<INeuron[]> hiddenLayers;
    public INeuron[] outputLayer;
    public Memory memory;
    public float gamma = 0.99f;
    public float explorationRate = 1.0f;
    public float explorationMin = 0.01f;
    public float explorationDecay = 0.995f;


    //==UTILS==
    static float randomWeight()
    {
        return UnityEngine.Random.Range(-1f, 1f);
    }
    static float dot(float[] a, float[] b)
    {
        float result = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }
    static float Tanh(float x)
    {
        if (x > 20f) return 1f;
        if (x < -20f) return -1f;
        float e1 = math.exp(x);
        float e2 = math.exp(-x);
        return (e1 - e2) / (e1 + e2);
    }
    static float TanhDerivative(float x)
    {
        float t = Tanh(x);
        return 1f - t * t;
    }
    static float HardTanh(float x)
    {
        if (x > 1) return 1f;
        if (x < -1) return -1f;
        return x;
    }
    public class INeuron
    {
        public float[] weights;
        public float bias;
        public float[] lastInputs;
        public float lastSum;
        public float lastOutput;

        public float Activate(float[] inputs)
        {
            lastInputs = inputs;
            lastSum = dot(weights, inputs) + bias;
            lastOutput = Tanh(lastSum);
            return lastOutput;
        }
    }
    public class ENeuron
    {
        public float dopamine;
        public float serotonin;
        public float adrenaline;
        public ENeuron()
        {
            dopamine = 1;
            serotonin = 0;
            adrenaline = 0;
        }
        public float Get(float x)
        {
            return HardTanh(x);
        }
        public float GetMotivation()
        {
            return Get((dopamine - serotonin) * (adrenaline + .77f));
        }
    }
    public class Memory
    {
        public struct Experience
        {
            public float[] State;
            public float[] Action;
            public float Reward;
        }

        public Queue<Experience> _buffer = new Queue<Experience>();
        public int _capacity = 1000;

        public void Add(Experience exp)
        {
            if (_buffer.Count >= _capacity)
                _buffer.Dequeue();
            _buffer.Enqueue(exp);
        }

        public Experience[] Sample(int batchSize)
        {
            return _buffer.OrderBy(x => UnityEngine.Random.value).Take(batchSize).ToArray();
        }
    }


    //==STRUCT==
    public Brain(int iq, int eq, int input, int output, int countOfHiddenLayers = 2, int countOfNeuronsInHiddenLayer = 8)
    {
        IQ = iq;
        EQ = eq;
        memory_level = countOfHiddenLayers * 10 + countOfNeuronsInHiddenLayer;
        memory = new Memory();
        relateIQ = (float)IQ / EQ;
        relateEQ = (float)EQ / IQ;
        inputCount = input;
        outputCount = output;
        hiddenLayersCount = countOfHiddenLayers;
        neuronsCountInHiddenLayer = countOfNeuronsInHiddenLayer;
        inputs = new float[inputCount];
        emotionsPerLayer = new List<ENeuron>();
        hiddenLayers = new List<INeuron[]>();
        for (int i = 0; i < hiddenLayersCount; i++)
        {
            int inputSize = i == 0 ? inputCount : neuronsCountInHiddenLayer;
            INeuron[] layer = new INeuron[neuronsCountInHiddenLayer];

            for (int j = 0; j < neuronsCountInHiddenLayer; j++)
            {
                layer[j] = new INeuron();
                layer[j].weights = new float[inputSize];
                layer[j].bias = randomWeight();
                for (int k = 0; k < inputSize; k++)
                    layer[j].weights[k] = randomWeight();
            }

            hiddenLayers.Add(layer);
            emotionsPerLayer.Add(new ENeuron());

        }

        //выходной слой
        outputLayer = new INeuron[outputCount];
        for (int i = 0; i < outputCount; i++)
        {
            outputLayer[i] = new INeuron();
            outputLayer[i].weights = new float[neuronsCountInHiddenLayer];
            outputLayer[i].bias = randomWeight();
            for (int j = 0; j < neuronsCountInHiddenLayer; j++)
                outputLayer[i].weights[j] = randomWeight();
        }
    }
    public Brain(BrainSaveData data)
    {
        IQ = data.IQ;
        EQ = data.EQ;
        memory_level = data.memory_level;
        relateIQ = data.relateIQ;
        relateEQ = data.relateEQ;
        previousLoss = data.previousLoss;
        inputCount = data.inputCount;
        outputCount = data.outputCount;
        hiddenLayersCount = data.hiddenLayersCount;
        neuronsCountInHiddenLayer = data.neuronsCountInHiddenLayer;

        inputs = (float[])data.inputs.Clone();
        emotionsPerLayer = new List<ENeuron>();
        foreach (BrainSaveData.ENeuronSaveData eNeuronSaveData in data.emotionsPerLayer)
            emotionsPerLayer.Add(new ENeuron()
            {
                dopamine = eNeuronSaveData.dopamine,
                serotonin = eNeuronSaveData.serotonin,
                adrenaline = eNeuronSaveData.adrenaline
            });

        hiddenLayers = new List<INeuron[]>();
        for (int i = 0; i < data.hiddenLayers.Length; i++)
        {
            INeuron[] layer = new INeuron[neuronsCountInHiddenLayer];
            for (int j = 0; j < neuronsCountInHiddenLayer; j++)
            {
                BrainSaveData.INeuronSaveData neuronSaveData = data.hiddenLayers[i][j];
                layer[j] = new INeuron()
                {
                    weights = (float[])neuronSaveData.weights.Clone(),
                    bias = neuronSaveData.bias,
                    lastInputs = (float[])neuronSaveData.lastInputs.Clone(),
                    lastSum = neuronSaveData.lastSum,
                    lastOutput = neuronSaveData.lastOutput
                };
            }
            hiddenLayers.Add(layer);
        }

        outputLayer = new INeuron[outputCount];
        for (int i = 0; i < outputCount; i++)
        {
            BrainSaveData.INeuronSaveData neuronSaveData = data.outputLayer[i];
            var neuron = new INeuron()
            {
                weights = (float[])neuronSaveData.weights.Clone(),
                bias = neuronSaveData.bias,
                lastInputs = (float[])neuronSaveData.lastInputs.Clone(),
                lastSum = neuronSaveData.lastSum,
                lastOutput = neuronSaveData.lastOutput
            };
            outputLayer[i] = neuron;
        }

        Queue<Experience> loadBuffer = new Queue<Experience>();
        foreach (BrainSaveData.ExperienceSaveData expSaveData in data.memory._buffer)
        {
            loadBuffer.Enqueue(new Experience()
            {
                State = (float[])expSaveData.State.Clone(),
                Action = (float[])expSaveData.Action.Clone(),
                Reward = expSaveData.Reward
            });
        }
        memory = new Memory()
        {
            _buffer = loadBuffer,
            _capacity = data.memory._capacity
        };
        gamma = data.gamma;
        explorationMin = data.explorationMin;
        explorationRate = data.explorationRate;
        explorationDecay = data.explorationDecay;
    }
    public BrainSaveData Save()
    {
        var data = new BrainSaveData();
        data.IQ = IQ;
        data.EQ = EQ;
        data.memory_level = memory_level;
        data.relateIQ = IQ;
        data.relateEQ = EQ;
        data.previousLoss = previousLoss;
        data.inputCount = inputCount;
        data.outputCount = outputCount;
        data.hiddenLayersCount = hiddenLayersCount;
        data.neuronsCountInHiddenLayer = neuronsCountInHiddenLayer;

        data.emotionsPerLayer = new BrainSaveData.ENeuronSaveData[emotionsPerLayer.Count];
        for (int i = 0; i < emotionsPerLayer.Count; i++)
        {
            ENeuron eNeuron = emotionsPerLayer[i];
            data.emotionsPerLayer[i] = new BrainSaveData.ENeuronSaveData()
            {
                dopamine = eNeuron.dopamine,
                serotonin = eNeuron.serotonin,
                adrenaline = eNeuron.adrenaline
            };
        }

        data.inputs = (float[])inputs.Clone();

        data.hiddenLayers = new BrainSaveData.INeuronSaveData[hiddenLayersCount][];
        for (int i = 0; i < hiddenLayers.Count; i++)
        {
            INeuron[] layer = hiddenLayers[i];
            BrainSaveData.INeuronSaveData[] dataLayer = new BrainSaveData.INeuronSaveData[layer.Length];
            for (int j = 0; j < dataLayer.Length; j++)
            {
                INeuron neuron = layer[j];
                dataLayer[j] = new BrainSaveData.INeuronSaveData()
                {
                    weights = (float[])neuron.weights.Clone(),
                    bias = neuron.bias,
                    lastInputs = (float[])neuron.lastInputs.Clone(),
                    lastSum = neuron.lastSum,
                    lastOutput = neuron.lastOutput
                };
            }
            data.hiddenLayers[i] = dataLayer;
        }

        data.outputLayer = new BrainSaveData.INeuronSaveData[outputCount];
        for (int i = 0; i < outputCount; i++)
        {
            INeuron neuron = outputLayer[i];
            data.outputLayer[i] = new BrainSaveData.INeuronSaveData()
            {
                weights = (float[])neuron.weights.Clone(),
                bias = neuron.bias,
                lastInputs = (float[])neuron.lastInputs.Clone(),
                lastSum = neuron.lastSum,
                lastOutput = neuron.lastOutput
            };
        }

        Queue<BrainSaveData.ExperienceSaveData> saveBuffer = new Queue<BrainSaveData.ExperienceSaveData>();
        foreach(Experience exp in memory._buffer)
        {
            saveBuffer.Enqueue(new BrainSaveData.ExperienceSaveData()
            {
                State = (float[])exp.State.Clone(),
                Action = (float[])exp.Action.Clone(),
                Reward = exp.Reward
            });
        }
        data.memory = new BrainSaveData.MemorySaveData()
        {
            _buffer = saveBuffer,
            _capacity = memory._capacity
        };

        data.gamma = gamma;
        data.explorationRate = explorationRate;
        data.explorationMin = explorationMin;
        data.explorationDecay = explorationDecay;

        return data;
    }
    public void SaveToFile(string path)
    {
        try
        {
            BrainSaveData saveData = Save();
            string json = JsonUtility.ToJson(saveData, true);
            string directory = Path.GetDirectoryName(path);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllText(path, json);
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save brain to {path}: {e.Message}");
        }
    }
    public static BrainSaveData Load(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                Debug.LogError($"File not found: {path}");
                return default(BrainSaveData);
            }

            string json = File.ReadAllText(path);
            BrainSaveData data = JsonUtility.FromJson<BrainSaveData>(json);
            return data;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load brain from {path}: {e.Message}");
            return default(BrainSaveData);
        }
    }


    //==METHODS==
    public void DebugBrainState()
    {
        Debug.Log($"IQ: {IQ}, EQ: {EQ}");
        Debug.Log($"Exploration Rate: {explorationRate}");
        foreach (var e in emotionsPerLayer)
        {
            Debug.Log($"D: {e.dopamine}, S: {e.serotonin}, A: {e.adrenaline}");
        }
    }
    public void SetInputs(float[] newInputs)
    {
        inputs = newInputs;
    }
    public float[] ForwardPass()
    {
        float[] currentValues = inputs;
        for (int i = 0; i < hiddenLayers.Count; i++)
        {
            float[] nextValues = new float[hiddenLayers[i].Length];
            INeuron[] layer = hiddenLayers[i];
            ENeuron eNeuron = emotionsPerLayer[i];
            float motivation = eNeuron.GetMotivation();
            for (int j = 0; j < layer.Length; j++)
            {
                INeuron neuron = layer[j];
                float output = neuron.Activate(currentValues);
                nextValues[j] = output * motivation * relateEQ;
                motivation = Mathf.Clamp01(eNeuron.GetMotivation());
                float dropoutRate = 1f - motivation;
                if (UnityEngine.Random.value < dropoutRate)
                    nextValues[j] = 0;
            }
            currentValues = nextValues;
        }
        float[] finalOutputs = new float[outputLayer.Length];
        for (int i = 0; i < outputLayer.Length; i++)
        {
            INeuron neuron = outputLayer[i];
            float output = neuron.Activate(currentValues);
            finalOutputs[i] = output;
        }
        return finalOutputs;
    }
    public void EmergencyEmotionsReset()
    {
        foreach (ENeuron neuron in emotionsPerLayer)
        {
            neuron.dopamine = 1;
            neuron.serotonin = 0;
            neuron.adrenaline = 1;
        }
    }
    public void UpdateEmotions(float dopamine, float serotonin, float adrenaline)
    {
        foreach (ENeuron neuron in emotionsPerLayer)
        {
            neuron.dopamine = Mathf.Clamp(dopamine, -5f, 5f);
            neuron.serotonin = Mathf.Clamp(serotonin, -5f, 5f);
            neuron.adrenaline = Mathf.Clamp(adrenaline, 0, 5f);
        }
    }
    public void ConcreteEmote(int x, float dopamine, float serotonin, float adrenaline)
    {
        ENeuron neuron = emotionsPerLayer[x];
        neuron.dopamine = dopamine;
        neuron.serotonin = serotonin;
        neuron.adrenaline = adrenaline;
    }
    public float[] Think(float[] newInputs)
    {
        SetInputs(newInputs);
        float[] outputs = ForwardPass();
        return outputs;
    }
    public void Train(float[] input, float[] expected, float learningRate)
    {
        SetInputs(input);
        float[] currentValues = inputs;
        List<float[]> layerOutputs = new List<float[]>();
        layerOutputs.Add(currentValues);

        // FORWARD PASS
        for (int i = 0; i < hiddenLayers.Count; i++)
        {
            float[] nextValues = new float[hiddenLayers[i].Length];
            INeuron[] layer = hiddenLayers[i];
            float motivation = emotionsPerLayer[i].GetMotivation();

            for (int j = 0; j < layer.Length; j++)
            {
                float output = layer[j].Activate(currentValues);
                nextValues[j] = output * motivation;
            }

            layerOutputs.Add(nextValues);
            currentValues = nextValues;
        }

        float[] predicted = new float[outputLayer.Length];
        for (int i = 0; i < outputLayer.Length; i++)
            predicted[i] = outputLayer[i].Activate(currentValues);

        // BACKWARD PASS

        //===Output layer deltas===
        float[] outputDeltas = new float[outputCount];
        for (int i = 0; i < outputCount; i++)
        {
            float error = predicted[i] - expected[i];
            outputDeltas[i] = error * TanhDerivative(outputLayer[i].lastSum);
        }

        //===Update output layer weights===
        for (int i = 0; i < outputLayer.Length; i++)
        {
            INeuron neuron = outputLayer[i];
            for (int j = 0; j < neuron.weights.Length; j++)
            {
                neuron.weights[j] -= learningRate * outputDeltas[i] * neuron.lastInputs[j];
            }
            neuron.bias -= learningRate * outputDeltas[i];
        }

        //===backpropagaton===
        float[] nextLayerDelta = outputDeltas;
        INeuron[] nextLayerNeurons = outputLayer;

        for (int layerIndex = hiddenLayers.Count - 1; layerIndex >= 0; layerIndex--)
        {
            INeuron[] layer = hiddenLayers[layerIndex];
            float[] layerDelta = new float[layer.Length];

            for (int i = 0; i < layer.Length; i++)
            {
                float sum = 0f;
                for (int j = 0; j < nextLayerNeurons.Length; j++)
                {
                    sum += nextLayerNeurons[j].weights[i] * nextLayerDelta[j];
                }
                layerDelta[i] = sum * TanhDerivative(layer[i].lastSum);
            }

            for (int i = 0; i < layer.Length; i++)
            {
                float motivation = Mathf.Clamp(emotionsPerLayer[layerIndex].GetMotivation(), 0.1f, 2f);
                float localLr = learningRate * motivation;
                INeuron neuron = layer[i];
                for (int j = 0; j < neuron.weights.Length; j++)
                {
                    neuron.weights[j] -= localLr * layerDelta[i] * neuron.lastInputs[j];
                }
                neuron.bias -= localLr * layerDelta[i];
            }

            nextLayerDelta = layerDelta;
            nextLayerNeurons = layer;
        }
    }
    public void RewardEmotions(float[] predicted, float[] expected)
    {
        float loss = 0f;
        for (int i = 0; i < predicted.Length; i++)
            loss += Mathf.Abs(predicted[i] - expected[i]);

        foreach (var e in emotionsPerLayer)
        {
            if (loss < previousLoss)
                e.dopamine = Mathf.Clamp(e.dopamine + .05f, -5f, 5f);
            else
                e.serotonin = Mathf.Clamp(e.serotonin + .05f, -5f, 5f);
        }

        previousLoss = loss;
    }
    public float[] ChooseAction(float[] state)
    {
        if (UnityEngine.Random.value < explorationRate)
        {
            // исследование
            float[] randomAction = new float[outputCount];
            for (int i = 0; i < outputCount; i++)
            {
                randomAction[i] = UnityEngine.Random.Range(-1f, 1f);
            }
            return randomAction;
        }
        else
        {
            return Think(state);
        }
    }
    public void TrainRL(int batchSize, float learningRate)
    {
        if (memory._buffer.Count < batchSize) return;

        var batch = memory.Sample(batchSize);

        foreach (var experience in batch)
        {
            float[] currentQValues = Think(experience.State);

            float[] targetQValues = (float[])currentQValues.Clone();
            int bestActionIdx = Array.IndexOf(currentQValues, currentQValues.Max());
            targetQValues[bestActionIdx] = experience.Reward + gamma * currentQValues.Max();

            Train(experience.State, targetQValues, learningRate);
        }

        explorationRate = Mathf.Max(explorationMin, explorationRate * explorationDecay);
    }
    public void Remember(float[] state, float[] action, float reward)
    {
        Memory.Experience exp;
        exp.State = state;
        exp.Action = action;
        exp.Reward = reward;
        memory.Add(exp);
    }
    public void Tick(float deltaTime)
    {
        float decayRate = 1f; // скорость затухани€
        foreach (ENeuron e in emotionsPerLayer)
        {
            if (UnityEngine.Random.value <= .05f)
            {
                //ћотивационный скачок
                e.dopamine = 2f;
                e.serotonin = 0f;
                e.adrenaline = 2f;
            }
            else
            {
                e.dopamine = Mathf.Clamp(e.dopamine - decayRate * deltaTime, -5f, 5f);
                e.serotonin = Mathf.Clamp(e.serotonin - decayRate * 1.1f * deltaTime, -5f, 5f);
                e.adrenaline = Mathf.Clamp(e.adrenaline - decayRate * 1.5f * deltaTime, 0, 5f);
            }
        }
    }
}
