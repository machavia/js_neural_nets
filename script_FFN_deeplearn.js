const {
    ENV,
    AdamOptimizer,
    Array1D,
    Array2D,
    Array3D,
    CostReduction,
    FeedEntry,
    Graph,
    MomentumOptimizer,
    RMSPropOptimizer,
    Scalar,
    Session,
    SGDOptimizer,
    Tensor,
    NDArray,
    NDArrayMath,
    MathBackendCPU,
    InCPUMemoryShuffledInputProviderBuilder
} = require('./node_modules/deeplearn/dist/deeplearn')

const {
    getPalindromeDataset,
} = require("./utils");



deeplearn = require('./node_modules/deeplearn/dist/deeplearn')

// ENV.math.basicLSTMCell

// Make a new input in the graph, called 'x', with shape [] (a Scalar).
// create an LSTM with one layer and as much input as what's provided

function graphSum({graph=null, toSum=null, }){
    if(toSum === null){
        throw('toSum argument was not given');
    }
    if(graph === null){
        throw('graphargument was not given');
    }

    let runningSum = [];
    let prev = []
    for(let pair=0; pair < toSum.length - 1; pair++){
        prev = pair === 0 ? toSum[pair]: runningSum;
        runningSum = graph.add(prev, toSum[pair + 1]);
    }
    return(runningSum);
}

/*
class RNN() {
    constructor(
        module,
        inputs=["x", "h", "c"],
        outputs,
        repeats=1
    ){


    }

    forward(){


    }
}
*/

class FFN1D {
    constructor({
        graph, input_size=2, hidden_size=3, nonlin='tanh',
        nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.nonlin = nonlin;

        this.x = graph.placeholder('x', [input_size]);
        this.Wh = graph.variable(
            'Wh', Array2D.randNormal([input_size + 1, hidden_size])
        );
        this.ones = graph.constant(NDArray.ones([1]));

        // define h
        let linAlgH = graph.matmul(
            graph.concat2d(this.x, this.ones, 0),
            this.Wh
        );

        this.h = graph.tanh(linAlgH);

        console.log("nPredictions:", nPredictions)

        this.Wp = graph.variable(
            'Wp', Array2D.randNormal([input_size + 1, nPredictions])
        );

        // define p
        this.logits = graph.matmul(
            graph.concat2d(this.x, this.ones, 0),
            this.Wp
        );

        console.log("logits shape:", this.logits.shape)

        this.p = graph.softmax(this.logits);

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

    }

}


class FFN {
    constructor({
        graph, input_size=2, hidden_size=3, batch_size=4, nonlin='tanh',
        nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.batch_size = batch_size;
        this.nonlin = nonlin;

        this.x = graph.placeholder('x', [batch_size, input_size]);
        this.Wh = graph.variable(
            'Wh', Array2D.randNormal([input_size + 1, hidden_size])
        );
        this.ones = graph.constant(NDArray.ones([batch_size, 1]));

        // define h
        let linAlgH = graph.matmul(
            graph.concat2d(this.x, this.ones, 1),
            this.Wh
        );

        this.h = graph.tanh(linAlgH);

        console.log("nPredictions:", nPredictions)

        this.Wp = graph.variable(
            'Wp', Array2D.randNormal([input_size + 1, nPredictions])
        );

        // define p
        this.logits = graph.matmul(
            graph.concat2d(this.x, this.ones, 1),
            this.Wp
        );

        console.log("logits shape:", this.logits.shape)

        this.p = graph.softmax(this.logits);

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

    }

}


class LSTMCell {
    constructor({
        graph, input_size=2, hidden_size=3, batch_size=4, nonlin='tanh',
        nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.batch_size = batch_size;
        this.nonlin = nonlin;

        this.x = graph.placeholder('x', [batch_size, input_size]);
        this.c_tm1 = graph.placeholder('c_tm1', [batch_size, hidden_size]);
        this.h_tm1 = graph.placeholder('h_tm1', [batch_size, hidden_size]);
        this.biasReplicator = graph.constant(NDArray.ones([batch_size, 1]));
        // this.cellEyeMaker = graph.constant(NDArray.ones([batch_size, 1]));
        // math.oneHot(Array1D.new([0, 1, 2]), 3, 1, 0)

        // define both i and f
        let inputs = new Map([
            ['x', ['x', input_size]],
            ['h_tm1', ['h', hidden_size]],
            ['c_tm1', ['c', hidden_size]]
        ]);
        this.defineGate(graph, 'i', inputs, "sigmoid");
        this.defineGate(graph, 'f', inputs, "sigmoid");

        // NOTE: Graves 2014 => matrices for cells are diagonal so that
        // only one input particpates, no trace in the tensoflow code

        // define c
        // now we should multiply ft and ct after setting them to diagonals...
        // or do a plain elementwise mult using extra funcs from math
        let c_part1 = graph.multiply(this.f, this.c_tm1);
        inputs = new Map([
            ['x', ['x', input_size]],
            ['h_tm1', ['h', hidden_size]]
        ]);
        this.defineGate(graph, 'c_half', inputs, "tanh");
        let c_part2 = graph.multiply(this.i, this.c_half);
        this.c = graph.add(c_part1, c_part2);

        // define o
        inputs = new Map([
            ['x', ['x', input_size]],
            ['h_tm1', ['h', hidden_size]],
            ['c', ['c', hidden_size]]
        ]);
        this.defineGate(graph, 'o', inputs, "sigmoid");

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        // add an extra layer for classif
        inputs = new Map([
            ['h', ['h', hidden_size]],
        ]);
        this.defineGate(graph, 'p', inputs, "sigmoid", nPredictions);

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

        this.x_check = graph.reshape(this.x, [batch_size, input_size]);
        this.m_check = graph.matmul(this.x, this.Wxi);
        this.h_check = graph.matmul(this.h_tm1, this.Whi);
        this.biasMultCheck = graph.matmul(this.biasReplicator, this.bi);

    }

    defineGate(graph, gateName, inputs, nonlin, outSize=null){
        // inputs: dict => keys = input name, values last dim

        let toSum = [];
        if(outSize == null){
            outSize = this.hidden_size;
        }
        for(let [inputName, [inputIndex, lastDim]] of inputs){
            if(! inputs.has(inputName)){
                throw('no input ' + inputName + ' in inputs');
            }
            if(! (inputName in this)){
                throw(
                    'no input ' + inputName + ' in LSTM: ' + Object.keys(this)
                );
            }
            let indexName = inputIndex + gateName;
            let matName = 'W' + indexName;
            this[matName] = graph.variable(
                matName, Array2D.randNormal([lastDim, outSize])
            );
            let linAlg = graph.matmul(this[inputName], this[matName])
            toSum.push(linAlg)
        }
        
        this['b' + gateName] = graph.variable(
            'b' + gateName,
            Array2D.randNormal([1, outSize])
        );

        toSum.push(graph.matmul(this.biasReplicator, this['b' + gateName]));

        this['add' + gateName] = graphSum({'toSum': toSum, 'graph': graph});

        switch(nonlin){
            case 'tanh':
                this[gateName] = graph.tanh(this['add' + gateName]);
                break;
            default:
                this[gateName] = graph.sigmoid(this['add' + gateName]);
        }


    }

    forward({session, x, c, h, printCheck=false}){
        // Shuffles inputs and labels and keeps them mutually in sync.
        const shuffledInputProviderBuilder =
            new InCPUMemoryShuffledInputProviderBuilder([x, h, c]);

        const [xProvider, hProvider, cProvider] =
            shuffledInputProviderBuilder.getInputProviders();

        // Maps tensors to InputProviders.
        const xFeed = {tensor: this.x, data: xProvider};
        const hFeed = {tensor: this.h_tm1, data: hProvider};
        const cFeed = {tensor: this.c_tm1, data: cProvider};

        const feedEntries = [xFeed, hFeed, cFeed];

        if(printCheck){
            let val_x =
                session.eval(this.x_check, [xFeed]).dataSync();
            let val_m_check =
                session.eval(this.m_check, [xFeed]).dataSync();
            let val_h_check =
                session.eval(this.h_check, [hFeed]).dataSync();
            let val_add_check =
                session.eval(this.addi, [xFeed, hFeed, cFeed]).dataSync();
            let biasMultCheck =
                session.eval(this.biasMultCheck, []).dataSync();
            console.log("add", val_add_check);
            console.log('bMM', biasMultCheck);
        }

        let val_c = session.eval(this.i, feedEntries);
        let val_h = session.eval(this.f, feedEntries);
        return(val_c, val_h);
    }
}

async function run(
    session, feedEntries, costTensor, optimizer, batchSize, numBatch=10
){
    for (let i = 0; i < numBatch; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );
        costVal = await cost.val();
        console.log('last average cost (' + i + '): ' + costVal);
    }
}


function playing(){
    // var safeMode = true;
    var safeMode = false;

    // math = NDArrayMath(new MathBackendWebGL(), safeMode)
    math = new NDArrayMath(new MathBackendCPU(), safeMode)

    const lstmKernel = Array2D.randNormal([3, 4]);
    const lstmBias = Array1D.randNormal([4]);
    const forgetBias = Scalar.new(1.0);

    const data = Array2D.randNormal([1, 2]);
    const batchedData = math.concat2D(data, data, 0);    // 2x2
    const batchedC = Array2D.randNormal([2, 1]);
    const batchedH =  Array2D.randNormal([2, 1]);
    const [newC, newH] = math.basicLSTMCell(
        forgetBias, lstmKernel, lstmBias, batchedData, batchedC, batchedH
    );
    

    // graph, input_size=2, hidden_size=3, batch_size=4, nonlin='tanh'
    
    [x, h, c] = [
        [new Array1D.randTruncatedNormal([4, 2])],
        [new Array1D.zeros([4, 3])],
        [new Array1D.zeros([4, 3])]
    ]
}

async function firstLearn(){
    math = ENV.math;
    const graph = new Graph();

    // training
    const firstDim = 4;
    const batchSize = 4;
    [x, h, c, labels] = [
        [
            new Array1D.randTruncatedNormal([firstDim, 2]),
            new Array1D.randTruncatedNormal([firstDim, 2]),
            new Array1D.randTruncatedNormal([firstDim, 2]),
        ],
        [
            new Array1D.zeros([firstDim, 3]),
            new Array1D.zeros([firstDim, 3]),
            new Array1D.zeros([firstDim, 3]),
        ],
        [
            new Array1D.zeros([firstDim, 3]),
            new Array1D.zeros([firstDim, 3]),
            new Array1D.zeros([firstDim, 3]),
        ],
        [
            new Array1D.ones([firstDim, 1]),
            new Array1D.ones([firstDim, 1]),
            new Array1D.ones([firstDim, 1]),
        ]
    ]

    const lstmCell = new LSTMCell({graph: graph, batch_size: firstDim});
    const session = new Session(graph, math);

    lstmCell.forward({session: session, x: x, h: h, c: c, printCheck: false});

    const learningRate = 0.5;
    let labelTensor = graph.placeholder('label', [batchSize, 1]);
    let costTensor = graph.meanSquaredCost(lstmCell.p, labelTensor);
    const optimizer = new SGDOptimizer(learningRate);

    // Shuffles inputs and labels and keeps them mutually in sync.
    const shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([x, h, c, labels]);

    const [xProvider, hProvider, cProvider, lProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    // Maps tensors to InputProviders.
    const xFeed = {tensor: lstmCell.x, data: xProvider};
    const hFeed = {tensor: lstmCell.h_tm1, data: hProvider};
    const cFeed = {tensor: lstmCell.c_tm1, data: cProvider};
    const lFeed = {tensor: labelTensor, data: lProvider};

    const feedEntries = [xFeed, hFeed, cFeed, lFeed];

    await run(session, feedEntries, costTensor, optimizer, batchSize);
}

function getDataSet({
    datasetSize, dim=1, flatten=true, seq_len=1, outputSize=2,
    hiddenSize=8
}){
    
    let ds = getPalindromeDataset({
        datasetSize: 8,
        dim: 1,
        flatten: true,
        seq_len: 1,
        outputLength: outputSize
    });

    let [x, h, c, labels] = [[], [], [], []];

    math = ENV.math;
    const graph = new Graph();
    const session = new Session(graph, math);

    let labelTensor = [];
    let costTensor = [];
    let optimizer = [];
    let feedEntries = [];

    let shuffledInputProviderBuilder = [];
    let [xProvider, hProvider, cProvider, lProvider] = []
    let [xFeed, hFeed, cFeed, lFeed] = []

    for(dat of ds){
        x.push(Array1D.new(dat.input));
        labels.push(Array1D.new(dat.output));
        h.push(Array1D.zeros([hiddenSize]));
        c.push(Array1D.zeros([hiddenSize]));
    }

    // Shuffles inputs and labels and keeps them mutually in sync.
    shuffledInputProviderBuilder =
        new InCPUMemoryShuffledInputProviderBuilder([x, h, c, labels]);

    [xProvider, hProvider, cProvider, lProvider] =
        shuffledInputProviderBuilder.getInputProviders();

    return([xProvider, hProvider, cProvider, lProvider]);
}

// math.basicLSTMCell
async function demo(){

    // await firstLearn();

    let batchSize = 8;
    let hiddenSize = 16;
    let outputSize = 2;
    let learningRate = 0.5;

    let [xProvider, hProvider, cProvider, lProvider] = getDataSet({
        datasetSize: batchSize,
        dim: 1,
        seq_len: 2,
        outputSize: 2
    })

    /*
    let count = 0;
    for(dat of ds){
        if(count % batchSize === 0){
            x.push([]);
            labels.push([]);
        }
        x[x.length - 1] = x[x.length - 1].concat(dat.input);
        labels[x.length - 1] = labels[x.length - 1].concat(dat.output);
        if(count % batchSize === (batchSize - 1)){
            x[x.length - 1] = Array2D.new(
                [batchSize, dat.input.length], x[x.length - 1]
            );
            labels[labels.length - 1] = Array2D.new(
                [batchSize, dat.output.length], labels[labels.length - 1]
            );
            
            h.push(Array1D.zeros([batchSize, hiddenSize]));
            c.push(Array1D.zeros([batchSize, hiddenSize]));
            
        }
        count += 1;
    }
    */

    labelTensor = graph.placeholder('label', [outputSize]);
    
    /*
    const lstmCell = new LSTMCell({
        graph: graph, batch_size: batchSize, nPredictions: outputSize,
        hidden_size: hiddenSize, input_size: ds[0].input.length
    });

    lstmCell.forward({session: session, x: x, h: h, c: c, printCheck: false});

    learningRate = 0.5;
    costTensor = graph.softmaxCrossEntropyCost(lstmCell.p, labelTensor);
    optimizer = new SGDOptimizer(learningRate);

    // Maps tensors to InputProviders.
    xFeed = {tensor: lstmCell.x, data: xProvider};
    hFeed = {tensor: lstmCell.h_tm1, data: hProvider};
    cFeed = {tensor: lstmCell.c_tm1, data: cProvider};
    lFeed = {tensor: labelTensor, data: lProvider};

    feedEntries = [xFeed, hFeed, cFeed, lFeed];

    tCheck = graph.reshape(labelTensor, labelTensor.shape);

    // run(session, feedEntries, costTensor, optimizer, batchSize, 100);
    /*
    let NUM_BATCHES = 10;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );
        costVal = await cost.val();
        console.log('last average cost (' + i + '): ' + costVal);

        const [p1, l1] = session.evalAll([lstmCell.p, tCheck], feedEntries);
        console.log(
            "p ==>", p1.dataSync().slice(0, 2),
            "t ==>",  l1.dataSync().slice(0, 2)
        );
    }
    */

    hiddenSize = 16;
    learningRate = 0.1;
    let momentum = 0.7;

    const ffn = new FFN1D({
        graph: graph, nPredictions: outputSize,
        hidden_size: hiddenSize, input_size: ds[0].input.length
    });

    // Maps tensors to InputProviders.
    xFeed = {tensor: ffn.x, data: xProvider};
    lFeed = {tensor: labelTensor, data: lProvider};

    feedEntries = [xFeed, lFeed];

    costTensor = graph.softmaxCrossEntropyCost(ffn.logits, labelTensor);
    // costTensor = graph.meanSquaredCost(ffn.p, labelTensor);

    feedEntries = [xFeed, lFeed];

    x_check = graph.reshape(ffn.x, ffn.x.shape);
    l_check = graph.reshape(labelTensor, labelTensor.shape);

    optimizer = new RMSPropOptimizer(learningRate, momentum);

    NUM_BATCHES = 100;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        const [l1, p1] =
            session.evalAll([ffn.logits, ffn.p], feedEntries);
        console.log("l ==>", l1.dataSync());
        console.log("p ==>", p1.dataSync());

        const cost = session.train(
            costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );

        /*
        costVal = await cost.val();
        console.log('last average cost (' + i + '): ' + costVal);
        */
    }

}

function testExample(){

    const g = new Graph();

    // Placeholders are input containers. This is the container for where we will
    // feed an input NDArray when we execute the graph.
    const inputShape = [3];
    const inputTensor = g.placeholder('input', inputShape);

    const labelShape = [1];
    const labelTensor = g.placeholder('label', labelShape);

    // Variables are containers that hold a value that can be updated from
    // training.
    // Here we initialize the multiplier variable randomly.
    const multiplier = g.variable('multiplier', Array2D.randNormal([1, 3]));

    // Top level graph methods take Tensors and return Tensors.
    const outputTensor = g.matmul(multiplier, inputTensor);
    const costTensor = g.meanSquaredCost(outputTensor, labelTensor);

    // Tensors, like NDArrays, have a shape attribute.
    console.log(outputTensor.shape);

    const learningRate = .00001;
    const batchSize = 3;
    const math = ENV.math;

    const session = new Session(g, math);
    const optimizer = new SGDOptimizer(learningRate);

    const inputs = [
      Array1D.new([1.0, 2.0, 3.0]),
      Array1D.new([10.0, 20.0, 30.0]),
      Array1D.new([100.0, 200.0, 300.0]),
      Array1D.new([1.0, 2.0, 3.0]),
      Array1D.new([10.0, 20.0, 30.0]),
      Array1D.new([100.0, 200.0, 300.0])
    ];

    const labels = [
      Array1D.new([4.0]),
      Array1D.new([40.0]),
      Array1D.new([400.0]),
      Array1D.new([4.0]),
      Array1D.new([40.0]),
      Array1D.new([400.0])
    ];

    // Shuffles inputs and labels and keeps them mutually in sync.
    const shuffledInputProviderBuilder =
      new InCPUMemoryShuffledInputProviderBuilder([inputs, labels]);
    const [inputProvider, labelProvider] =
      shuffledInputProviderBuilder.getInputProviders();

    // Maps tensors to InputProviders.
    const feedEntries = [
      {tensor: inputTensor, data: inputProvider},
      {tensor: labelTensor, data: labelProvider}
    ];

    const NUM_BATCHES = 10;
    for (let i = 0; i < NUM_BATCHES; i++) {
      // Train takes a cost tensor to minimize. Trains one batch. Returns the
      // average cost as a Scalar.
      const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

      console.log('last average cost (' + i + '): ' + cost.dataSync());
    }

}


function slightlyModifiedTestExample(){

    const g = new Graph();

    seqLen = 4;
    const labelShape = 2;
    let batchSize = 8;
    let hiddenSize = 16;
    let learningRate = 0.005;

    // Placeholders are input containers. This is the container for where we will
    // feed an input NDArray when we execute the graph.
    const inputShape = [seqLen];
    const inputTensor = g.placeholder('input', inputShape);

    const labelTensor = g.placeholder('label', [labelShape]);

    // Variables are containers that hold a value that can be updated from
    // training.
    // Here we initialize the multiplier variable randomly.
    const multiplier = g.variable(
        'multiplier',
        Array2D.randNormal([labelShape, seqLen])
    );

    // Top level graph methods take Tensors and return Tensors.
    const outputTensor = g.tanh(g.matmul(multiplier, inputTensor));
    const costTensor = g.meanSquaredCost(outputTensor, labelTensor);

    // Tensors, like NDArrays, have a shape attribute.
    console.log(outputTensor.shape);

    const math = ENV.math;

    const session = new Session(g, math);
    const optimizer = new SGDOptimizer(learningRate);

    let [inputProvider, , , labelProvider] = getDataSet({
        datasetSize: batchSize,
        dim: 1,
        seq_len: seqLen / 2,
        outputSize: labelShape
    })

    // Maps tensors to InputProviders.
    const feedEntries = [
      {tensor: inputTensor, data: inputProvider},
      {tensor: labelTensor, data: labelProvider}
    ];

    const NUM_BATCHES = 10;
    for (let i = 0; i < NUM_BATCHES; i++) {
      // Train takes a cost tensor to minimize. Trains one batch. Returns the
      // average cost as a Scalar.
      const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

      console.log('last average cost (' + i + '): ' + cost.dataSync());
    }

}



slightlyModifiedTestExample();


// demo();
