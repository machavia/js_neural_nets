const {
    ENV,
    AdagradOptimizer,
    AdamOptimizer,
    AdamaxOptimizer,
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

class FFN1Dold {
    constructor({
        graph, input_size=2, hidden_size=3, nonlin='tanh',
        nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.nonlin = nonlin;

        this.x = graph.placeholder('x', [input_size]);
        this.Wh = graph.variable(
            'Wh', Array2D.randNormal([hidden_size, input_size + 1])
        );
        this.ones = graph.constant(NDArray.ones([1]));

        // define h
        let linAlgH = graph.matmul(
            this.Wh,
            graph.concat1d(this.x, this.ones)
        );

        this.h = graph.tanh(linAlgH);

        console.log("nPredictions:", nPredictions)

        this.Wp = graph.variable(
            'Wp', Array2D.randNormal([nPredictions, input_size + 1])
        );

        // define p
        this.logits = graph.matmul(
            this.Wp,
            graph.concat1d(this.x, this.ones)
        );

        console.log("logits shape:", this.logits.shape)

        this.p = graph.softmax(this.logits);

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

    }

}

class FFN1D {
    constructor({
        graph, input_size=2, hidden_size=3, nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;

        this.x = graph.placeholder('x', [input_size]);
        this.y = graph.placeholder('x', [nPredictions]);

        const EPSILON = 1e-7;
        // Variables are containers that hold a value that can be updated from
        // trainingraph.
        // Here we initialize the multiplier variable randomly.
        const hiddenLayer = graph.layers.dense(
            'hiddenLayer', this.x, hidden_size, (x) => graph.relu(x), true
        );
        this.output = graph.layers.dense(
            'outputLayer', hiddenLayer, nPredictions, (x) => graph.sigmoid(x),
            true
        );
        this.cost = graph.reduceSum(graph.add(
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    this.y,
                    graph.log(
                        graph.add(this.output, graph.constant([EPSILON]))))),
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    graph.subtract(graph.constant([1]), this.y),
                    graph.log(
                        graph.add(
                            graph.subtract(graph.constant([1]), this.output),
                            graph.constant([EPSILON])))))));

        console.log("nPredictions:", nPredictions)

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
 class LSTMCell{
    constructor({
        graph, input_size=2, hidden_size=3, nPredictions=1
    }){
        this.input_size = input_size;
        this.hidden_size = hidden_size;

        this.x = graph.placeholder('x', [input_size]);
        this.y = graph.placeholder('y', [nPredictions]);
        this.c_tm1 = graph.placeholder('c_tm1', [hidden_size]);
        this.h_tm1 = graph.placeholder('h_tm1', [hidden_size]);

        let XHC_prev = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c_tm1)
        )
        let XH = graph.concat1d(this.x, this.h_tm1)

        this.i = graph.layers.dense(
            'i', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true);
        this.f = graph.layers.dense(
            'f', XHC_prev, hidden_size,
            (x) => graph.sigmoid(x), true);

        this.c = graph.add(
            graph.multiply(this.f, this.c_tm1),
            graph.multiply(
                this.i,
                graph.layers.dense(
                    'c_half', XH, hidden_size,
                    (x) => graph.tanh(x), true)
            )
        );

        let XHC = graph.concat1d(
            this.x, graph.concat1d(this.h_tm1, this.c)
        )

        this.o = graph.layers.dense(
            'o', XHC, hidden_size,
            (x) => graph.sigmoid(x), true);

        // define h
        this.h = graph.multiply(this.o, graph.tanh(this.c));

        this.output = graph.layers.dense(
            'classif', this.h, nPredictions,
            (x) => graph.sigmoid(x), true);


        const EPSILON = 1e-7;
        this.cost = graph.reduceSum(graph.add(
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    this.y,
                    graph.log(
                        graph.add(this.output, graph.constant([EPSILON]))))),
            graph.multiply(
                graph.constant([-1]),
                graph.multiply(
                    graph.subtract(graph.constant([1]), this.y),
                    graph.log(
                        graph.add(
                            graph.subtract(graph.constant([1]), this.output),
                            graph.constant([EPSILON])))))));

        console.log("nPredictions:", nPredictions)

    }
}


class LSTMCellOld {
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
    hiddenSize=8, noise=0
}){
    
    let ds = getPalindromeDataset({
        datasetSize: datasetSize,
        dim: dim,
        flatten: true,
        seq_len: seq_len,
        outputLength: outputSize,
        noise: noise
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

    return([ds, xProvider, hProvider, cProvider, lProvider]);
}

async function lstmDemo(){

    // await firstLearn();

    let batchSize = 8;
    let hiddenSize = 16;
    let outputSize = 2;
    let learningRate = 0.5;

    let [ds, xProvider, hProvider, cProvider, lProvider] = getDataSet({
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

}

function customSigmoidCost(graph, y, l){
    const EPSILON = 1e-7;
    let output = y;
    y = l;
    return(
        graph.reduceSum(
            graph.multiply(
                graph.constant([-1]),
                graph.add(
                    graph.multiply(
                        y,
                        graph.log(
                            graph.add(
                                output,
                                graph.constant([EPSILON])
                            )
                        )
                    ),
                    graph.multiply(
                        graph.subtract(graph.constant([1]), y),
                        graph.log(
                            graph.add(
                                graph.subtract(
                                    graph.constant([1]),
                                    output
                                ),
                                graph.constant([EPSILON])
                            )
                        )
                    )
                )
            )
        )
    );
}

function testExample({
    withTanh=false, withCustomSigmoid=false, withSuperCustomSigmoid=false
}){

    const grph = new Graph();
    const math = ENV.math;
    const session = new Session(grph, math);

    let learningRate = 0.1; // .00001
    let momentum = 0.9;
    let batchSize = 64; // 3
    let graphCostFunc = (prey, y, l) => {return(grph.meanSquaredCost(y, l))};

    console.log(
        'withTanh', withTanh, // 'withSigmoid', withSigmoid,
        'withCustomSigmoid', withCustomSigmoid,
        'withSuperCustomSigmoid', withSuperCustomSigmoid,
        "withCustomSigmoid || withSuperCustomSigmoid", withCustomSigmoid || withSuperCustomSigmoid
    )

    console.log("???????????????????????????????????????????????????????????")
    if (withTanh){
        console.log("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        transformFunc = Math.tanh;
        graphTransformFunc = (x) => {return(grph.tanh(x))};
    } else if (withCustomSigmoid || withSuperCustomSigmoid){
        console.log(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        transformFunc = (x) => {return((x > 1.5) + 0)};
        // graphTransformFunc = (x) => {return(g.sigmoid(x))};
        graphTransformFunc = (x) => {
            return(
                grph.divide(
                    grph.constant([1]), 
                    grph.add(
                        grph.constant([1]), 
                        grph.exp(grph.multiply(grph.constant([-1]), x))
                    )
                )
            )    
        };
        learningRate = 0.1; // .00001
        momentum = 0.8;
        batchSize = 64; // 3
        if(withCustomSigmoid){
            console.log("----------------------------------------------------")
            // from demo xor
            const EPSILON = 1e-7;

            const graphCostFunc = (prey, y, l) => {
                // copied from xor demo
                // output = y;
                output = y;
                y = l;
                grph.reduceSum(
                    grph.add(
                    grph.multiply(
                        grph.constant([-1]),
                        grph.multiply(
                            y, grph.log(
                                grph.add(output, grph.constant([EPSILON]))
                            )
                        )
                    ),
                    grph.multiply(
                        grph.constant([-1]),
                        grph.multiply(
                            grph.subtract(grph.constant([1]), y),
                            grph.log(grph.add(
                                grph.subtract(grph.constant([1]), output),
                                grph.constant([EPSILON])))))));
            }
        }else if(withSuperCustomSigmoid){
            console.log("====================================================")
            graphCostFunc = (prey, y, l) => {
                return(customSigmoidCost(grph, y, l))
                // output = y;
            }
        }else{
            console.log("::::::::::::::::::::::::::::::::::::::::::::::::::::")
            graphCostFunc = (prey, y, l) => {
                return(grph.softmaxCrossEntropyCost(prey, l))
            };
        }
    }else{
        console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        transformFunc = (x) => {return(x)};
        graphTransformFunc = (x) => {return(idt(grph, x))};
        learningRate = 0.1; // .00001
        momentum = 0.8;
        batchSize = 64; // 3
    }

    // Placeholders are input containers. This is the container for where we will
    // feed an input NDArray when we execute the graph.
    const inputShape = [3];
    const inputTensor = grph.placeholder('input', inputShape);

    const labelShape = [1];
    const labelTensor = grph.placeholder('label', labelShape);

    // Variables are containers that hold a value that can be updated from
    // training.
    // Here we initialize the multiplier variable randomly.
    const multiplier = grph.variable('multiplier', Array2D.randNormal([1, 3]));

    // Top level graph methods take Tensors and return Tensors.
    const preOut = grph.matmul(multiplier, inputTensor)
    const outputTensor = graphTransformFunc(preOut)
    ;
    const costTensor = graphCostFunc(preOut, outputTensor, labelTensor);

    // Tensors, like NDArrays, have a shape attribute.
    console.log(outputTensor.shape);

    // const optimizer = new SGDOptimizer(learningRate);
    // const optimizer = new MomentumOptimizer(learningRate, momentum);
    const optimizer = new MomentumOptimizer(learningRate, momentum);

    const inputs = [];
    const labels = [];
    for(let i = 0; i <= 1000; i++){
        // const mult = Math.random() * 100;
        const a = Math.random();
        const b = Math.random();
        const c = Math.random();
        const out = transformFunc(a + b + c);
        // console.log(mult, a, b, c , out)
        inputs.push(Array1D.new([a, b, c]));
        labels.push(Array1D.new([out]));
    
        /*
        if ( !(graphTransformFunc === null)){
            console.log(
                out,
                session.eval(g.tanh(g.constant(a + b + c))).dataSync();
            )
        }
        */
    }


    /*
    const inputs = [
      Array1D.new([1.0, 2.0, 3.0]),
      Array1D.new([10.0, 20.0, 30.0]),
      Array1D.new([100.0, 200.0, 300.0]),
      Array1D.new([1.0, 2.0, 3.0]),
      Array1D.new([10.0, 20.0, 30.0]),
      Array1D.new([100.0, 200.0, 300.0])
    ];

    const labels = [
      Array1D.new([transformFunc(4.0)]),
      Array1D.new([transformFunc(40.0)]),
      Array1D.new([transformFunc(400.0)]),
      Array1D.new([transformFunc(4.0)]),
      Array1D.new([transformFunc(40.0)]),
      Array1D.new([transformFunc(400.0)])
    ];
    */

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

    const NUM_BATCHES = 100;
    for (let i = 0; i < NUM_BATCHES; i++) {
      // Train takes a cost tensor to minimize. Trains one batch. Returns the
      // average cost as a Scalar.
      const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN);

      console.log('last average cost (' + i + '): ' + cost.dataSync());
    }

}

// math.basicLSTMCell
async function demoFFN1D(printInfo=false){

    math = ENV.math;
    const graph = new Graph();
    const session = new Session(graph, math);

    // await firstLearn();

    let batchSize = 64;
    let hiddenSize = 8;
    let outputSize = 1;
    let learningRate = 0.2;
    let momentum = 0.9;
    let seqLen = 2;
    let noise = 0.4;

    let [ds, xProvider, hProvider, cProvider, lProvider] = getDataSet({
        datasetSize: 10 * batchSize,
        dim: 1,
        seq_len: seqLen / 2,
        outputSize: outputSize,
        noise: noise
    })

    const ffn = new FFN1D({
        graph: graph, nPredictions: outputSize,
        hidden_size: hiddenSize, input_size: seqLen
    });

    // Maps tensors to InputProviders.
    xFeed = {tensor: ffn.x, data: xProvider};
    lFeed = {tensor: ffn.y, data: lProvider};

    feedEntries = [xFeed, lFeed];

    x_check = graph.reshape(ffn.x, ffn.x.shape);
    l_check = graph.reshape(ffn.y, ffn.y.shape);

    optimizer = new MomentumOptimizer(learningRate, momentum);
    // optimizer = new SGDOptimizer(learningRate);

    NUM_BATCHES = 100;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        if(printInfo){
            /*
            const [p1, x1, t1] = session.evalAll(
                [ffn.output, x_check, l_check], feedEntries
            );
            console.log("p:", p1.dataSync(), "t:", t1.dataSync());
            */
            for(let i = 0; i <= 100; i++){
                let input = Array1D.new(ds[i].input);
                let target = ds[i].output; 
                let pred = session.eval(
                    ffn.output, [{tensor: ffn.x, data: input}]
                );
                console.log("p:", pred.dataSync(), "t:", target);
            }
        }
        

        const cost = session.train(
            ffn.cost, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );
        costVal = await cost.val();
        console.log('last average cost (' + i + '): ' + costVal);
    }

}

async function demoLSTM(printInfo=false){

    math = ENV.math;
    const graph = new Graph();
    const session = new Session(graph, math);

    // await firstLearn();

    let batchSize = 64;
    let hiddenSize = 8;
    let outputSize = 1;
    let learningRate = 0.2;
    let momentum = 0.9;
    let seqLen = 2;
    let noise = 0.4;

    let [ds, xProvider, hProvider, cProvider, lProvider] = getDataSet({
        datasetSize: 10 * batchSize,
        dim: 1,
        seq_len: seqLen / 2,
        outputSize: outputSize,
        noise: noise
    })

    const lstm = new LSTMCell({
        graph: graph, nPredictions: outputSize,
        hidden_size: hiddenSize, input_size: seqLen
    });

    // Maps tensors to InputProviders.
    xFeed = {tensor: lstm.x, data: xProvider};
    lFeed = {tensor: lstm.y, data: lProvider};
    hFeed = {tensor: lstm.h_tm1, data: hProvider};
    cFeed = {tensor: lstm.c_tm1, data: cProvider};

    feedEntries = [xFeed, lFeed, hFeed, cFeed];

    x_check = graph.reshape(lstm.x, lstm.x.shape);
    l_check = graph.reshape(lstm.y, lstm.y.shape);

    optimizer = new MomentumOptimizer(learningRate, momentum);
    // optimizer = new SGDOptimizer(learningRate);

    NUM_BATCHES = 1000;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.
        if(printInfo){
            /*
            const [p1, x1, t1] = session.evalAll(
                [ffn.output, x_check, l_check], feedEntries
            );
            console.log("p:", p1.dataSync(), "t:", t1.dataSync());
            */
            for(let i = 0; i <= 100; i++){
                let input = Array1D.new(ds[i].input);
                let target = ds[i].output; 
                let pred = session.eval(
                    lstm.output, [
                        {tensor: lstm.x, data: input},
                        {tensor: lstm.c_tm1, data: Array1D.zeros([hiddenSize])},
                        {tensor: lstm.h_tm1, data: Array1D.zeros([hiddenSize])}
                    ]
                );
                console.log("p:", pred.dataSync(), "t:", target);
            }
        }
        

        const cost = session.train(
            lstm.cost, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );
        costVal = await cost.val();
        console.log('last average cost (' + i + '): ' + costVal);
    }

}


function idt(graph, x){
    return(graph.reshape(x, x.shape));
}



function slightlyModifiedTestExample(printInfo=false){
    const g = new Graph();
    const math = ENV.math;
    const session = new Session(g, math);

    seqLen = 2;
    const labelShape = 1;
    let batchSize = 64;
    let hiddenSize = 2;
    let learningRate = 0.1;
    let momentum = 0.8;
    let datasetSize = 640;
    let noise = 0.49;


    // Placeholders are input containers. This is the container for where we will
    // feed an input NDArray when we execute the graph.
    const inputShape = [seqLen];

    const inputTensor = g.placeholder('input', inputShape);
    const labelTensor = g.placeholder('label', [labelShape]);

    const EPSILON = 1e-7;
    // Variables are containers that hold a value that can be updated from
    // training.
    // Here we initialize the multiplier variable randomly.
    const hiddenLayer = g.layers.dense(
        'hiddenLayer', inputTensor, hiddenSize,
        (x) => g.relu(x),
        true
    );
    const outputTensor = g.layers.dense(
        'outputLayer', hiddenLayer, labelShape,
        (x) => g.sigmoid(x),
        true
    );
    const costTensor = g.reduceSum(g.add(
        g.multiply(
            g.constant([-1]),
            g.multiply(
                outputTensor, g.log(g.add(outputTensor, g.constant([EPSILON]))))),
        g.multiply(
            g.constant([-1]),
            g.multiply(
                g.subtract(g.constant([1]), outputTensor),
                g.log(g.add(
                    g.subtract(g.constant([1]), outputTensor),
                    g.constant([EPSILON])))))));
    const optimizer = new MomentumOptimizer(learningRate, momentum);
  

    /*
    const multiplier = g.variable(
        'multiplier',
        Array2D.randNormal([labelShape, seqLen + 1])
    );

    // Top level graph methods take Tensors and return Tensors.
    const eInput = g.concat1d(inputTensor, g.constant(Array1D.ones([1])));
    const lin = g.matmul(multiplier, eInput);
    const outputTensor = g.sigmoid(lin);
    const costTensor = customSigmoidCost(g, outputTensor, labelTensor);
    // const costTensor = g.softmaxCrossEntropyCost(lin, labelTensor);

    const optimizer = new MomentumOptimizer(learningRate, momentum);
    // const optimizer = new AdamOptimizer(learningRate, momentum);
    // const optimizer = new AdagradOptimizer(learningRate);
    */

    let [ds, inputProvider, , , labelProvider] = getDataSet({
        datasetSize: datasetSize,
        dim: 1,
        seq_len: seqLen / 2,
        outputSize: labelShape,
        noise: noise 
    });

    // Maps tensors to InputProviders.
    const feedEntries = [
        {tensor: inputTensor, data: inputProvider},
        {tensor: labelTensor, data: labelProvider}
    ];

    const NUM_BATCHES = 1000;
    for (let i = 0; i < NUM_BATCHES; i++) {
        // Train takes a cost tensor to minimize. Trains one batch. Returns the
        // average cost as a Scalar.

        if(printInfo){
            const [x, l, p] =
                session.evalAll(
                    [idt(g, inputTensor), idt(g, labelTensor), outputTensor], 
                    feedEntries
                );
            console.log('x', x.dataSync(), 'l', l.dataSync(), 'p', p.dataSync());
        }
        const cost = session.train(
          costTensor, feedEntries, batchSize, optimizer, CostReduction.MEAN
        );

        console.log('last average cost (' + i + '): ' + cost.dataSync());
    }
}

/*
testExample({});
testExample({withTanh: true});
testExample({withSigmoid: true});
testExample({withCustomSigmoid: true});
testExample({withSuperCustomSigmoid: true});
slightlyModifiedTestExample(printInfo=false);
demoFFN1D(printInfo=true);
*/
demoLSTM(printInfo=false);
