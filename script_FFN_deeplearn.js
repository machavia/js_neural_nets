const {
    ENV,
    Array1D,
    Array2D,
    Array3D,
    AdamOptimizer,
    FeedEntry,
    Graph,
    Scalar,
    Session,
    Tensor,
    NDArray,
    NDArrayMath,
    MathBackendCPU,
    InCPUMemoryShuffledInputProviderBuilder
} = require('./node_modules/deeplearn/dist/deeplearn')

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

class LSTM {
    constructor(
        graph, input_size=2, hidden_size=3, batch_size=4, nonlin='tanh'
    ){
        this.input_size = input_size;
        this.hidden_size = hidden_size;
        this.batch_size = batch_size;
        this.nonlin = nonlin;

        this.x = graph.placeholder('x', [batch_size, input_size]);
        this.c_tm1 = graph.placeholder('c_tm1', [batch_size, hidden_size]);
        this.h_tm1 = graph.placeholder('h_tm1', [batch_size, hidden_size]);
        this.biasReplicator = graph.constant(NDArray.ones([batch_size, 1]));
        this.cellEyeMaker = graph.constant(NDArray.ones([batch_size, 1]));

        let inputs = new Map([
            ['x', ['x', input_size]],
            ['h_tm1', ['h', hidden_size]],
            ['c_tm1', ['c', hidden_size]]
        ])
        this.defineGate(graph, 'i', inputs, nonlin);
        this.defineGate(graph, 'f', inputs, nonlin);

        // now we should multiply ft and ct after setting them to diagonals...

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

        this.x_check = graph.reshape(this.x, [batch_size, input_size]);
        this.m_check = graph.matmul(this.x, this.Wxi);
        this.h_check = graph.matmul(this.h_tm1, this.Whi);
        this.biasMultCheck = graph.matmul(this.biasReplicator, this.bi);

        if(nonlin === 'tanh'){}
    }

    defineGate(graph, gateName, inputs, nonlin){
        // inputs: dict => keys = input name, values last dim

        let toSum = [];
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
                matName, Array2D.randNormal([lastDim, this.hidden_size])
            );
            let linAlg = graph.matmul(this[inputName], this[matName])
            toSum.push(linAlg)
        }
        
        this['b' + gateName] = graph.variable(
            'b' + gateName,
            Array2D.randNormal([1, this.hidden_size])
        );

        toSum.push(graph.matmul(this.biasReplicator, this['b' + gateName]));

        this['add' + gateName] = graphSum({'toSum': toSum, 'graph': graph});

        this[gateName] = graph.log(this['add' + gateName]);

    }

    forward(session, x, c, h, printCheck=false){

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

        let val_i = session.eval(this.i, feedEntries);
        let val_f = session.eval(this.f, feedEntries);
        return(val_i, val_f);
    }
}

// math.basicLSTMCell
{

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
    
    console.log(batchedC, newC);

    const graph = new Graph();

    math = ENV.math;

    // graph, input_size=2, hidden_size=3, batch_size=4, nonlin='tanh'
    
    [x, h, c] = [
        [new Array1D.randTruncatedNormal([4, 2])],
        [new Array1D.randTruncatedNormal([4, 3])],
        [new Array1D.randTruncatedNormal([4, 3])]
    ]

    const lstm = new LSTM(graph);
    const session = new Session(graph, math);

    lstm.forward(session, x, h, c, true);
}
