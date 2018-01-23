/*
const {Array1D, Array2D, Scalar, NDArray} =
    require('./node_modules/deeplearn/dist/math/ndarray');

const {MathBackendWebGL} = 
    require("./node_modules/deeplearn/dist/math/backends/backend_webgl");

const {MathBackendCPU} = 
    require("./node_modules/deeplearn/dist/math/backends/backend_cpu");

const {NDArrayMath} = 
    require("./node_modules/deeplearn/dist/math/math");
*/

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

        this.Wxi  = graph.variable(
            'Wxi',
            Array2D.randNormal([input_size, hidden_size])
        );
        this.Whi  = graph.variable(
            'Whi',
            Array2D.randNormal([hidden_size, hidden_size])
        );
        this.Wci  = graph.variable(
            'Wci',
            Array2D.randNormal([hidden_size, hidden_size])
        );
        this.bi = graph.variable(
            'bi',
            Array2D.randNormal([1, hidden_size])
        );
        this.biasReplicator = graph.constant(NDArray.ones([batch_size, 1]));

        this.x_check = graph.reshape(this.x, [batch_size, input_size]);
        this.m_check = graph.matmul(this.x, this.Wxi);
        this.h_check = graph.matmul(this.h_tm1, this.Whi);
        this.biasMultCheck = graph.matmul(this.biasReplicator, this.bi);

        let toSum = [
            graph.matmul(this.x, this.Wxi),
            graph.matmul(this.h_tm1, this.Whi),
            graph.matmul(this.h_tm1, this.Whi),
            graph.matmul(this.c_tm1, this.Wci),
            graph.matmul(this.biasReplicator, this.bi)
        ]

        this.add = graphSum({'toSum': toSum, 'graph': graph});

        if(nonlin === 'tanh'){
            this.i = graph.tanh(this.add);
        }

        // WARNING: placeholder are not included in the graph apparently so
        // evaluating them will throw an error !!!

        /* TEST */
        this.xt = graph.placeholder('xt', [4, 2]);
        this.multiplier = graph.variable(
            'multiplier', Array2D.randNormal([2, 4])
        );
        this.z = graph.matmul(this.xt, this.multiplier);
    }


    forward(session, x, c, h){

        /* TEST */
        let shuffledInputProviderBuilderA =
            new InCPUMemoryShuffledInputProviderBuilder([x]);
        let [xProviderA] = shuffledInputProviderBuilderA.getInputProviders();
        // Maps tensors to InputProviders.
        let xFeeder = {tensor: this.xt, data: xProviderA};
        let w = session.eval(this.z, [xFeeder]).dataSync();

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

        let val_x = session.eval(this.x_check, [xFeed]).dataSync();
        console.log("=====>", val_x);

        let val_m_check = session.eval(
            this.m_check, [xFeed]
        ).dataSync();

        let val_h_check = session.eval(
            this.h_check, [hFeed]
        ).dataSync();

        let val_add_check = session.eval(
            this.add, [xFeed, hFeed, cFeed]
        ).dataSync();

        let biasMultCheck = session.eval(
            this.biasMultCheck, []
        ).dataSync();

        console.log(val_add_check);
        console.log('bMM', biasMultCheck);
        console.log('bMM', biasMultCheck.shape);
        console.log('bMM', this.biasMultCheck.shape);

        /*
        val_i = session.eval(this.i, feedEntries);
        return(val_i);
        */
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


    lstm.forward(session, x, h, c);
}
